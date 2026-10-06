"""Render the external deterministic-Core8 Decision technical report.

This module is a presentation adapter, not an evaluator.  It consumes the
sealed summary written by the external three-model Decision evaluation,
verifies the bound results and evaluation manifest, and writes two
create-only report sources:

* ``technical_report.md`` for direct review; and
* ``artifact.json`` in the canonical Data Analytics report shape.

The adapter never pools the historical N=19 and post-cutoff N=12 panels.  It
also never converts the mechanically zero-filled three-class average for the
N=12 panel into an estimable headline: that panel has no hike support, so its
reported balanced accuracy averages cut and hold recall only.

HTML packaging is intentionally outside this module.  Once real results
exist, ``artifact.json`` can be passed to the Data Analytics build-report
portable HTML packager without maintaining a second presentation model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_external_deterministic_core8_n19_n12_v4_20260824"
)
DEFAULT_SUMMARY = DEFAULT_EVALUATION_ROOT / "report/summary.json"
DEFAULT_OUTPUT_DIR = DEFAULT_EVALUATION_ROOT / "portable_technical_report"
DEFAULT_SELECTION = ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_cp38_direct_grpo_checkpoint_probe_v2_batch8_20260814/selection.json"
)
DEFAULT_FINAL_STATUS = ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_cp450_final_test_v1_20260814/FINAL_STATUS.md"
)

SUMMARY_SCHEMA = "chk4-external-deterministic-core8-summary-v1"
EVALUATION_MANIFEST_SCHEMA = (
    "chk4-external-deterministic-core8-evaluation-manifest-v1"
)
REPORT_MANIFEST_SCHEMA = (
    "chk4-external-deterministic-core8-technical-report-manifest-v1"
)
REPORT_TITLE = "External Deterministic Core8 Decision Evaluation"

PANELS = ("historical_n19", "postcutoff_n12")
PANEL_N = {"historical_n19": 19, "postcutoff_n12": 12}
PANEL_CLASS_COUNTS = {
    "historical_n19": {"cut": 4, "hold": 10, "hike": 5},
    "postcutoff_n12": {"cut": 3, "hold": 9, "hike": 0},
}
DIRECTIONS = ("cut", "hold", "hike")
PREDICTIONS = (*DIRECTIONS, "invalid")
MODEL_ORDER = (
    "model_chk1_cp200",
    "model_chk3_sft_cp38",
    "model_chk3_grpo_cp450",
)
MODEL_DISPLAY = {
    "model_chk1_cp200": "Model chk-1 cp200",
    "model_chk3_sft_cp38": "Model chk-3 SFT cp38",
    "model_chk3_grpo_cp450": "Model chk-3 GRPO cp450",
}
MODEL_TRAINING_STATE = {
    "model_chk1_cp200": "analysis_sft",
    "model_chk3_sft_cp38": "decision_sft_pre_grpo",
    "model_chk3_grpo_cp450": (
        "decision_grpo_best_observed_exploratory_candidate"
    ),
}
EXPECTED_RESULT_ROWS = sum(PANEL_N.values()) * len(MODEL_ORDER)
INTEGRITY_ALGORITHM = "sha256(canonical-json-without-integrity)"
INPUT_CONTRACT_VERSION = "deterministic_core8_source_analysis_concatenation_v1"
HOLM_FAMILY = "within_panel_three_pairwise_model_comparisons"
HOLM_FAMILY_SIZE = 3


class ExternalDecisionReportError(RuntimeError):
    """A sealed input or report materialization failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalDecisionReportError(message)


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
        raise ExternalDecisionReportError(
            f"value cannot be encoded as canonical JSON: {exc}"
        ) from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalDecisionReportError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _verify_sealed_payload(
    value: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    _require(value.get("schema_version") == schema, f"{label} schema drift")
    integrity = value.get("integrity")
    _require(isinstance(integrity, Mapping), f"{label} integrity block is missing")
    _require(
        integrity.get("algorithm") == INTEGRITY_ALGORITHM,
        f"{label} integrity algorithm drift",
    )
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("integrity", None)
    expected = _sha256_text(_canonical(unsigned))
    _require(
        integrity.get("payload_sha256") == expected,
        f"{label} payload SHA-256 drift",
    )
    return copy.deepcopy(dict(value))


def _binding(path: Path) -> dict[str, Any]:
    unresolved = path.expanduser()
    _require(not unresolved.is_symlink(), f"bound source is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    _require(resolved.is_file(), f"bound source is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _intended_binding(staged_path: Path, final_path: Path) -> dict[str, Any]:
    value = _binding(staged_path)
    value["path"] = str(final_path.expanduser().resolve())
    return value


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ExternalDecisionReportError(
            f"portable source is outside repository root: {resolved}"
        ) from exc


def _finite_rate(value: Any, *, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{label} is invalid")
    return result


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    _require(
        not isinstance(value, bool) and isinstance(value, int) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return int(value)


def _close(observed: float, expected: float, *, label: str) -> None:
    _require(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{label} does not reconcile ({observed!r} != {expected!r})",
    )


def _verify_bound_results(summary: Mapping[str, Any], summary_path: Path) -> Path:
    record = summary.get("results")
    _require(isinstance(record, Mapping), "summary results binding is missing")
    relative = record.get("path")
    _require(isinstance(relative, str) and relative, "results path is invalid")
    _require(not Path(relative).is_absolute(), "results path must be evaluation-relative")
    evaluation_root = summary_path.resolve().parent.parent
    path = (evaluation_root / relative).resolve()
    try:
        path.relative_to(evaluation_root)
    except ValueError as exc:
        raise ExternalDecisionReportError("results path escapes evaluation root") from exc
    observed = _binding(path)
    _require(observed["bytes"] == record.get("bytes"), "results byte count drift")
    _require(observed["sha256"] == record.get("sha256"), "results SHA-256 drift")
    with path.open("rb") as handle:
        rows = sum(bool(line.strip()) for line in handle)
    _require(rows == EXPECTED_RESULT_ROWS, "results row count drift")
    _require(record.get("rows") == EXPECTED_RESULT_ROWS, "summary result-row drift")
    return path


def _exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(a_only, b_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _holm_adjusted(values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original row order."""

    _require(bool(values), "Holm family cannot be empty")
    normalized = [
        _finite_rate(value, label=f"Holm raw p[{index}]")
        for index, value in enumerate(values)
    ]
    family_size = len(normalized)
    ordered = sorted(range(family_size), key=lambda index: (normalized[index], index))
    result = [0.0] * family_size
    running = 0.0
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * normalized[original_index])
        running = max(running, candidate)
        result[original_index] = running
    return result


def _validate_score_block(
    block: Mapping[str, Any], *, panel: str, model: str
) -> dict[str, Any]:
    n = PANEL_N[panel]
    cases = _integer(block.get("cases"), label=f"{panel}/{model} cases", minimum=1)
    _require(cases == n, f"{panel}/{model} denominator drift")
    direction_correct = _integer(
        block.get("direction_correct"),
        label=f"{panel}/{model} direction correct",
    )
    _require(direction_correct <= n, f"{panel}/{model} correct count exceeds N")
    direction_accuracy = _finite_rate(
        block.get("direction_accuracy"),
        label=f"{panel}/{model} direction accuracy",
    )
    _close(
        direction_accuracy,
        direction_correct / n,
        label=f"{panel}/{model} direction accuracy",
    )

    per_class = block.get("per_class")
    matrix = block.get("confusion_matrix")
    _require(isinstance(per_class, Mapping), f"{panel}/{model} recalls are missing")
    _require(isinstance(matrix, Mapping), f"{panel}/{model} matrix is missing")
    _require(set(per_class) == set(DIRECTIONS), f"{panel}/{model} recall labels drift")
    _require(set(matrix) == set(DIRECTIONS), f"{panel}/{model} matrix rows drift")
    normalized_classes: dict[str, Any] = {}
    normalized_matrix: dict[str, dict[str, int]] = {}
    for direction in DIRECTIONS:
        class_block = per_class[direction]
        row = matrix[direction]
        _require(
            isinstance(class_block, Mapping),
            f"{panel}/{model}/{direction} recall block is invalid",
        )
        _require(
            isinstance(row, Mapping) and set(row) == set(PREDICTIONS),
            f"{panel}/{model}/{direction} matrix columns drift",
        )
        normalized_row = {
            prediction: _integer(
                row[prediction],
                label=f"{panel}/{model}/{direction}->{prediction}",
            )
            for prediction in PREDICTIONS
        }
        support = _integer(
            class_block.get("support"),
            label=f"{panel}/{model}/{direction} support",
        )
        correct = _integer(
            class_block.get("correct"),
            label=f"{panel}/{model}/{direction} correct",
        )
        _require(
            support == PANEL_CLASS_COUNTS[panel][direction],
            f"{panel}/{model}/{direction} support drift",
        )
        _require(sum(normalized_row.values()) == support, f"{panel}/{model}/{direction} matrix row does not close")
        _require(normalized_row[direction] == correct, f"{panel}/{model}/{direction} diagonal drift")
        recall_value = class_block.get("recall")
        if support == 0:
            _require(recall_value is None, f"{panel}/{model}/{direction} recall must be null")
            recall: float | None = None
        else:
            recall = _finite_rate(
                recall_value,
                label=f"{panel}/{model}/{direction} recall",
            )
            _close(
                recall,
                correct / support,
                label=f"{panel}/{model}/{direction} recall",
            )
        normalized_classes[direction] = {
            "support": support,
            "correct": correct,
            "recall": recall,
        }
        normalized_matrix[direction] = normalized_row

    matrix_correct = sum(normalized_matrix[value][value] for value in DIRECTIONS)
    _require(matrix_correct == direction_correct, f"{panel}/{model} matrix accuracy drift")
    supported = [
        direction
        for direction in DIRECTIONS
        if normalized_classes[direction]["support"] > 0
    ]
    _require(block.get("supported_classes") == supported, f"{panel}/{model} supported-class drift")
    supported_ba = sum(
        float(normalized_classes[value]["recall"]) for value in supported
    ) / len(supported)
    mechanical_zero_insertion = sum(
        float(normalized_classes[value]["recall"] or 0.0) for value in DIRECTIONS
    ) / len(DIRECTIONS)
    observed_supported = _finite_rate(
        block.get("supported_class_balanced_accuracy"),
        label=f"{panel}/{model} supported-class BA",
    )
    _close(observed_supported, supported_ba, label=f"{panel}/{model} supported BA")
    observed_mechanical = _finite_rate(
        block.get("mechanical_zero_insertion_balanced_accuracy"),
        label=f"{panel}/{model} mechanical zero-insertion BA",
    )
    _close(
        observed_mechanical,
        mechanical_zero_insertion,
        label=f"{panel}/{model} mechanical zero-insertion BA",
    )

    fixed_value = block.get("fixed_three_class_balanced_accuracy")
    if len(supported) == len(DIRECTIONS):
        observed_fixed: float | None = _finite_rate(
            fixed_value,
            label=f"{panel}/{model} fixed-three-class BA",
        )
        _close(
            observed_fixed,
            mechanical_zero_insertion,
            label=f"{panel}/{model} fixed BA",
        )
        _require(
            block.get("mechanical_zero_insertion_is_estimand") is True,
            f"{panel}/{model} fixed-three estimand flag drift",
        )
    else:
        _require(
            fixed_value is None,
            f"{panel}/{model} fixed-three-class BA must be null without all classes",
        )
        _require(
            block.get("mechanical_zero_insertion_is_estimand") is False,
            f"{panel}/{model} zero-insertion value was misclassified as an estimand",
        )
        observed_fixed = None

    expected_definition = "supported_classes" if panel == "postcutoff_n12" else "fixed_three_class"
    _require(
        block.get("primary_balanced_accuracy_definition") == expected_definition,
        f"{panel}/{model} primary BA definition drift",
    )
    primary = _finite_rate(
        block.get("primary_balanced_accuracy"),
        label=f"{panel}/{model} primary BA",
    )
    expected_primary = (
        supported_ba if panel == "postcutoff_n12" else observed_fixed
    )
    _require(expected_primary is not None, f"{panel}/{model} primary BA is undefined")
    _close(primary, expected_primary, label=f"{panel}/{model} primary BA")
    return {
        "cases": cases,
        "direction_correct": direction_correct,
        "direction_accuracy": direction_accuracy,
        "per_class": normalized_classes,
        "confusion_matrix": normalized_matrix,
        "supported_classes": supported,
        "supported_class_balanced_accuracy": supported_ba,
        "fixed_three_class_balanced_accuracy": observed_fixed,
        "mechanical_zero_insertion_balanced_accuracy": mechanical_zero_insertion,
        "mechanical_zero_insertion_is_estimand": observed_fixed is not None,
        "primary_balanced_accuracy_definition": expected_definition,
        "primary_balanced_accuracy": expected_primary,
    }


def _validate_pairwise(
    rows: Any, *, panel: str, model_correct: Mapping[str, int]
) -> list[dict[str, Any]]:
    _require(isinstance(rows, list), f"{panel} McNemar rows are missing")
    expected_pairs = {frozenset(pair) for pair in itertools.combinations(MODEL_ORDER, 2)}
    observed_pairs: set[frozenset[str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        _require(isinstance(raw, Mapping), f"{panel} McNemar row {index} is invalid")
        model_a = str(raw.get("model_a") or "")
        model_b = str(raw.get("model_b") or "")
        pair = frozenset((model_a, model_b))
        _require(len(pair) == 2 and pair in expected_pairs, f"{panel} McNemar pair drift")
        _require(pair not in observed_pairs, f"{panel} duplicate McNemar pair")
        observed_pairs.add(pair)
        _require(raw.get("model_a_display") == MODEL_DISPLAY[model_a], f"{panel} model-A display drift")
        _require(raw.get("model_b_display") == MODEL_DISPLAY[model_b], f"{panel} model-B display drift")
        counts = {
            key: _integer(raw.get(key), label=f"{panel}/{model_a}/{model_b}/{key}")
            for key in (
                "a_only_correct",
                "b_only_correct",
                "both_correct",
                "neither_correct",
                "discordant_pairs",
            )
        }
        _require(
            counts["a_only_correct"] + counts["both_correct"] == model_correct[model_a],
            f"{panel} model-A McNemar margin drift",
        )
        _require(
            counts["b_only_correct"] + counts["both_correct"] == model_correct[model_b],
            f"{panel} model-B McNemar margin drift",
        )
        _require(
            counts["a_only_correct"]
            + counts["b_only_correct"]
            + counts["both_correct"]
            + counts["neither_correct"]
            == PANEL_N[panel],
            f"{panel} McNemar table does not close",
        )
        _require(
            counts["discordant_pairs"]
            == counts["a_only_correct"] + counts["b_only_correct"],
            f"{panel} discordant-pair count drift",
        )
        _require(raw.get("test") == "exact_two_sided_mcnemar", f"{panel} test-name drift")
        p_value_raw = _finite_rate(
            raw.get("p_value_raw"), label=f"{panel} raw McNemar p"
        )
        expected_p = _exact_mcnemar_p(
            counts["a_only_correct"], counts["b_only_correct"]
        )
        _close(p_value_raw, expected_p, label=f"{panel} raw McNemar p")
        p_value_holm = _finite_rate(
            raw.get("p_value_holm"), label=f"{panel} Holm-adjusted McNemar p"
        )
        _require(
            raw.get("holm_family") == HOLM_FAMILY,
            f"{panel} Holm family drift",
        )
        _require(
            raw.get("holm_family_panel") == panel,
            f"{panel} Holm family-panel drift",
        )
        _require(
            raw.get("holm_family_size") == HOLM_FAMILY_SIZE,
            f"{panel} Holm family-size drift",
        )
        normalized.append(
            {
                "model_a": model_a,
                "model_a_display": MODEL_DISPLAY[model_a],
                "model_b": model_b,
                "model_b_display": MODEL_DISPLAY[model_b],
                **counts,
                "test": "exact_two_sided_mcnemar",
                "p_value_raw": p_value_raw,
                "p_value_holm": p_value_holm,
                "holm_family": HOLM_FAMILY,
                "holm_family_panel": panel,
                "holm_family_size": HOLM_FAMILY_SIZE,
            }
        )
    _require(observed_pairs == expected_pairs, f"{panel} McNemar coverage drift")
    normalized.sort(
        key=lambda row: (
            MODEL_ORDER.index(str(row["model_a"])),
            MODEL_ORDER.index(str(row["model_b"])),
        )
    )
    expected_adjusted = _holm_adjusted(
        [float(row["p_value_raw"]) for row in normalized]
    )
    for row, expected_holm in zip(normalized, expected_adjusted, strict=True):
        _close(
            float(row["p_value_holm"]),
            expected_holm,
            label=f"{panel} Holm-adjusted McNemar p",
        )
    return normalized


def _validate_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = _verify_sealed_payload(
        value, schema=SUMMARY_SCHEMA, label="external Decision summary"
    )
    _require(summary.get("status") == "complete", "summary is not complete")
    manifest_sha = summary.get("evaluation_manifest_sha256")
    _require(
        isinstance(manifest_sha, str)
        and len(manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha),
        "evaluation-manifest SHA-256 is invalid",
    )
    _require(summary.get("pooled_result") is None, "pooled result is prohibited")
    panels = summary.get("panels")
    _require(isinstance(panels, Mapping) and set(panels) == set(PANELS), "panel inventory drift")
    normalized_panels: dict[str, Any] = {}
    for panel in PANELS:
        raw_panel = panels[panel]
        _require(isinstance(raw_panel, Mapping), f"{panel} is invalid")
        population = raw_panel.get("population")
        _require(isinstance(population, Mapping), f"{panel} population is missing")
        _require(population.get("meetings") == PANEL_N[panel], f"{panel} meeting N drift")
        raw_counts = population.get("class_counts")
        _require(isinstance(raw_counts, Mapping), f"{panel} class counts are missing")
        observed_counts = {
            direction: _integer(raw_counts.get(direction, 0), label=f"{panel}/{direction} count")
            for direction in DIRECTIONS
        }
        _require(observed_counts == PANEL_CLASS_COUNTS[panel], f"{panel} class distribution drift")
        raw_models = raw_panel.get("models")
        _require(isinstance(raw_models, Mapping) and set(raw_models) == set(MODEL_ORDER), f"{panel} model inventory drift")
        models = {
            model: _validate_score_block(raw_models[model], panel=panel, model=model)
            for model in MODEL_ORDER
        }
        pairwise = _validate_pairwise(
            raw_panel.get("pairwise"),
            panel=panel,
            model_correct={model: models[model]["direction_correct"] for model in MODEL_ORDER},
        )
        normalized_panels[panel] = {
            "population": {
                "meetings": PANEL_N[panel],
                "class_counts": observed_counts,
            },
            "models": models,
            "pairwise": pairwise,
        }
    interpretation = summary.get("interpretation")
    _require(isinstance(interpretation, Mapping), "summary interpretation is missing")
    input_contract = interpretation.get("input_contract")
    _require(isinstance(input_contract, Mapping), "input contract is missing")
    _require(input_contract.get("version") == INPUT_CONTRACT_VERSION, "input-contract version drift")
    _require(input_contract.get("blocks_per_meeting") == 8, "Core8 block count drift")
    _require(input_contract.get("meeting_identity_in_prompt") is False, "meeting identity contract drift")
    _require(input_contract.get("gold_in_prompt") is False, "gold-label contract drift")
    _require(input_contract.get("official_text_in_prompt") is False, "official-text contract drift")
    _require(input_contract.get("distinct_from_teacher_compressed_n13") is True, "N13 distinction is missing")
    _require(input_contract.get("byte_comparable_to_teacher_compressed_n13") is False, "N13 byte-comparability drift")
    summary["panels"] = normalized_panels
    return summary


def _validate_evaluation_manifest(
    summary: Mapping[str, Any], summary_path: Path
) -> tuple[dict[str, Any], Path]:
    path = summary_path.resolve().parent.parent / "evaluation_manifest.json"
    _require(
        path.is_file() and not path.is_symlink(),
        "evaluation manifest is missing or unsafe",
    )
    _require(
        _sha256_file(path) == summary["evaluation_manifest_sha256"],
        "evaluation-manifest file SHA-256 drift",
    )
    manifest = _verify_sealed_payload(
        _read_json(path, label="evaluation manifest"),
        schema=EVALUATION_MANIFEST_SCHEMA,
        label="evaluation manifest",
    )
    _require(manifest.get("status") == "prepared", "evaluation manifest status drift")
    _require(
        manifest.get("purpose")
        == "separate_panel_greedy_decision_evaluation_without_retraining",
        "evaluation purpose drift",
    )
    population = manifest.get("population")
    _require(isinstance(population, Mapping), "manifest population is missing")
    _require(population.get("training_performed") is False, "evaluation performed training")
    _require(population.get("direct_decision_release_overlap") == 0, "Decision-release overlap drift")
    _require(population.get("panels") == PANEL_N, "manifest panel counts drift")
    input_contract = manifest.get("input_contract")
    _require(
        input_contract == summary["interpretation"]["input_contract"],
        "summary/manifest input contract mismatch",
    )
    generation = manifest.get("generation")
    _require(isinstance(generation, Mapping), "generation contract is missing")
    _require(generation.get("mode") == "greedy", "generation mode drift")
    _require(generation.get("completions_per_meeting_model") == 1, "generation K drift")
    _require(generation.get("expected_rows") == EXPECTED_RESULT_ROWS, "generation row contract drift")
    reporting = manifest.get("reporting")
    _require(isinstance(reporting, Mapping), "reporting contract is missing")
    _require(reporting.get("panels_reported_separately") is True, "separate-panel contract drift")
    _require(reporting.get("pooled_result_prohibited") is True, "pooling prohibition drift")
    _require(
        reporting.get("pairwise_test")
        == "exact_two_sided_mcnemar_direction_correctness",
        "pairwise-test contract drift",
    )
    _require(
        reporting.get("pairwise_multiplicity")
        == (
            "Holm adjustment within each panel's family of three "
            "model-pair comparisons"
        ),
        "pairwise multiplicity contract drift",
    )
    models = manifest.get("models")
    _require(isinstance(models, list) and len(models) == len(MODEL_ORDER), "manifest model inventory drift")
    by_label = {str(model.get("label")): model for model in models if isinstance(model, Mapping)}
    _require(set(by_label) == set(MODEL_ORDER), "manifest model labels drift")
    for model in MODEL_ORDER:
        _require(by_label[model].get("display_name") == MODEL_DISPLAY[model], f"{model} display drift")
        _require(by_label[model].get("training_state") == MODEL_TRAINING_STATE[model], f"{model} training-state drift")
    return manifest, path


def _validate_checkpoint_governance(
    selection_path: Path, final_status_path: Path
) -> tuple[dict[str, Any], str]:
    selection = _read_json(selection_path, label="cp450 checkpoint selection")
    _require(selection.get("selected_checkpoint_step") is None, "cp450 was unexpectedly selected")
    _require(selection.get("best_observed_checkpoint_step") == 450, "best-observed checkpoint drift")
    _require(
        selection.get("status") == "provisional_no_candidate_passed_all_gates",
        "checkpoint-selection status drift",
    )
    _require(final_status_path.is_file() and not final_status_path.is_symlink(), "cp450 final status is missing or unsafe")
    text = final_status_path.read_text(encoding="utf-8")
    _require("`quality_not_demonstrated`" in text, "cp450 quality verdict drift")
    _require("not automatically gate-selected" in text, "cp450 override disclosure drift")
    return selection, text


def load_report_model(
    summary_path: Path,
    *,
    selection_path: Path = DEFAULT_SELECTION,
    final_status_path: Path = DEFAULT_FINAL_STATUS,
) -> dict[str, Any]:
    """Load and independently reconcile every report-facing input."""

    summary_path = summary_path.expanduser().resolve()
    summary = _validate_summary(_read_json(summary_path, label="external Decision summary"))
    results_path = _verify_bound_results(summary, summary_path)
    evaluation_manifest, evaluation_manifest_path = _validate_evaluation_manifest(
        summary, summary_path
    )
    selection, final_status = _validate_checkpoint_governance(
        selection_path.expanduser().resolve(), final_status_path.expanduser().resolve()
    )
    return {
        "summary": summary,
        "evaluation_manifest": evaluation_manifest,
        "checkpoint_selection": selection,
        "final_status_text": final_status,
        "source_bindings": {
            "summary": _binding(summary_path),
            "results": _binding(results_path),
            "evaluation_manifest": _binding(evaluation_manifest_path),
            "checkpoint_selection": _binding(selection_path),
            "cp450_final_status": _binding(final_status_path),
        },
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _fraction(correct: int, support: int, recall: float | None) -> str:
    if support == 0 or recall is None:
        return "Not estimable (support = 0)"
    return f"{correct}/{support} ({_pct(recall)})"


def _metric_rows(model: Mapping[str, Any], panel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        score = model["summary"]["panels"][panel]["models"][model_id]
        classes = score["per_class"]
        if panel == "historical_n19":
            primary_label = "Fixed-three-class BA"
            primary_display = _pct(score["fixed_three_class_balanced_accuracy"])
            fixed_display = primary_display
        else:
            primary_label = "Supported-class BA (cut/hold)"
            primary_display = _pct(score["supported_class_balanced_accuracy"])
            fixed_display = "Not estimable (no hike support)"
        rows.append(
            {
                "model_id": model_id,
                "model": MODEL_DISPLAY[model_id],
                "direction_correct": score["direction_correct"],
                "cases": score["cases"],
                "direction_accuracy": score["direction_accuracy"],
                "direction_accuracy_display": (
                    f"{score['direction_correct']}/{score['cases']} "
                    f"({_pct(score['direction_accuracy'])})"
                ),
                "cut_recall": classes["cut"]["recall"],
                "cut_recall_display": _fraction(
                    classes["cut"]["correct"],
                    classes["cut"]["support"],
                    classes["cut"]["recall"],
                ),
                "hold_recall": classes["hold"]["recall"],
                "hold_recall_display": _fraction(
                    classes["hold"]["correct"],
                    classes["hold"]["support"],
                    classes["hold"]["recall"],
                ),
                "hike_recall": classes["hike"]["recall"],
                "hike_recall_display": _fraction(
                    classes["hike"]["correct"],
                    classes["hike"]["support"],
                    classes["hike"]["recall"],
                ),
                "primary_balanced_accuracy": score["primary_balanced_accuracy"],
                "primary_balanced_accuracy_label": primary_label,
                "primary_balanced_accuracy_display": primary_display,
                "fixed_three_class_balanced_accuracy_display": fixed_display,
            }
        )
    return rows


def _pairwise_rows(model: Mapping[str, Any], panel: str) -> list[dict[str, Any]]:
    result = []
    for row in model["summary"]["panels"][panel]["pairwise"]:
        result.append(
            {
                **copy.deepcopy(row),
                "p_value_raw_display": f"{float(row['p_value_raw']):.6f}",
                "p_value_holm_display": f"{float(row['p_value_holm']):.6f}",
                "holm_family_display": (
                    f"{panel}: three within-panel model comparisons"
                ),
            }
        )
    return result


def _support_rows(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for panel in PANELS:
        population = model["summary"]["panels"][panel]["population"]
        result.append(
            {
                "panel": "Historical missing-meeting panel" if panel == "historical_n19" else "Post-cutoff regular-meeting panel",
                "panel_id": panel,
                "meetings": population["meetings"],
                "cut": population["class_counts"]["cut"],
                "hold": population["class_counts"]["hold"],
                "hike": population["class_counts"]["hike"],
                "balanced_accuracy_contract": (
                    "Fixed-three-class" if panel == "historical_n19" else "Supported classes only (cut and hold)"
                ),
            }
        )
    return result


def _chart_support_rows(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    populations = {
        row["panel_id"]: row for row in _support_rows(model)
    }
    rows: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        for panel in PANELS:
            population = populations[panel]
            count = int(population[direction])
            denominator = int(population["meetings"])
            panel_label = str(population["panel"])
            rows.append(
                {
                    "panel_id": panel,
                    "panel": panel_label,
                    "direction": direction.title(),
                    "meeting_count": count,
                    "panel_denominator": denominator,
                    "within_panel_share": count / denominator,
                    "within_panel_share_display": (
                        f"{count}/{denominator} ({_pct(count / denominator)})"
                    ),
                    "bar_label": f"{panel_label}: {count}",
                }
            )
    return rows


def _leader_text(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[str, float]:
    best = max(float(row[field]) for row in rows)
    names = [str(row["model"]) for row in rows if math.isclose(float(row[field]), best, rel_tol=0.0, abs_tol=1e-12)]
    return ", ".join(names), best


def _panel_summary_sentence(rows: Sequence[Mapping[str, Any]], *, panel: str) -> str:
    direction_names, direction_value = _leader_text(rows, "direction_accuracy")
    ba_names, ba_value = _leader_text(rows, "primary_balanced_accuracy")
    ba_label = "fixed-three-class balanced accuracy" if panel == "historical_n19" else "supported-class cut/hold balanced accuracy"
    return (
        f"The highest observed direction accuracy is **{_pct(direction_value)}** "
        f"({direction_names}); the highest observed {ba_label} is "
        f"**{_pct(ba_value)}** ({ba_names}). These are descriptive maxima, not "
        "evidence of population-level model superiority."
    )


def _mcnemar_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    _require(
        len(rows) == HOLM_FAMILY_SIZE,
        "paper-facing McNemar summary requires one three-comparison panel family",
    )
    raw_values = [float(row["p_value_raw"]) for row in rows]
    adjusted_values = [float(row["p_value_holm"]) for row in rows]
    significant = sum(value < 0.05 for value in adjusted_values)
    return (
        "Raw exact two-sided McNemar p-values range from "
        f"{min(raw_values):.6f} to {max(raw_values):.6f}; their independently "
        "computed within-panel Holm-adjusted values range from "
        f"{min(adjusted_values):.6f} to {max(adjusted_values):.6f}. "
        f"{significant} of the three paired comparisons remains below 0.05 "
        "after adjustment. Paper-facing inference uses the Holm-adjusted values. "
        "These small-panel tests are descriptive, and a non-significant value "
        "does not establish equivalence."
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def build_markdown_report(model: Mapping[str, Any]) -> str:
    """Create the English technical report from the validated report model."""

    historical = _metric_rows(model, "historical_n19")
    postcutoff = _metric_rows(model, "postcutoff_n12")
    historical_pairs = _pairwise_rows(model, "historical_n19")
    postcutoff_pairs = _pairwise_rows(model, "postcutoff_n12")
    support = _support_rows(model)
    topic_order = model["evaluation_manifest"]["input_contract"]["topic_order"]

    def metrics_table(rows: Sequence[Mapping[str, Any]], *, post: bool) -> str:
        headers = [
            "Model/state",
            "Direction accuracy",
            "Cut recall",
            "Hold recall",
            "Hike recall",
            "Supported-class BA" if post else "Fixed-three-class BA",
        ]
        values = [
            [
                str(row["model"]),
                str(row["direction_accuracy_display"]),
                str(row["cut_recall_display"]),
                str(row["hold_recall_display"]),
                str(row["hike_recall_display"]),
                str(row["primary_balanced_accuracy_display"]),
            ]
            for row in rows
        ]
        return _markdown_table(headers, values)

    def pairs_table(rows: Sequence[Mapping[str, Any]]) -> str:
        return _markdown_table(
            [
                "Model A",
                "Model B",
                "A only correct",
                "B only correct",
                "Both correct",
                "Neither correct",
                "Discordant",
                "Exact McNemar p (raw)",
                "Holm-adjusted p",
            ],
            [
                [
                    str(row["model_a_display"]),
                    str(row["model_b_display"]),
                    str(row["a_only_correct"]),
                    str(row["b_only_correct"]),
                    str(row["both_correct"]),
                    str(row["neither_correct"]),
                    str(row["discordant_pairs"]),
                    str(row["p_value_raw_display"]),
                    str(row["p_value_holm_display"]),
                ]
                for row in rows
            ],
        )

    support_table = _markdown_table(
        ["Panel", "N", "Cut", "Hold", "Hike", "Balanced-accuracy contract"],
        [
            [
                str(row["panel"]),
                str(row["meetings"]),
                str(row["cut"]),
                str(row["hold"]),
                str(row["hike"]),
                str(row["balanced_accuracy_contract"]),
            ]
            for row in support
        ],
    )
    topics = ", ".join(str(value) for value in topic_order)
    return "\n\n".join(
        [
            f"# {REPORT_TITLE}",
            "## Technical summary\n\n"
            + _panel_summary_sentence(historical, panel="historical_n19")
            + "\n\n"
            + _panel_summary_sentence(postcutoff, panel="postcutoff_n12")
            + "\n\nThe N=19 and N=12 panels are kept separate. The evaluation "
            "used frozen checkpoints and performed no retraining. Model chk-3 "
            "GRPO cp450 remains a best-observed exploratory evaluation candidate, "
            "not a selected, promoted, or final model.",
            "## Historical N=19: all three decision classes are observed\n\n"
            + _panel_summary_sentence(historical, panel="historical_n19")
            + "\n\n"
            + metrics_table(historical, post=False),
            "### Exact paired direction comparisons for N=19\n\n"
            + _mcnemar_summary(historical_pairs)
            + "\n\n"
            + pairs_table(historical_pairs),
            "## Post-cutoff N=12: hike performance is not estimable\n\n"
            + _panel_summary_sentence(postcutoff, panel="postcutoff_n12")
            + "\n\nThe panel contains three cuts, nine holds, and no hikes. "
            "Accordingly, balanced accuracy is the mean of cut and hold recall. "
            "Fixed-three-class balanced accuracy is **not estimable**; inserting "
            "a zero for the absent hike class is only a mechanical calculation "
            "and is not reported as a performance estimate.\n\n"
            + metrics_table(postcutoff, post=True),
            "### Exact paired direction comparisons for N=12\n\n"
            + _mcnemar_summary(postcutoff_pairs)
            + "\n\n"
            + pairs_table(postcutoff_pairs),
            "## Scope and metric definitions\n\n"
            + support_table
            + "\n\nDirection accuracy is the number of meetings with the correct "
            "cut/hold/hike direction divided by all meetings in that panel; invalid "
            "outputs remain in the denominator and are incorrect. Class recall is "
            "the number correctly predicted within a true class divided by that "
            "class's support. Historical fixed-three-class balanced accuracy is "
            "the arithmetic mean of cut, hold, and hike recall. Post-cutoff "
            "supported-class balanced accuracy is the arithmetic mean of cut and "
            "hold recall only.",
            "## Experimental design and input contract\n\n"
            "For every meeting, the evaluator concatenates exactly eight "
            f"deterministic D-1 source-analysis blocks in this fixed order: {topics}. "
            "Meeting identity, the gold action, and official policy text are absent "
            "from the model prompt; official documents are used only to establish "
            "labels. Each frozen model produces one greedy completion per meeting. "
            "No model is trained or updated by this workflow.\n\n"
            "This direct Core8-concatenation contract differs from the "
            "teacher-compressed meeting brief used in the already-opened N=13 "
            "Decision diagnostic. The inputs are not byte-comparable, and neither "
            "external panel is pooled with N=13 or with the other external panel.",
            "## Exact McNemar methodology\n\n"
            "For each model pair, correctness is paired by meeting. Let b be the "
            "number correct only for Model A and c the number correct only for "
            "Model B. Conditional on b+c discordant meetings, the reported exact "
            "two-sided p-value is `min(1, 2 * P[Binomial(b+c, 0.5) <= min(b,c)])`. "
            "Within each panel, the three model-pair raw p-values form one Holm "
            "family and are adjusted by the step-down procedure. The two panels "
            "are separate families: no p-values are pooled or adjusted across "
            "panels. Both values are shown for auditability, while paper-facing "
            "inference uses the Holm-adjusted p-value. With only 19 or 12 meetings, "
            "these tests are a sensitivity diagnostic rather than a strong ranking "
            "device.",
            "## Checkpoint governance remains unresolved for GRPO cp450\n\n"
            "The checkpoint-selection record identifies step 450 only as the "
            "best-observed exploratory candidate and keeps "
            "`selected_checkpoint_step = null` with status "
            "`provisional_no_candidate_passed_all_gates`. The sealed N=13 test "
            "records `quality_not_demonstrated`. These external diagnostics do not "
            "retroactively select, promote, or finalize cp450.",
            "## Limitations and interpretation boundaries\n\n"
            "- The historical N=19 meetings were excluded from direct Decision "
            "training exposure, but their public history may have appeared in the "
            "base model's pretraining data; this is not a pretraining-independent "
            "test.\n"
            "- The post-cutoff N=12 labels were outside the Decision training-data "
            "endpoint, but the predictions were generated after the decisions were "
            "public. The panel is retrospective, not prospectively sealed.\n"
            "- The N=12 panel contains no hike, so three-class performance cannot be "
            "estimated.\n"
            "- Small class counts make point estimates and McNemar p-values unstable. "
            "A non-significant comparison does not establish equivalence.\n"
            "- Results compare a deterministic Core8 input contract; they do not "
            "isolate the effect of SFT or GRPO from input-construction or checkpoint "
            "selection choices.\n"
            "- No N=31 aggregate and no pooled-with-N=13 estimate is authorized.",
            "## Why performance evidence remains table-first\n\n"
            "Each panel has only three model rows, while exact numerators, "
            "denominators, absent-class support, and McNemar discordant counts are "
            "essential to interpretation. Performance results therefore remain in "
            "small exact tables. The portable artifact includes only one minimal "
            "grouped bar chart: class support by panel, which makes the missing N=12 "
            "hike class visible without graphing model performance.",
            "## Recommended next steps\n\n"
            "1. Report these panels as separate retrospective diagnostics and keep "
            "the opened teacher-compressed N=13 result separate.\n"
            "2. Begin a genuinely prospective extension by sealing model outputs "
            "before each future FOMC decision.\n"
            "3. Reassess GRPO only after a panel with all three directions and a "
            "predeclared checkpoint rule is available; do not infer cp450 promotion "
            "from these small external panels.",
            "## Further questions\n\n"
            "Will the ordering of cp38 and cp450 persist under prospectively sealed "
            "meetings, a panel containing new hikes, and the original "
            "teacher-compressed input contract? How sensitive are conclusions to "
            "one-completion greedy decoding versus a predeclared stochastic "
            "replicate design?",
        ]
    ) + "\n"


def _source(
    *,
    source_id: str,
    label: str,
    binding: Mapping[str, Any],
    description: str,
    json_query: bool = False,
    executed_at: str | None = None,
) -> dict[str, Any]:
    relative_path = _repo_relative(Path(str(binding["path"])))
    result: dict[str, Any] = {
        "id": source_id,
        "label": label,
        "path": relative_path,
        "description": description,
        "sha256": str(binding["sha256"]),
    }
    if json_query:
        escaped_path = relative_path.replace("'", "''")
        result["query"] = {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{escaped_path}')",
            "description": (
                "Reads the sealed summary. Reader-facing rows are bounded, "
                "deterministic projections that preserve the sealed panel, model, "
                "class-support, and paired-test values."
            ),
            "tables_used": [relative_path],
            "filters": {
                "panels": list(PANELS),
                "models": list(MODEL_ORDER),
                "pooled_result": "prohibited",
            },
            "metric_definitions": {
                "direction_accuracy": "direction_correct divided by panel meetings",
                "class_recall": "class-correct predictions divided by true-class support",
                "historical_balanced_accuracy": "mean cut, hold, and hike recall",
                "postcutoff_balanced_accuracy": "mean cut and hold recall; hike absent",
                "class_support": "number of official meeting labels in each direction",
                "mcnemar_p_raw": (
                    "exact two-sided paired direction-correctness p-value"
                ),
                "mcnemar_p_holm": (
                    "Holm step-down adjustment over the three model-pair raw "
                    "p-values within the same panel; used for paper-facing inference"
                ),
                "mcnemar_family_scope": (
                    "one independent family per panel; N=19 and N=12 are not pooled"
                ),
            },
            **({"executed_at": executed_at} if executed_at else {}),
        }
    return result


def build_canonical_artifact(model: Mapping[str, Any]) -> dict[str, Any]:
    """Build a table-first canonical report artifact without packaging HTML."""

    historical = _metric_rows(model, "historical_n19")
    postcutoff = _metric_rows(model, "postcutoff_n12")
    historical_pairs = _pairwise_rows(model, "historical_n19")
    postcutoff_pairs = _pairwise_rows(model, "postcutoff_n12")
    support = _support_rows(model)
    chart_support = _chart_support_rows(model)
    bindings = model["source_bindings"]
    generated_at = str(model["summary"]["created_at_utc"])
    sources = [
        _source(
            source_id="external_summary_source",
            label="Sealed external Decision evaluation summary",
            binding=bindings["summary"],
            description="Authoritative panel metrics and exact paired comparisons.",
            json_query=True,
            executed_at=generated_at,
        ),
        _source(
            source_id="evaluation_manifest_source",
            label="Sealed external Decision evaluation manifest",
            binding=bindings["evaluation_manifest"],
            description="Frozen models, populations, input contract, and no-retraining declaration.",
        ),
        _source(
            source_id="checkpoint_selection_source",
            label="Model chk-3 GRPO checkpoint-selection record",
            binding=bindings["checkpoint_selection"],
            description="Records best-observed step 450 and null formal selection.",
        ),
        _source(
            source_id="cp450_final_status_source",
            label="Sealed cp450 N=13 final status",
            binding=bindings["cp450_final_status"],
            description="Records the quality_not_demonstrated verdict.",
        ),
    ]
    metric_columns = [
        {"field": "model", "label": "Model/state", "type": "text"},
        {"field": "direction_accuracy_display", "label": "Direction accuracy", "type": "text"},
        {"field": "cut_recall_display", "label": "Cut recall", "type": "text"},
        {"field": "hold_recall_display", "label": "Hold recall", "type": "text"},
        {"field": "hike_recall_display", "label": "Hike recall", "type": "text"},
    ]
    historical_metric_columns = [
        *metric_columns,
        {
            "field": "primary_balanced_accuracy_display",
            "label": "Fixed-three-class BA",
            "type": "text",
        },
    ]
    postcutoff_metric_columns = [
        *metric_columns,
        {
            "field": "primary_balanced_accuracy_display",
            "label": "Supported-class BA (cut/hold)",
            "type": "text",
        },
        {
            "field": "fixed_three_class_balanced_accuracy_display",
            "label": "Fixed-three-class BA",
            "type": "text",
        },
    ]
    pair_columns = [
        {"field": "model_a_display", "label": "Model A", "type": "text"},
        {"field": "model_b_display", "label": "Model B", "type": "text"},
        {"field": "a_only_correct", "label": "A only correct", "format": "number"},
        {"field": "b_only_correct", "label": "B only correct", "format": "number"},
        {"field": "both_correct", "label": "Both correct", "format": "number"},
        {"field": "neither_correct", "label": "Neither correct", "format": "number"},
        {"field": "discordant_pairs", "label": "Discordant", "format": "number"},
        {
            "field": "p_value_raw_display",
            "label": "Exact two-sided p (raw)",
            "type": "text",
        },
        {
            "field": "p_value_holm_display",
            "label": "Holm-adjusted p",
            "type": "text",
        },
    ]
    charts = [
        {
            "id": "class_support_chart",
            "title": "Decision-class support by evaluation panel",
            "subtitle": (
                "Historical N=19: 4 cut, 10 hold, 5 hike; post-cutoff N=12: "
                "3 cut, 9 hold, 0 hike."
            ),
            "intent": "comparison",
            "question": (
                "Why do the two panels require different balanced-accuracy "
                "definitions?"
            ),
            "rationale": (
                "A single grouped count chart makes the absent post-cutoff hike "
                "class immediately visible; exact counts and denominators remain "
                "in the adjacent table."
            ),
            "comparisonContext": {
                "baseline": "zero meetings",
                "denominator": "Historical N=19; post-cutoff N=12",
                "grain": "one official FOMC meeting",
                "normalization": "raw class counts; no normalization",
                "semanticFamily": "decision-class support",
                "unit": "meetings",
            },
            "type": "bar",
            "dataset": "class_support_chart",
            "sourceId": "external_summary_source",
            "encodings": {
                "x": {
                    "field": "direction",
                    "type": "ordinal",
                    "label": "Official decision direction",
                },
                "y": {
                    "field": "meeting_count",
                    "type": "quantitative",
                    "label": "Meetings",
                    "unit": "meetings",
                },
                "color": {
                    "field": "panel",
                    "type": "nominal",
                    "label": "Evaluation panel",
                },
                "label": {
                    "field": "bar_label",
                    "type": "text",
                    "label": "Panel and exact count",
                },
                "tooltip": [
                    {"field": "panel", "type": "nominal", "label": "Panel"},
                    {"field": "direction", "type": "ordinal", "label": "Direction"},
                    {
                        "field": "meeting_count",
                        "type": "quantitative",
                        "label": "Meetings",
                    },
                    {
                        "field": "panel_denominator",
                        "type": "quantitative",
                        "label": "Panel N",
                    },
                    {
                        "field": "within_panel_share_display",
                        "type": "nominal",
                        "label": "Within-panel support",
                    },
                ],
            },
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "Zero support",
                    "color": "neutral",
                    "lineStyle": "solid",
                }
            ],
            "palette": {"kind": "categorical"},
            "legend": {
                "position": "bottom",
                "sort": "spec",
                "title": "Evaluation panel",
            },
            "labels": {"values": "all"},
            "settings": {
                "groupMode": "grouped",
                "orientation": "vertical",
                "sort": "custom",
                "showValues": True,
                "categoryLabelPolicy": "wrap",
            },
            "valueFormat": "number",
            "unit": "meetings",
            "layout": "full",
            "maxRows": 6,
            "surface": {
                "viewMode": "both",
                "interactiveLegend": False,
                "showControls": False,
            },
        }
    ]
    tables = [
        {
            "id": "population_support_table",
            "title": "Separate panel populations and class support",
            "subtitle": "N=19 supports all directions; N=12 contains no hike.",
            "dataset": "population_support",
            "sourceId": "external_summary_source",
            "defaultSort": {"field": "meetings", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "panel", "label": "Panel", "type": "text"},
                {"field": "meetings", "label": "N", "format": "number"},
                {"field": "cut", "label": "Cut", "format": "number"},
                {"field": "hold", "label": "Hold", "format": "number"},
                {"field": "hike", "label": "Hike", "format": "number"},
                {"field": "balanced_accuracy_contract", "label": "BA contract", "type": "text"},
            ],
        },
        {
            "id": "historical_metrics_table",
            "title": "Historical N=19 model performance",
            "subtitle": "All three classes have support; BA averages cut, hold, and hike recall.",
            "dataset": "historical_metrics",
            "sourceId": "external_summary_source",
            "defaultSort": {"field": "model", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": historical_metric_columns,
        },
        {
            "id": "historical_mcnemar_table",
            "title": "Historical N=19 exact paired direction comparisons",
            "subtitle": (
                "Raw and Holm-adjusted exact two-sided McNemar p-values; "
                "one three-comparison family within N=19."
            ),
            "dataset": "historical_mcnemar",
            "sourceId": "external_summary_source",
            "defaultSort": {"field": "model_a_display", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": pair_columns,
        },
        {
            "id": "postcutoff_metrics_table",
            "title": "Post-cutoff N=12 model performance",
            "subtitle": "Supported-class BA averages cut and hold; three-class BA is not estimable.",
            "dataset": "postcutoff_metrics",
            "sourceId": "external_summary_source",
            "defaultSort": {"field": "model", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": postcutoff_metric_columns,
        },
        {
            "id": "postcutoff_mcnemar_table",
            "title": "Post-cutoff N=12 exact paired direction comparisons",
            "subtitle": (
                "Raw and Holm-adjusted exact two-sided McNemar p-values; "
                "one three-comparison family within N=12."
            ),
            "dataset": "postcutoff_mcnemar",
            "sourceId": "external_summary_source",
            "defaultSort": {"field": "model_a_display", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": pair_columns,
        },
    ]
    historical_summary = _panel_summary_sentence(historical, panel="historical_n19")
    postcutoff_summary = _panel_summary_sentence(postcutoff, panel="postcutoff_n12")
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary_results",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": (
                "## The panels provide separate descriptive diagnostics, not a pooled ranking\n\n"
                f"**Historical N=19.** {historical_summary}\n\n"
                f"**Post-cutoff N=12.** {postcutoff_summary}\n\n"
                "The N=12 panel has no hike, so fixed-three-class balanced accuracy "
                "is not estimable. No N=31 result is reported."
            ),
        },
        {
            "id": "technical_summary_design",
            "type": "markdown",
            "sourceId": "evaluation_manifest_source",
            "body": (
                "The workflow uses three frozen checkpoints, one greedy completion "
                "per meeting, and performs no retraining. The external inputs are "
                "deterministic Core8 source-analysis concatenations, not the "
                "teacher-compressed briefs used in the opened N=13 diagnostic."
            ),
        },
        {
            "id": "population_support_finding",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": (
                "## Class support determines which balanced accuracy is identifiable\n\n"
                "N=19 supports cut, hold, and hike recall. N=12 supports only cut "
                "and hold recall; the absent hike class is not silently assigned a "
                "report-facing zero."
            ),
        },
        {
            "id": "class_support_chart_block",
            "type": "chart",
            "chartId": "class_support_chart",
            "layout": "full",
        },
        {"id": "population_support_block", "type": "table", "tableId": "population_support_table", "layout": "full"},
        {
            "id": "historical_finding",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": "## Historical N=19 performance includes all three directions\n\n" + historical_summary,
        },
        {"id": "historical_metrics_block", "type": "table", "tableId": "historical_metrics_table", "layout": "full"},
        {
            "id": "historical_mcnemar_finding",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": "### N=19 paired correctness sensitivity\n\n" + _mcnemar_summary(historical_pairs),
        },
        {"id": "historical_mcnemar_block", "type": "table", "tableId": "historical_mcnemar_table", "layout": "full"},
        {
            "id": "postcutoff_finding",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": (
                "## Post-cutoff N=12 cannot estimate hike recall\n\n"
                + postcutoff_summary
                + " The reported balanced accuracy averages cut and hold recall only."
            ),
        },
        {"id": "postcutoff_metrics_block", "type": "table", "tableId": "postcutoff_metrics_table", "layout": "full"},
        {
            "id": "postcutoff_mcnemar_finding",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": "### N=12 paired correctness sensitivity\n\n" + _mcnemar_summary(postcutoff_pairs),
        },
        {"id": "postcutoff_mcnemar_block", "type": "table", "tableId": "postcutoff_mcnemar_table", "layout": "full"},
        {
            "id": "scope_and_metrics",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": (
                "## Metric definitions preserve denominators and absent classes\n\n"
                "Direction accuracy is correct meeting-level direction divided by "
                "all panel meetings; invalid outputs remain incorrect in the "
                "denominator. Class recall is correct predictions within a true "
                "class divided by that class's support. Historical fixed-three-class "
                "BA averages all three recalls. Post-cutoff supported-class BA "
                "averages cut and hold recall only."
            ),
        },
        {
            "id": "experimental_design",
            "type": "markdown",
            "sourceId": "evaluation_manifest_source",
            "body": (
                "## Frozen deterministic-Core8 evaluation design\n\n"
                "Each prompt concatenates eight D-1 source-analysis blocks in a "
                "fixed topic order. Meeting identity, gold action, and official "
                "policy text are excluded from prompts. Official documents serve "
                "only as label sources. The direct Core8 contract is not byte-"
                "comparable to the teacher-compressed N=13 input contract, so the "
                "panels are not pooled."
            ),
        },
        {
            "id": "mcnemar_method",
            "type": "markdown",
            "sourceId": "external_summary_source",
            "body": (
                "## Exact paired comparisons use meeting-level discordances\n\n"
                "For each pair, the exact two-sided McNemar test conditions on the "
                "meetings where exactly one model is direction-correct. All three "
                "model-pair comparisons form one Holm family within each panel. "
                "The N=19 and N=12 families remain separate, and paper-facing "
                "inference uses the adjusted p-values. Raw values are retained for "
                "auditability; non-significance does not establish equivalence."
            ),
        },
        {
            "id": "cp450_selection_status",
            "type": "markdown",
            "sourceId": "checkpoint_selection_source",
            "body": (
                "## GRPO cp450 remains null-selected\n\n"
                "Step 450 is the best-observed exploratory candidate used for "
                "evaluation. The formal record keeps `selected_checkpoint_step = "
                "null` and `provisional_no_candidate_passed_all_gates`; it is not "
                "a selected, promoted, or final model."
            ),
        },
        {
            "id": "cp450_quality_status",
            "type": "markdown",
            "sourceId": "cp450_final_status_source",
            "body": (
                "The already-opened sealed N=13 test records "
                "`quality_not_demonstrated`. The external panels do not revise "
                "that checkpoint-governance verdict."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "evaluation_manifest_source",
            "body": (
                "## Historical and retrospective scope limits generalization\n\n"
                "- N=19 has zero direct Decision-release overlap, but public "
                "historical events may have appeared in base-model pretraining.\n"
                "- N=12 is after the Decision training-data cutoff but predictions "
                "were not sealed before the public decisions; it is retrospective, "
                "not prospective.\n"
                "- N=12 has no hike; three-class performance is not estimable.\n"
                "- Small class counts make point estimates and exact paired tests "
                "unstable.\n"
                "- No N=31 or pooled-with-N=13 result is authorized."
            ),
        },
        {
            "id": "visual_omission",
            "type": "markdown",
            "body": (
                "## Model-performance evidence remains in exact tables\n\n"
                "Each panel has only three model rows. Exact numerators, denominators, "
                "and McNemar discordances are interpretation-critical, so no model-"
                "performance chart is used. The sole chart shows class support and "
                "explains why N=12 cannot identify three-class balanced accuracy."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Keep N=19, N=12, and the opened teacher-compressed N=13 "
                "diagnostic separate.\n"
                "2. Seal predictions before future decisions to create a genuinely "
                "prospective panel.\n"
                "3. Reassess checkpoint selection only with all three directions "
                "and a predeclared selection rule."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "Does cp38-versus-cp450 ordering persist in prospectively sealed "
                "meetings, when a new hike is observed, and under a predeclared "
                "stochastic replicate design?"
            ),
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": (
                "Technical report for separate N=19 historical and N=12 "
                "post-cutoff deterministic-Core8 Decision diagnostics."
            ),
            "generatedAt": generated_at,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "population_support": support,
                "class_support_chart": chart_support,
                "historical_metrics": historical,
                "historical_mcnemar": historical_pairs,
                "postcutoff_metrics": postcutoff,
                "postcutoff_mcnemar": postcutoff_pairs,
            },
        },
        "sources": copy.deepcopy(sources),
    }


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            if not value.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_text(
        path,
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True),
    )


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    return {
        **unsigned,
        "integrity": {
            "algorithm": INTEGRITY_ALGORITHM,
            "payload_sha256": _sha256_text(_canonical(unsigned)),
        },
    }


def render_report(
    summary_path: Path,
    output_dir: Path,
    *,
    selection_path: Path = DEFAULT_SELECTION,
    final_status_path: Path = DEFAULT_FINAL_STATUS,
) -> dict[str, Any]:
    """Atomically publish Markdown, artifact JSON, and a materialization seal."""

    output = output_dir.expanduser().resolve()
    _require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_report_model(
        summary_path,
        selection_path=selection_path,
        final_status_path=final_status_path,
    )
    markdown = build_markdown_report(model)
    artifact = build_canonical_artifact(model)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        markdown_path = staging / "technical_report.md"
        artifact_path = staging / "artifact.json"
        _write_new_text(markdown_path, markdown)
        _write_new_json(artifact_path, artifact)
        manifest = _sealed(
            {
                "schema_version": REPORT_MANIFEST_SCHEMA,
                "status": "complete_unpacked",
                "created_at_utc": _utc_now(),
                "input": model["source_bindings"],
                "renderer_provenance": {
                    "implementation": _binding(Path(__file__)),
                    "focused_test": _binding(
                        Path(__file__).resolve().parents[2]
                        / "tests/test_render_chk4_external_core8_technical_report.py"
                    ),
                },
                "outputs": {
                    "technical_report": _intended_binding(
                        markdown_path, output / "technical_report.md"
                    ),
                    "artifact": _intended_binding(
                        artifact_path, output / "artifact.json"
                    ),
                },
                "panels_reported_separately": True,
                "pooled_result": None,
                "training_performed": False,
                "html_packaged": False,
                "native_chart_count": 1,
                "performance_chart_omission_reason": (
                    "tiny three-row panels require exact denominators, absent-class "
                    "support, and McNemar discordance counts"
                ),
                "chart_map": [
                    {
                        "section": "class support and metric identification",
                        "analytical_question": (
                            "why the two panels require different balanced-accuracy "
                            "definitions"
                        ),
                        "family": "comparison",
                        "type": "grouped_bar",
                        "fields": {
                            "x": "direction",
                            "y": "meeting_count",
                            "color": "panel",
                            "label": "bar_label",
                        },
                        "supported_takeaway": (
                            "postcutoff_n12 has zero hikes while historical_n19 "
                            "supports cut, hold, and hike"
                        ),
                        "palette_policy": (
                            "hard two-root cap implemented as two categorical panel "
                            "series plus neutral zero reference"
                        ),
                        "non_color_plan": (
                            "direct panel-and-count labels, exact tooltip counts, "
                            "fixed direction order, and adjacent exact table"
                        ),
                        "delivery": "native chart in canonical artifact.json",
                    }
                ],
                "portable_render_qa": "deferred_until_real_summary_exists",
            }
        )
        _write_new_json(staging / "report_manifest.json", manifest)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "complete_unpacked",
        "output_dir": str(output),
        "technical_report": _binding(output / "technical_report.md"),
        "artifact": _binding(output / "artifact.json"),
        "report_manifest": _binding(output / "report_manifest.json"),
        "html_packaged": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--final-status", type=Path, default=DEFAULT_FINAL_STATUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = render_report(
        args.summary,
        args.output_dir,
        selection_path=args.selection,
        final_status_path=args.final_status,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
