#!/usr/bin/env python3
"""Run the authors' ordinalForest estimator on the reduced three-feature panel.

This is intentionally separate from ``evaluate_decision_comparison_baselines``.
The R estimator and its hyperparameters match Yoon and Fan (2024), while the
predictor set and matched-roster evaluation contract are project adaptations.
The authors' ``perffunction='equal'`` configuration yields point classes but no
class probabilities, so probability scores are explicitly unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).resolve()
R_RUNNER = ROOT / "jobs/eval/run_yoon_fan_ordinal_forest.R"
DEFAULT_CONFIG = ROOT / "configs/main/yoon_fan_ordinal_forest_reduced_v1.json"
SCHEMA = "yoon-fan-ordinal-forest-reduced-result-v1"
CLASS_ORDER = ("cut", "hold", "hike")
FEATURE_NAMES = (
    "tbill6m_minus_effr_pp",
    "unemployment_rate_pct",
    "real_gdp_growth_annualized_pct",
)
EXPECTED_TRAIN_COUNTS = {"cut": 23, "hold": 156, "hike": 32}
EXPECTED_H19_COUNTS = {"cut": 4, "hold": 10, "hike": 5}
PROBABILITY_UNAVAILABLE = "unavailable_equal_performance_author_configuration"


class YoonFanReplicationError(RuntimeError):
    """The frozen algorithm, input, or output contract failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash file names and contents in a package tree deterministically."""
    if path.is_symlink() or not path.is_dir():
        raise YoonFanReplicationError(f"package tree must be a directory: {path}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise YoonFanReplicationError(f"package tree is empty: {path}")
    for candidate in files:
        if candidate.is_symlink():
            raise YoonFanReplicationError(
                f"package tree contains a symbolic link: {candidate}"
            )
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise YoonFanReplicationError(f"JSON input must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise YoonFanReplicationError(f"expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise YoonFanReplicationError(f"JSONL input must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise YoonFanReplicationError(
                    f"blank JSONL row at {path}:{line_number}"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise YoonFanReplicationError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "yoon-fan-ordinal-forest-reduced-v1":
        raise YoonFanReplicationError("configuration schema drift")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise YoonFanReplicationError("model configuration missing")
    expected_parameters = {
        "nsets": 1000,
        "ntreeperdiv": 100,
        "ntreefinal": 5000,
        "importance": "rps",
        "perffunction": "equal",
        "nbest": 10,
        "naive": False,
        "num_threads": 1,
        "npermtrial": 500,
        "permperdefault": False,
        "mtry": 1,
        "min_node_size": 5,
        "replace": True,
        "sample_fraction": 1.0,
        "keep_inbag": False,
    }
    if (
        model.get("r_package") != "ordinalForest"
        or model.get("r_package_version") != "2.4-3"
        or model.get("r_package_archive_sha256")
        != "fa60dcb890818b650a7cf35cda60cc85e29cd53864c811b336bb09622baf7419"
        or model.get("class_order") != list(CLASS_ORDER)
        or model.get("parameters") != expected_parameters
    ):
        raise YoonFanReplicationError("frozen ordinalForest model contract drift")
    data = config.get("data_contract")
    if not isinstance(data, Mapping):
        raise YoonFanReplicationError("data contract missing")
    if (
        data.get("training_contract") != "matched_unique_train_n211"
        or data.get("evaluation_panel") != "historical_n19"
        or data.get("feature_names") != list(FEATURE_NAMES)
        or data.get("expected_training_rows") != 211
        or data.get("expected_training_classes") != EXPECTED_TRAIN_COUNTS
        or data.get("expected_evaluation_rows") != 19
        or data.get("expected_evaluation_classes") != EXPECTED_H19_COUNTS
        or data.get("expected_training_rows_after_h19_end") != 106
    ):
        raise YoonFanReplicationError("frozen reduced data contract drift")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get(
        "probability_metrics", "missing"
    ) is not None:
        raise YoonFanReplicationError("probability metrics must remain null")


def verify_source_hashes(source_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    status = load_json(source_root / "status.json")
    if status.get("state") != "complete" or status.get("phase") != "score":
        raise YoonFanReplicationError("source comparison artifact is not complete")
    expected_hashes = config["data_contract"]["source_hashes"]
    observed: dict[str, Any] = {}
    for relative, expected_hash in expected_hashes.items():
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise YoonFanReplicationError(f"source input missing: {path}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise YoonFanReplicationError(
                f"source hash drift for {relative}: {digest} != {expected_hash}"
            )
        observed[relative] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return observed


def _finite_feature_vector(row: Mapping[str, Any], meeting_id: str) -> dict[str, float]:
    values = row.get("features")
    if not isinstance(values, Mapping) or set(values) != set(FEATURE_NAMES):
        raise YoonFanReplicationError(
            f"feature allowlist mismatch for {meeting_id}"
        )
    result: dict[str, float] = {}
    for name in FEATURE_NAMES:
        value = values.get(name)
        if isinstance(value, bool):
            raise YoonFanReplicationError(f"boolean feature {name} for {meeting_id}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise YoonFanReplicationError(
                f"non-numeric feature {name} for {meeting_id}"
            ) from exc
        if not math.isfinite(numeric):
            raise YoonFanReplicationError(
                f"non-finite feature {name} for {meeting_id}"
            )
        result[name] = numeric
    return result


def build_reduced_inputs(
    source_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_source_hashes(source_root, config)
    roster = load_jsonl(source_root / "action_roster.jsonl")
    features = load_jsonl(source_root / "features.jsonl")
    roster_by_id = {str(row.get("meeting_id")): row for row in roster}
    feature_by_id = {str(row.get("meeting_id")): row for row in features}
    if (
        len(roster) != 268
        or len(features) != 268
        or len(roster_by_id) != 268
        or len(feature_by_id) != 268
        or set(roster_by_id) != set(feature_by_id)
    ):
        raise YoonFanReplicationError("source roster/feature closure failed")

    train_roster = [row for row in roster if row.get("role") == "train"]
    h19_roster = [row for row in roster if row.get("panel") == "historical_n19"]
    train_roster.sort(key=lambda row: (row["meeting_start_date"], row["meeting_id"]))
    h19_roster.sort(key=lambda row: (row["meeting_start_date"], row["meeting_id"]))
    if len(train_roster) != 211 or len(h19_roster) != 19:
        raise YoonFanReplicationError("N211/H19 row-count closure failed")
    if Counter(str(row.get("direction")) for row in train_roster) != Counter(
        EXPECTED_TRAIN_COUNTS
    ):
        raise YoonFanReplicationError("N211 class distribution drift")
    if Counter(str(row.get("direction")) for row in h19_roster) != Counter(
        EXPECTED_H19_COUNTS
    ):
        raise YoonFanReplicationError("H19 class distribution drift")

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for split, selected, destination in (
        ("train", train_roster, train_rows),
        ("historical_n19", h19_roster, test_rows),
    ):
        for meeting in selected:
            meeting_id = str(meeting["meeting_id"])
            feature = feature_by_id[meeting_id]
            if (
                feature.get("role") != meeting.get("role")
                or feature.get("panel") != meeting.get("panel")
                or feature.get("target_direction") != meeting.get("direction")
                or feature.get("meeting_start_date") != meeting.get("meeting_start_date")
                or feature.get("evidence_cutoff") != meeting.get("evidence_cutoff")
            ):
                raise YoonFanReplicationError(
                    f"roster/feature metadata mismatch for {meeting_id}"
                )
            row: dict[str, Any] = {"row_id": meeting_id}
            if split == "train":
                row["direction"] = str(meeting["direction"])
            row.update(_finite_feature_vector(feature, meeting_id))
            destination.append(row)
            metadata[meeting_id] = {
                "meeting_id": meeting_id,
                "meeting_start_date": str(meeting["meeting_start_date"]),
                "meeting_end_date": str(meeting["meeting_end_date"]),
                "evidence_cutoff": str(meeting["evidence_cutoff"]),
                "true_direction": str(meeting["direction"]),
                "feature_sha256": str(feature["feature_sha256"]),
                "original_paper_sample_eligible": str(meeting["meeting_start_date"])
                >= "1994-02-01",
            }
    if set(row["row_id"] for row in train_rows) & set(
        row["row_id"] for row in test_rows
    ):
        raise YoonFanReplicationError("training/H19 meeting overlap")
    return train_rows, test_rows, metadata


def write_r_inputs(
    directory: Path,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=False)
    train_path = directory / "train_n211.csv"
    test_path = directory / "historical_n19.csv"
    with train_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "direction", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(train_rows)
    with test_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(test_rows)
    return train_path, test_path


def preflight_r(r_library: Path) -> dict[str, str]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise YoonFanReplicationError("Rscript is not available")
    if not r_library.is_dir():
        raise YoonFanReplicationError(
            f"frozen ordinalForest R library is missing: {r_library}"
        )
    environment = os.environ.copy()
    environment.update(
        {
            "R_LIBS_USER": str(r_library.resolve()),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        rscript,
        "-e",
        (
            "suppressPackageStartupMessages(library(ordinalForest));"
            "cat(as.character(getRversion()),'\\n',sep='');"
            "cat(as.character(packageVersion('ordinalForest')),'\\n',sep='');"
            "p<-c('Rcpp','combinat','nnet','verification');"
            "cat(paste(vapply(p,function(x)paste0(x,'=',as.character(packageVersion(x))),character(1)),collapse=';'),'\\n',sep='')"
        ),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if completed.returncode != 0:
        raise YoonFanReplicationError(
            "ordinalForest R preflight failed: " + completed.stderr.strip()
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 3 or lines[1] != "2.4.3":
        raise YoonFanReplicationError(
            f"unexpected R/ordinalForest preflight output: {lines}"
        )
    return {
        "rscript": str(Path(rscript).resolve()),
        "r": lines[0],
        "ordinalForest": lines[1],
        "dependencies": lines[2],
    }


def run_r_model(
    train_path: Path,
    test_path: Path,
    work_root: Path,
    r_library: Path,
    *,
    seed: int,
) -> tuple[Path, Path, str]:
    runtime = preflight_r(r_library)
    predictions_path = work_root / "r_predictions.csv"
    diagnostics_path = work_root / "r_diagnostics.csv"
    environment = os.environ.copy()
    environment.update(
        {
            "R_LIBS_USER": str(r_library.resolve()),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = [
        runtime["rscript"],
        str(R_RUNNER.resolve()),
        str(train_path.resolve()),
        str(test_path.resolve()),
        str(predictions_path.resolve()),
        str(diagnostics_path.resolve()),
        str(seed),
        "1",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=4 * 60 * 60,
    )
    log = (
        f"command={json.dumps(command)}\n"
        f"returncode={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
    )
    if completed.returncode != 0:
        raise YoonFanReplicationError(
            "ordinalForest R run failed; captured log:\n" + log[-8000:]
        )
    if not predictions_path.is_file() or not diagnostics_path.is_file():
        raise YoonFanReplicationError("R run did not create both output files")
    return predictions_path, diagnostics_path, log


def load_r_predictions(
    path: Path, expected_row_ids: Sequence[str]
) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_columns = {"row_id", "predicted_direction", "probability_status"}
    if any(set(row) != expected_columns for row in rows):
        raise YoonFanReplicationError("R prediction schema drift")
    if [row["row_id"] for row in rows] != list(expected_row_ids):
        raise YoonFanReplicationError("R prediction row order/coverage drift")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise YoonFanReplicationError("duplicate R prediction row_id")
    result: dict[str, str] = {}
    for row in rows:
        prediction = row["predicted_direction"]
        if prediction not in CLASS_ORDER:
            raise YoonFanReplicationError(f"invalid R prediction class: {prediction}")
        if row["probability_status"] != PROBABILITY_UNAVAILABLE:
            raise YoonFanReplicationError("R probability availability contract drift")
        result[row["row_id"]] = prediction
    return result


def score_point_predictions(
    truth_by_id: Mapping[str, str], prediction_by_id: Mapping[str, str]
) -> dict[str, Any]:
    if set(truth_by_id) != set(prediction_by_id) or not truth_by_id:
        raise YoonFanReplicationError("truth/prediction meeting closure failed")
    confusion = {
        actual: {predicted: 0 for predicted in CLASS_ORDER}
        for actual in CLASS_ORDER
    }
    per_class: dict[str, Any] = {}
    correct = 0
    f1_values: list[float] = []
    for meeting_id, actual in truth_by_id.items():
        predicted = prediction_by_id[meeting_id]
        confusion[actual][predicted] += 1
        correct += int(actual == predicted)
    for label in CLASS_ORDER:
        support = sum(confusion[label].values())
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[actual][label] for actual in CLASS_ORDER if actual != label
        )
        recall = true_positive / support
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "correct": true_positive,
            "support": support,
            "accuracy": recall,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return {
        "n_meetings": len(truth_by_id),
        "correct": correct,
        "overall_accuracy": correct / len(truth_by_id),
        "balanced_accuracy": sum(
            per_class[label]["recall"] for label in CLASS_ORDER
        )
        / len(CLASS_ORDER),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "probability_metrics": None,
        "probability_unavailable_reason": (
            "ordinalForest 2.4-3 returns classprobs=NA under "
            "perffunction='equal', the configuration selected by Yoon and Fan (2024)"
        ),
    }


def read_diagnostics(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(set(row) != {"name", "value"} for row in rows):
        raise YoonFanReplicationError("R diagnostics schema drift")
    result = {row["name"]: row["value"] for row in rows}
    if len(result) != len(rows) or result.get("ordinalForest_version") != "2.4.3":
        raise YoonFanReplicationError("R diagnostics version/key closure failed")
    if result.get("perffunction") != "equal" or result.get("probability_output") != "NA_under_equal":
        raise YoonFanReplicationError("R diagnostics model-output contract drift")
    return result


def execute(
    *,
    config_path: Path,
    source_root: Path,
    output_root: Path,
    r_library: Path,
    seed: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise YoonFanReplicationError(
            f"create-only output root already exists: {output_root}"
        )
    if seed != 20260827:
        raise YoonFanReplicationError("frozen seed must be 20260827")
    config = load_json(config_path)
    validate_config(config)
    source_inputs = verify_source_hashes(source_root, config)
    train_rows, test_rows, metadata = build_reduced_inputs(source_root, config)
    runtime = preflight_r(r_library)
    training_dates = [metadata[str(row["row_id"])]["meeting_start_date"] for row in train_rows]
    evaluation_dates = [metadata[str(row["row_id"])]["meeting_start_date"] for row in test_rows]
    h19_end = max(evaluation_dates)
    training_rows_after_h19_end = sum(date > h19_end for date in training_dates)
    if training_rows_after_h19_end != 106:
        raise YoonFanReplicationError(
            "retrospective training-date contract drift: "
            f"{training_rows_after_h19_end} != 106"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        input_root = temporary / "inputs"
        train_path, test_path = write_r_inputs(input_root, train_rows, test_rows)
        r_predictions, r_diagnostics, r_log = run_r_model(
            train_path,
            test_path,
            temporary,
            r_library,
            seed=seed,
        )
        predicted = load_r_predictions(
            r_predictions, [str(row["row_id"]) for row in test_rows]
        )
        diagnostics = read_diagnostics(r_diagnostics)
        truth = {
            meeting_id: row["true_direction"]
            for meeting_id, row in metadata.items()
            if meeting_id in predicted
        }
        metrics = score_point_predictions(truth, predicted)
        prediction_rows: list[dict[str, Any]] = []
        for test_row in test_rows:
            meeting_id = str(test_row["row_id"])
            meta = metadata[meeting_id]
            prediction_rows.append(
                {
                    **meta,
                    "model": config["model"]["id"],
                    "model_display_name": config["model"]["display_name"],
                    "algorithm": "ordinalForest::ordfor",
                    "r_package_version": "2.4-3",
                    "fit_contract": "matched_unique_train_n211",
                    "predicted_direction": predicted[meeting_id],
                    "direction_correct": predicted[meeting_id]
                    == meta["true_direction"],
                    "probabilities": None,
                    "probability_status": PROBABILITY_UNAVAILABLE,
                    "seed": seed,
                }
            )
        write_jsonl(temporary / "predictions.jsonl", prediction_rows)
        (temporary / "r_run.log").write_text(r_log, encoding="utf-8")

        summary = {
            "schema_version": SCHEMA,
            "model": config["model"],
            "replication_status": config["replication_status"],
            "data_contract": {
                "training_contract": "matched_unique_train_n211",
                "training_rows": 211,
                "training_class_counts": EXPECTED_TRAIN_COUNTS,
                "evaluation_panel": "historical_n19",
                "evaluation_rows": 19,
                "evaluation_class_counts": EXPECTED_H19_COUNTS,
                "feature_names": list(FEATURE_NAMES),
                "training_meeting_start_range": [min(training_dates), max(training_dates)],
                "evaluation_meeting_start_range": [
                    min(evaluation_dates),
                    max(evaluation_dates),
                ],
                "training_rows_after_evaluation_panel_end": (
                    training_rows_after_h19_end
                ),
                "original_paper_protocol_evaluated": False,
                "matched_roster_adaptation_h19_coverage": 19,
            },
            "metrics": metrics,
            "r_diagnostics": diagnostics,
            "interpretation": (
                "Point-classification results from the authors' ordinalForest estimator "
                "using a reduced three-predictor, matched-roster project contract. This "
                "is not a replication of the paper's 45-predictor rolling-origin result."
            ),
            "pooled_result": None,
        }
        write_json(temporary / "summary.json", summary)
        manifest = {
            "schema_version": SCHEMA,
            "created_at_utc": utc_now(),
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
            "source_inputs": source_inputs,
            "implementation": {
                "python": {
                    "path": str(IMPLEMENTATION.resolve()),
                    "sha256": sha256_file(IMPLEMENTATION),
                },
                "r": {
                    "path": str(R_RUNNER.resolve()),
                    "sha256": sha256_file(R_RUNNER),
                },
            },
            "environment": {
                "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "python": platform.python_version(),
                "r": runtime["r"],
                "ordinalForest": runtime["ordinalForest"],
                "r_dependency_versions": runtime["dependencies"],
                "r_library": str(r_library.resolve()),
                "ordinalForest_installed_tree_sha256": sha256_tree(
                    r_library / "ordinalForest"
                ),
            },
            "algorithm_identity": {
                "same_estimator_as_yoon_fan_2024": True,
                "r_package": "ordinalForest",
                "version": "2.4-3",
                "archive_sha256": config["model"]["r_package_archive_sha256"],
                "explicit_parameters": config["model"]["parameters"],
            },
            "input_artifacts": {
                "inputs/train_n211.csv": {
                    "bytes": train_path.stat().st_size,
                    "sha256": sha256_file(train_path),
                    "contains_target": True,
                },
                "inputs/historical_n19.csv": {
                    "bytes": test_path.stat().st_size,
                    "sha256": sha256_file(test_path),
                    "contains_target": False,
                },
            },
            "output_artifacts": {},
        }
        for relative in (
            "predictions.jsonl",
            "summary.json",
            "r_predictions.csv",
            "r_diagnostics.csv",
            "r_run.log",
        ):
            artifact = temporary / relative
            manifest["output_artifacts"][relative] = {
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        write_json(temporary / "manifest.json", manifest)
        status = {
            "schema_version": "yoon-fan-ordinal-forest-reduced-status-v1",
            "state": "complete",
            "phase": "score",
            "updated_at_utc": utc_now(),
            "output_root": str(output_root.resolve()),
            "summary_sha256": sha256_file(temporary / "summary.json"),
        }
        write_json(temporary / "status.json", status)
        os.chmod(temporary, 0o700)
        temporary.rename(output_root)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def status_payload(
    config_path: Path, source_root: Path, r_library: Path
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_config(config)
    payload: dict[str, Any] = {
        "config": "valid",
        "source": "unchecked",
        "r_runtime": "unchecked",
    }
    try:
        verify_source_hashes(source_root, config)
        build_reduced_inputs(source_root, config)
        payload["source"] = "ready"
    except Exception as exc:  # status reports rather than raising
        payload["source"] = f"unavailable:{type(exc).__name__}:{exc}"
    try:
        payload["r_runtime"] = preflight_r(r_library)
    except Exception as exc:  # status reports rather than raising
        payload["r_runtime"] = f"unavailable:{type(exc).__name__}:{exc}"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Yoon--Fan's ordinalForest estimator with the frozen reduced "
            "three-predictor matched-roster contract"
        )
    )
    parser.add_argument("--phase", choices=("status", "run"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--r-library", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    source_root = args.source_root or ROOT / config["data_contract"]["source_root"]
    output_root = args.output_root or ROOT / config["default_output_root"]
    r_library = args.r_library or ROOT / config["default_r_library"]
    seed = int(config["seed"] if args.seed is None else args.seed)
    if args.phase == "status":
        print(
            json.dumps(
                status_payload(args.config, source_root, r_library),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = execute(
        config_path=args.config,
        source_root=source_root,
        output_root=output_root,
        r_library=r_library,
        seed=seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
