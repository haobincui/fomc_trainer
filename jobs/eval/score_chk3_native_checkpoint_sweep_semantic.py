"""Score a frozen chk3 native-checkpoint sweep with pinned semantic encoders.

This module is deliberately generation-free.  It accepts the sealed native
``analysis -> Minutes`` N12 run manifests produced by
``eval_chk3_native_three_model`` and deep-validates every persisted generation
before semantic inference.  The formal output is one immutable, sealed JSON
containing row-level scores, cohort summaries, and (when supplied) paired
contrasts against the chk1 baseline.

The v1 sweep inventory is fixed before looking at results: checkpoint 170,
200, 230, and 318.  BERTScore and MPNet are loaded only from the paths and
hashes in the pinned semantic-model manifest.  Invalid delivery or an
independently recomputed core-fidelity failure is never hidden by a plausible
similarity score: such rows are not sent to the encoders and receive a fixed
zero in the all-N12 and non-identity-N11 views.  The generation runner's
persisted ``quality_valid`` remains visible for discrepancy auditing, but is
not authoritative for semantic-score eligibility.  Exact-merged finalists use
the preregistered delivery/structure/numeric/date/degeneration gate; the
post-hoc signed-surface occurrence check is reported separately as an extended
diagnostic and cannot by itself suppress a merged-finalist semantic score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.eval import eval_checkpoint_generation as semantic_eval
from jobs.eval import eval_chk3_native_three_model as native_eval
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-native-checkpoint-sweep-semantic-v1"
ROW_SCHEMA_VERSION = "chk3-native-checkpoint-sweep-semantic-row-v1"
EVALUATION_ID = "chk3-native-analysis-to-minutes-cp170-cp200-cp230-cp318-n12-v1"
MERGED_FINALIST_EVALUATION_ID = (
    "chk3-native-analysis-to-minutes-merged-cp200-cp318-n12-v1"
)
SCORING_POLICY_VERSION = "recomputed-core-hard-gate-zero-penalty-v2"
PREREGISTERED_SCORING_POLICY_VERSION = "preregistered-core-zero-penalty-v1"
SIGNED_NUMBER_SURFACE_POLICY = "terminal-punctuation-aware-surface-v1"
REQUIRED_CHECKPOINT_STEPS = (170, 200, 230, 318)
MERGED_FINALIST_CHECKPOINT_STEPS = (200, 318)
EXPECTED_SAMPLE_COUNT = 12
EXPECTED_NON_IDENTITY_COUNT = 11
BASELINE_RUN_ID = "chk1_baseline"
METRICS = (
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "mpnet_cosine",
)
COHORTS = ("all_n12", "non_identity_n11", "valid_only")
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_MERGED_LABEL_CHECKPOINT_RE = re.compile(r"(?:^|[-_])cp(\d+)(?=$|[-_])")
_SIGNED_NUMBER_SURFACE_RE = re.compile(
    r"(?<![\w.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\w|\.\d)"
)


class NativeSweepSemanticError(RuntimeError):
    """A sealed input, scoring result, or output invariant failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    return native_eval.native_probe.common_probe.sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise NativeSweepSemanticError(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeSweepSemanticError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeSweepSemanticError(f"JSON root must be an object: {resolved}")
    return value


def _file_binding(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise NativeSweepSemanticError(f"artifact is not a regular file: {resolved}")
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
            raise NativeSweepSemanticError(
                f"sealed artifact integrity failed: {resolved}: {exc}"
            ) from exc
    return result


def _assert_file_binding_unchanged(
    binding: Mapping[str, Any], *, label: str, sealed: bool = False
) -> None:
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise NativeSweepSemanticError(f"{label} has no bound path")
    observed = _file_binding(Path(path_value), sealed=sealed)
    keys = ("sha256", "bytes", *(("payload_sha256",) if sealed else ()))
    for key in keys:
        if observed.get(key) != binding.get(key):
            raise NativeSweepSemanticError(f"{label} changed during semantic scoring")


def _write_new_sealed_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without replacing any existing path."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".staging", dir=str(resolved.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as exc:
            raise NativeSweepSemanticError(
                f"refusing to overwrite semantic artifact: {resolved}"
            ) from exc
        directory_fd = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_step(run: Mapping[str, Any]) -> int:
    manifest = run.get("manifest")
    if not isinstance(manifest, Mapping):
        raise NativeSweepSemanticError("validated candidate has no sealed manifest")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, Mapping):
        raise NativeSweepSemanticError(
            "formal v1 candidate must be an adapter-bound native run"
        )
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise NativeSweepSemanticError("candidate has no bound chk1 base model")
    model_path = model.get("path")
    declared_base = adapter.get("declared_base_model_path")
    if (
        not isinstance(model_path, str)
        or not isinstance(declared_base, str)
        or Path(model_path).expanduser().resolve()
        != Path(declared_base).expanduser().resolve()
    ):
        raise NativeSweepSemanticError(
            "candidate adapter/base-model lineage binding drift"
        )
    raw_path = adapter.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise NativeSweepSemanticError("candidate adapter path is missing")
    match = _CHECKPOINT_RE.fullmatch(Path(raw_path).name)
    if match is None:
        raise NativeSweepSemanticError(
            f"candidate adapter is not a checkpoint directory: {raw_path}"
        )
    return int(match.group(1))


def _merged_checkpoint_step(run: Mapping[str, Any]) -> int:
    """Bind an exact-merged finalist to the checkpoint named in its label."""

    manifest = run.get("manifest")
    if not isinstance(manifest, Mapping):
        raise NativeSweepSemanticError("validated merged finalist has no manifest")
    if manifest.get("adapter") is not None:
        raise NativeSweepSemanticError(
            "exact-merged finalist must not contain an adapter overlay"
        )
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise NativeSweepSemanticError("exact-merged finalist has no model binding")
    label = run.get("model_label")
    if not isinstance(label, str) or not label:
        raise NativeSweepSemanticError("exact-merged finalist has no model label")
    matches = {int(value) for value in _MERGED_LABEL_CHECKPOINT_RE.findall(label)}
    if len(matches) != 1:
        raise NativeSweepSemanticError(
            f"merged finalist label must name exactly one checkpoint: {label!r}"
        )
    step = next(iter(matches))
    if step not in MERGED_FINALIST_CHECKPOINT_STEPS:
        raise NativeSweepSemanticError(
            f"merged finalist label names unsupported checkpoint: {label!r}"
        )
    return step


def _run_row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample_id"),
        row.get("length_bucket"),
        row.get("source_prompt_sha256"),
        row.get("source_analysis_sha256"),
        row.get("reference_minutes_sha256"),
        bool(row.get("normalized_identity")),
    )


def _validate_aligned_runs(
    runs: Mapping[str, Mapping[str, Any]], *, require_shared_model: bool
) -> None:
    """Fail closed unless every deep-validated run is the same frozen N12 task."""

    if not runs:
        raise NativeSweepSemanticError("semantic sweep has no validated runs")
    reference_run = next(iter(runs.values()))
    reference_contract = reference_run.get("generation_contract")
    reference_rows = reference_run.get("results")
    if not isinstance(reference_contract, Mapping) or not isinstance(
        reference_rows, list
    ):
        raise NativeSweepSemanticError("validated run is missing its contract or rows")
    if len(reference_rows) != EXPECTED_SAMPLE_COUNT:
        raise NativeSweepSemanticError("formal semantic sweep requires N12")
    reference_identity = [_run_row_identity(row) for row in reference_rows]
    reference_model = reference_run.get("manifest", {}).get("model")
    if not isinstance(reference_model, Mapping):
        raise NativeSweepSemanticError("reference run has no model fingerprint")
    if sum(bool(row.get("normalized_identity")) for row in reference_rows) != 1:
        raise NativeSweepSemanticError("formal N12 must contain exactly one identity row")

    model_labels: set[str] = set()
    for run_id, run in runs.items():
        rows = run.get("results")
        label = run.get("model_label")
        model = run.get("manifest", {}).get("model")
        if not isinstance(rows, list) or len(rows) != EXPECTED_SAMPLE_COUNT:
            raise NativeSweepSemanticError(f"{run_id} is not a complete N12 run")
        if run.get("generation_contract") != reference_contract:
            raise NativeSweepSemanticError(
                f"generation contract drift between runs: {run_id}"
            )
        if [_run_row_identity(row) for row in rows] != reference_identity:
            raise NativeSweepSemanticError(f"sample/reference drift in {run_id}")
        if not isinstance(model, Mapping):
            raise NativeSweepSemanticError(f"{run_id} has no model fingerprint")
        if require_shared_model and model != reference_model:
            raise NativeSweepSemanticError(f"chk1 base-model drift in {run_id}")
        if not isinstance(label, str) or not label or label in model_labels:
            raise NativeSweepSemanticError(f"duplicate or invalid model label: {label!r}")
        model_labels.add(label)


def _load_baseline_run(
    *,
    baseline_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
) -> dict[str, Any]:
    baseline_path = baseline_manifest.expanduser().resolve()
    try:
        baseline = native_eval.load_and_validate_run(
            baseline_path,
            expected_stage_id="chk1",
            sample_manifest_path=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        )
    except Exception as exc:
        raise NativeSweepSemanticError(
            f"chk1 baseline deep validation failed: {baseline_path}: {exc}"
        ) from exc
    baseline = dict(baseline)
    if baseline.get("manifest", {}).get("adapter") is not None:
        raise NativeSweepSemanticError("chk1 baseline must not load an adapter")
    baseline["run_id"] = BASELINE_RUN_ID
    baseline["role"] = "baseline"
    baseline["checkpoint_step"] = None
    return baseline


def load_sweep_runs(
    *,
    candidate_manifests: Sequence[Path],
    baseline_manifest: Path | None,
    sample_manifest: Path,
    sample_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Deep-validate the fixed candidate inventory and optional chk1 baseline."""

    paths = [path.expanduser().resolve() for path in candidate_manifests]
    if len(paths) != len(REQUIRED_CHECKPOINT_STEPS) or len(set(paths)) != len(paths):
        raise NativeSweepSemanticError(
            "formal v1 sweep requires four distinct candidate manifests"
        )

    candidates: dict[int, dict[str, Any]] = {}
    for path in paths:
        try:
            run = native_eval.load_and_validate_run(
                path,
                expected_stage_id="chk3",
                sample_manifest_path=sample_manifest,
                sample_manifest_sha256=sample_manifest_sha256,
            )
        except Exception as exc:
            raise NativeSweepSemanticError(
                f"candidate run deep validation failed: {path}: {exc}"
            ) from exc
        step = _checkpoint_step(run)
        if step in candidates:
            raise NativeSweepSemanticError(f"duplicate checkpoint in sweep: {step}")
        run = dict(run)
        run["run_id"] = f"cp{step}"
        run["role"] = "candidate"
        run["checkpoint_step"] = step
        candidates[step] = run
    if set(candidates) != set(REQUIRED_CHECKPOINT_STEPS):
        raise NativeSweepSemanticError(
            "candidate inventory drift: expected "
            f"{list(REQUIRED_CHECKPOINT_STEPS)}, observed={sorted(candidates)}"
        )

    runs: dict[str, dict[str, Any]] = {}
    if baseline_manifest is not None:
        runs[BASELINE_RUN_ID] = _load_baseline_run(
            baseline_manifest=baseline_manifest,
            sample_manifest=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        )
    for step in REQUIRED_CHECKPOINT_STEPS:
        runs[f"cp{step}"] = candidates[step]
    _validate_aligned_runs(runs, require_shared_model=True)
    return runs


def load_merged_finalist_runs(
    *,
    finalist_manifests: Sequence[Path],
    baseline_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Deep-validate exactly the cp200/cp318 exact-merged finalist inventory."""

    paths = [path.expanduser().resolve() for path in finalist_manifests]
    if len(paths) != len(MERGED_FINALIST_CHECKPOINT_STEPS) or len(set(paths)) != len(
        paths
    ):
        raise NativeSweepSemanticError(
            "merged-finalist mode requires two distinct manifests"
        )
    candidates: dict[int, dict[str, Any]] = {}
    for path in paths:
        try:
            run = native_eval.load_and_validate_run(
                path,
                expected_stage_id="chk3",
                sample_manifest_path=sample_manifest,
                sample_manifest_sha256=sample_manifest_sha256,
            )
        except Exception as exc:
            raise NativeSweepSemanticError(
                f"merged finalist deep validation failed: {path}: {exc}"
            ) from exc
        step = _merged_checkpoint_step(run)
        if step in candidates:
            raise NativeSweepSemanticError(
                f"duplicate checkpoint in merged finalists: {step}"
            )
        run = dict(run)
        run["run_id"] = f"cp{step}"
        run["role"] = "candidate"
        run["checkpoint_step"] = step
        candidates[step] = run
    if set(candidates) != set(MERGED_FINALIST_CHECKPOINT_STEPS):
        raise NativeSweepSemanticError(
            "merged-finalist inventory drift: expected "
            f"{list(MERGED_FINALIST_CHECKPOINT_STEPS)}, "
            f"observed={sorted(candidates)}"
        )

    runs = {
        BASELINE_RUN_ID: _load_baseline_run(
            baseline_manifest=baseline_manifest,
            sample_manifest=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        ),
        **{
            f"cp{step}": candidates[step]
            for step in MERGED_FINALIST_CHECKPOINT_STEPS
        },
    }
    _validate_aligned_runs(runs, require_shared_model=False)

    content_fingerprints: set[str] = set()
    for run_id, run in runs.items():
        files = run.get("manifest", {}).get("model", {}).get("files")
        if not isinstance(files, Mapping) or not files:
            raise NativeSweepSemanticError(
                f"{run_id} exact-merged model has no file fingerprint"
            )
        fingerprint = json.dumps(files, sort_keys=True, separators=(",", ":"))
        if fingerprint in content_fingerprints:
            raise NativeSweepSemanticError(
                f"duplicate exact-merged model contents in {run_id}"
            )
        content_fingerprints.add(fingerprint)
    return runs


def _assert_run_sources_unchanged(runs: Mapping[str, Mapping[str, Any]]) -> None:
    for run_id, run in runs.items():
        manifest_binding = run.get("manifest_binding")
        manifest = run.get("manifest")
        if not isinstance(manifest_binding, Mapping) or not isinstance(
            manifest, Mapping
        ):
            raise NativeSweepSemanticError(f"{run_id} lost its source bindings")
        _assert_file_binding_unchanged(
            manifest_binding, label=f"{run_id} run manifest", sealed=True
        )
        artifacts = manifest.get("artifacts")
        generations = (
            artifacts.get("generations") if isinstance(artifacts, Mapping) else None
        )
        if not isinstance(generations, Mapping):
            raise NativeSweepSemanticError(f"{run_id} lost its generations binding")
        _assert_file_binding_unchanged(
            generations, label=f"{run_id} generations", sealed=False
        )


def load_formal_semantic_backends(
    *, semantic_manifest_path: Path, batch_size: int, device: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Instantiate and hash-validate the two formal local semantic backends."""

    if batch_size < 1:
        raise NativeSweepSemanticError("semantic batch size must be positive")
    manifest_path = semantic_manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    try:
        payload_sha = validate_manifest_integrity(manifest)
    except Exception as exc:
        raise NativeSweepSemanticError(
            f"semantic manifest integrity failed: {exc}"
        ) from exc
    if manifest.get("schema_version") != semantic_eval.SEMANTIC_MANIFEST_SCHEMA_VERSION:
        raise NativeSweepSemanticError("unsupported semantic manifest schema")
    if manifest.get("network_at_scoring_time") is not False:
        raise NativeSweepSemanticError("semantic manifest must require offline scoring")
    source_evaluation_id = manifest.get("created_for_evaluation_id")
    if not isinstance(source_evaluation_id, str) or not source_evaluation_id.strip():
        raise NativeSweepSemanticError(
            "semantic manifest has no source evaluation identity"
        )
    models = manifest.get("models")
    if not isinstance(models, Mapping) or set(models) != {
        "bertscore",
        "embedding_cosine",
    }:
        raise NativeSweepSemanticError("semantic manifest model inventory drift")
    bert_record = models["bertscore"]
    mpnet_record = models["embedding_cosine"]
    if not isinstance(bert_record, Mapping) or not isinstance(mpnet_record, Mapping):
        raise NativeSweepSemanticError("semantic model records must be objects")
    try:
        bert_path = semantic_eval._resolve_semantic_model_path(
            bert_record.get("local_path"),
            manifest_path=manifest_path,
            label="semantic manifest BERTScore",
        )
        mpnet_path = semantic_eval._resolve_semantic_model_path(
            mpnet_record.get("local_path"),
            manifest_path=manifest_path,
            label="semantic manifest MPNet",
        )
        bert = semantic_eval.BERTScoreBackend(
            bert_path,
            str(bert_record.get("directory_sha256")),
            num_layers=int(bert_record.get("num_layers")),
            batch_size=batch_size,
            device=device,
            verify_checksum=False,
        )
        mpnet = semantic_eval.MPNetCosineBackend(
            mpnet_path,
            str(mpnet_record.get("directory_sha256")),
            batch_size=batch_size,
            device=device,
            verify_checksum=False,
        )
        formal = semantic_eval._validate_formal_semantic_backends(
            semantic_manifest_path=manifest_path,
            semantic_scorers=(bert, mpnet),
            evaluation_id=source_evaluation_id,
        )
    except Exception as exc:
        raise NativeSweepSemanticError(
            f"formal semantic backend validation failed: {exc}"
        ) from exc
    return bert, mpnet, {
        "binding": {
            **_file_binding(manifest_path),
            "payload_sha256": payload_sha,
        },
        "source_evaluation_id": source_evaluation_id,
        "network_at_scoring_time": False,
        "models": formal["models"],
        "required_scorer_ids": formal["required_scorer_ids"],
    }


def _delivery_valid(row: Mapping[str, Any]) -> bool:
    text = row.get("generated_text")
    answer = row.get("answer")
    if not isinstance(text, str) or not isinstance(answer, str):
        raise NativeSweepSemanticError("deep-validated row lost persisted text")
    delimiter = native_eval.native_probe.BOUNDARY
    reasoning = text.split(delimiter, 1)[0].strip() if text.count(delimiter) == 1 else ""
    return all(
        (
            bool(row.get("hit_eos")),
            not bool(row.get("cap_reached")),
            row.get("think_boundary_count") == 1,
            row.get("exact_boundary_delimiter_count") == 1,
            bool(reasoning),
            bool(answer.strip()),
            bool(row.get("final_answer_single_paragraph")),
        )
    )


def _surface_payload(values: Counter[str]) -> list[list[object]]:
    return [[surface, values[surface]] for surface in sorted(values)]


def _signed_number_surfaces(text: str) -> Counter[str]:
    """Extract number surfaces without treating terminal periods as decimals."""

    return Counter(match.group(1) for match in _SIGNED_NUMBER_SURFACE_RE.finditer(text))


def _corrected_signed_number_metrics(source: str, answer: str) -> dict[str, Any]:
    source_values = _signed_number_surfaces(source)
    answer_values = _signed_number_surfaces(answer)
    missing = source_values - answer_values
    unsupported = answer_values - source_values
    canonical_json = native_eval.native_probe.common_probe.canonical_json
    sha256_text = native_eval.native_probe.common_probe.sha256_text
    return {
        "policy": SIGNED_NUMBER_SURFACE_POLICY,
        "source_surface_sha256": sha256_text(
            canonical_json(_surface_payload(source_values))
        ),
        "answer_surface_sha256": sha256_text(
            canonical_json(_surface_payload(answer_values))
        ),
        "source_occurrences": sum(source_values.values()),
        "answer_occurrences": sum(answer_values.values()),
        "missing_occurrences": sum(missing.values()),
        "unsupported_occurrences": sum(unsupported.values()),
        "missing_surfaces": _surface_payload(missing),
        "unsupported_surfaces": _surface_payload(unsupported),
        "preserved": not missing and not unsupported,
    }


def _recompute_core_hard_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute semantic eligibility from sealed text, not stored quality_valid."""

    source = row.get("source_analysis")
    answer = row.get("answer")
    text = row.get("generated_text")
    if not all(isinstance(value, str) for value in (source, answer, text)):
        raise NativeSweepSemanticError("core hard gate requires persisted source/answer")
    delivery_valid = _delivery_valid(row)
    feature_metrics = native_eval.native_probe._feature_metrics(source, answer)
    structure_metrics = native_eval._native_structure_metrics(text)
    signed_metrics = _corrected_signed_number_metrics(source, answer)
    try:
        full_repetition = float(row["full_token_4gram_repetition"])
        tail_repetition = float(row["tail_token_4gram_repetition"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeSweepSemanticError(
            "core hard gate requires persisted repetition metrics"
        ) from exc
    if not math.isfinite(full_repetition) or not math.isfinite(tail_repetition):
        raise NativeSweepSemanticError("core hard gate repetition metric is non-finite")
    degeneration_free = all(
        (
            not bool(row.get("strict_periodic_tail")),
            full_repetition < native_eval.native_probe.FULL_REPETITION_LIMIT,
            tail_repetition < native_eval.native_probe.TAIL_REPETITION_LIMIT,
        )
    )
    preregistered_failures: list[str] = []
    if not delivery_valid:
        preregistered_failures.append("delivery_invalid")
    if not bool(structure_metrics["native_structure_valid"]):
        preregistered_failures.extend(
            f"native_structure:{failure}"
            for failure in structure_metrics["native_structure_failures"]
        )
    if not bool(feature_metrics["numeric_multiset_preserved"]):
        preregistered_failures.append("numeric_multiset_not_preserved")
    if not bool(feature_metrics["date_set_preserved"]):
        preregistered_failures.append("date_set_not_preserved")
    if not degeneration_free:
        preregistered_failures.append("degeneration_detected")
    preregistered_failures = sorted(set(preregistered_failures))
    strict_extended_failures = list(preregistered_failures)
    if not bool(signed_metrics["preserved"]):
        strict_extended_failures.append(
            "corrected_signed_number_surface_not_preserved"
        )
    strict_extended_failures = sorted(set(strict_extended_failures))
    stored_quality_valid = row.get("quality_valid")
    stored_quality_failures = row.get("quality_failures")
    if not isinstance(stored_quality_valid, bool) or not isinstance(
        stored_quality_failures, list
    ) or any(not isinstance(value, str) for value in stored_quality_failures):
        raise NativeSweepSemanticError("stored runner quality audit is malformed")
    removed_failures = sorted(
        set(stored_quality_failures) - set(strict_extended_failures)
    )
    added_failures = sorted(
        set(strict_extended_failures) - set(stored_quality_failures)
    )
    preregistered_core_valid = not preregistered_failures
    strict_extended_core_valid = not strict_extended_failures
    return {
        "delivery_valid": delivery_valid,
        "preregistered_core_valid": preregistered_core_valid,
        "preregistered_core_failures": preregistered_failures,
        "extended_signed_surface_valid": bool(signed_metrics["preserved"]),
        "strict_extended_core_valid": strict_extended_core_valid,
        "strict_extended_core_failures": strict_extended_failures,
        # Backward-compatible aliases for the original strict extended policy.
        "core_hard_valid": strict_extended_core_valid,
        "core_hard_failures": strict_extended_failures,
        "degeneration_free": degeneration_free,
        "numeric_multiset_preserved": bool(
            feature_metrics["numeric_multiset_preserved"]
        ),
        "date_set_preserved": bool(feature_metrics["date_set_preserved"]),
        "native_structure_valid": bool(
            structure_metrics["native_structure_valid"]
        ),
        "corrected_signed_number_metrics": signed_metrics,
        "stored_quality_valid": stored_quality_valid,
        "stored_quality_failures": list(stored_quality_failures),
        "stored_vs_preregistered": {
            "validity_changed": stored_quality_valid != preregistered_core_valid,
        },
        "stored_vs_recomputed": {
            "validity_changed": stored_quality_valid != strict_extended_core_valid,
            "failure_set_changed": bool(removed_failures or added_failures),
            "removed_stored_failures": removed_failures,
            "added_recomputed_failures": added_failures,
        },
    }


def _metric_vector(value: Any, *, expected: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise NativeSweepSemanticError(f"{label} did not return a metric vector")
    if len(value) != expected:
        raise NativeSweepSemanticError(
            f"{label} result length drift: expected={expected}, observed={len(value)}"
        )
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise NativeSweepSemanticError(f"{label}[{index}] is not numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise NativeSweepSemanticError(
                f"{label}[{index}] is not numeric"
            ) from exc
        if not math.isfinite(number) or number < -1.000001 or number > 1.000001:
            raise NativeSweepSemanticError(
                f"{label}[{index}] is outside the finite similarity range"
            )
        result.append(number)
    return result


def _score_semantic_pairs(
    *, work_items: Sequence[tuple[str, Mapping[str, Any]]], bert: Any, mpnet: Any
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if not work_items:
        return [], {
            "status": "skipped",
            "reason": "no_delivery_and_recomputed_core_hard_valid_rows",
            "scored_pairs": 0,
            "bertscore": None,
            "mpnet": None,
        }
    candidates = [str(row["answer"]) for _, row in work_items]
    references = [str(row["reference_minutes"]) for _, row in work_items]
    try:
        bert_values = bert.score(candidates, references)
        mpnet_values = mpnet.score(candidates, references)
    except Exception as exc:
        raise NativeSweepSemanticError(f"semantic inference failed: {exc}") from exc
    if not isinstance(bert_values, Mapping) or set(bert_values) != {
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    }:
        raise NativeSweepSemanticError("BERTScore metric inventory drift")
    vectors = {
        metric: _metric_vector(bert_values[metric], expected=len(work_items), label=metric)
        for metric in (
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
        )
    }
    vectors["mpnet_cosine"] = _metric_vector(
        mpnet_values, expected=len(work_items), label="mpnet_cosine"
    )
    scores = [
        {metric: vectors[metric][index] for metric in METRICS}
        for index in range(len(work_items))
    ]
    bert_metadata = (
        bert.semantic_metadata() if hasattr(bert, "semantic_metadata") else None
    )
    mpnet_metadata = (
        mpnet.semantic_metadata() if hasattr(mpnet, "semantic_metadata") else None
    )
    for name, metadata in (
        ("bertscore", bert_metadata),
        ("mpnet", mpnet_metadata),
    ):
        if not isinstance(metadata, Mapping):
            raise NativeSweepSemanticError(f"{name} backend metadata is missing")
        chunk_audit = metadata.get("chunk_audit")
        if not isinstance(chunk_audit, Mapping) or chunk_audit.get(
            "silent_truncation"
        ) is not False:
            raise NativeSweepSemanticError(
                f"{name} long-text audit did not prove zero silent truncation"
            )
    return scores, {
        "status": "complete",
        "scored_pairs": len(work_items),
        "bertscore": dict(bert_metadata),
        "mpnet": dict(mpnet_metadata),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _cohort_rows(
    rows: Sequence[Mapping[str, Any]], cohort: str
) -> list[Mapping[str, Any]]:
    if cohort == "all_n12":
        return list(rows)
    if cohort == "non_identity_n11":
        return [row for row in rows if not bool(row["normalized_identity"])]
    if cohort == "valid_only":
        return [row for row in rows if bool(row["semantic_eligible"])]
    raise NativeSweepSemanticError(f"unknown semantic cohort: {cohort}")


def _summarize_run(
    rows: Sequence[Mapping[str, Any]], *, scoring_policy_version: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort in COHORTS:
        selected = _cohort_rows(rows, cohort)
        if cohort == "all_n12" and len(selected) != EXPECTED_SAMPLE_COUNT:
            raise NativeSweepSemanticError("all-N12 cohort size drift")
        if cohort == "non_identity_n11" and len(selected) != EXPECTED_NON_IDENTITY_COUNT:
            raise NativeSweepSemanticError("non-identity-N11 cohort size drift")
        metric_field = "raw_metrics" if cohort == "valid_only" else "penalized_metrics"
        failure_counts: Counter[str] = Counter()
        for row in selected:
            failure_counts.update(row["penalty_reasons"])
        result[cohort] = {
            "scoring_policy": (
                "eligible-rows-only"
                if cohort == "valid_only"
                else scoring_policy_version
            ),
            "cases": len(selected),
            "semantic_eligible_cases": sum(
                bool(row["semantic_eligible"]) for row in selected
            ),
            "penalized_cases": sum(
                not bool(row["semantic_eligible"]) for row in selected
            ),
            "delivery_valid_cases": sum(bool(row["delivery_valid"]) for row in selected),
            "stored_runner_quality_valid_cases": sum(
                bool(row["stored_runner_quality_valid"]) for row in selected
            ),
            "recomputed_core_hard_valid_cases": sum(
                bool(row["recomputed_core_hard_valid"]) for row in selected
            ),
            "preregistered_core_valid_cases": sum(
                bool(row["preregistered_core_valid"]) for row in selected
            ),
            "extended_signed_surface_valid_cases": sum(
                bool(row["extended_signed_surface_valid"]) for row in selected
            ),
            "stored_vs_recomputed_discrepancy_cases": sum(
                bool(row["stored_vs_recomputed"]["validity_changed"])
                or bool(row["stored_vs_recomputed"]["failure_set_changed"])
                for row in selected
            ),
            "stored_vs_preregistered_validity_discrepancy_cases": sum(
                bool(row["stored_vs_preregistered"]["validity_changed"])
                for row in selected
            ),
            "sample_ids": [row["sample_id"] for row in selected],
            "penalty_reason_counts": dict(sorted(failure_counts.items())),
            "metrics": {
                metric: _numeric_summary(
                    [float(row[metric_field][metric]) for row in selected]
                )
                for metric in METRICS
            },
        }
    return result


def _paired_contrast(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    candidate = {str(row["sample_id"]): row for row in candidate_rows}
    if list(baseline) != list(candidate):
        raise NativeSweepSemanticError("paired semantic sample order drift")
    result: dict[str, Any] = {}
    for cohort in COHORTS:
        sample_ids = list(baseline)
        if cohort == "non_identity_n11":
            sample_ids = [
                sample_id
                for sample_id in sample_ids
                if not bool(baseline[sample_id]["normalized_identity"])
            ]
        elif cohort == "valid_only":
            sample_ids = [
                sample_id
                for sample_id in sample_ids
                if bool(baseline[sample_id]["semantic_eligible"])
                and bool(candidate[sample_id]["semantic_eligible"])
            ]
        metric_field = "raw_metrics" if cohort == "valid_only" else "penalized_metrics"
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            deltas = [
                float(candidate[sample_id][metric_field][metric])
                - float(baseline[sample_id][metric_field][metric])
                for sample_id in sample_ids
            ]
            metrics[metric] = {
                **_numeric_summary(deltas),
                "wins": sum(delta > 1e-12 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                "losses": sum(delta < -1e-12 for delta in deltas),
            }
        result[cohort] = {
            "paired_cases": len(sample_ids),
            "sample_ids": sample_ids,
            "candidate_minus_chk1": metrics,
        }
    return result


def build_semantic_sweep(
    *,
    runs: Mapping[str, Mapping[str, Any]],
    bert: Any,
    mpnet: Any,
    semantic_provenance: Mapping[str, Any],
    sample_manifest_binding: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    candidate_steps: Sequence[int] = REQUIRED_CHECKPOINT_STEPS,
    evaluation_id: str = EVALUATION_ID,
    candidate_representation: str = "adapter_overlay",
) -> dict[str, Any]:
    """Score validated runs and construct the sealed row-level/summary payload."""

    steps = tuple(candidate_steps)
    if (
        not steps
        or len(set(steps)) != len(steps)
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in steps
        )
    ):
        raise NativeSweepSemanticError("candidate checkpoint inventory is invalid")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise NativeSweepSemanticError("semantic evaluation ID is invalid")
    if candidate_representation not in {"adapter_overlay", "exact_merged"}:
        raise NativeSweepSemanticError("candidate representation is invalid")
    eligibility_basis = (
        "preregistered_core"
        if candidate_representation == "exact_merged"
        else "strict_extended_core"
    )
    scoring_policy_version = (
        PREREGISTERED_SCORING_POLICY_VERSION
        if eligibility_basis == "preregistered_core"
        else SCORING_POLICY_VERSION
    )
    expected_run_ids = {f"cp{step}" for step in steps}
    unexpected_run_ids = set(runs) - expected_run_ids - {BASELINE_RUN_ID}
    if unexpected_run_ids:
        raise NativeSweepSemanticError(
            f"unexpected run in semantic sweep: {sorted(unexpected_run_ids)}"
        )
    observed_candidates = {
        run_id for run_id, run in runs.items() if run.get("role") == "candidate"
    }
    if observed_candidates != expected_run_ids:
        raise NativeSweepSemanticError("prepared candidate run inventory drift")

    work_items: list[tuple[str, Mapping[str, Any]]] = []
    gate_audits: dict[tuple[str, str], dict[str, Any]] = {}
    for run_id, run in runs.items():
        rows = run.get("results")
        if not isinstance(rows, list) or len(rows) != EXPECTED_SAMPLE_COUNT:
            raise NativeSweepSemanticError(f"{run_id} is not a prepared N12 run")
        for row in rows:
            sample_id = str(row.get("sample_id"))
            gate = _recompute_core_hard_gate(row)
            eligible = bool(
                gate[
                    "preregistered_core_valid"
                    if eligibility_basis == "preregistered_core"
                    else "strict_extended_core_valid"
                ]
            )
            gate["semantic_eligibility_basis"] = eligibility_basis
            gate["semantic_eligible"] = eligible
            gate["semantic_penalty_reasons"] = list(
                gate[
                    "preregistered_core_failures"
                    if eligibility_basis == "preregistered_core"
                    else "strict_extended_core_failures"
                ]
            )
            gate_audits[(run_id, sample_id)] = gate
            if eligible:
                work_items.append((run_id, row))

    raw_scores, backend_audit = _score_semantic_pairs(
        work_items=work_items, bert=bert, mpnet=mpnet
    )
    backend_audit["pair_order"] = [
        {
            "run_id": run_id,
            "sample_id": row["sample_id"],
            "answer_sha256": row["answer_sha256"],
            "reference_minutes_sha256": row["reference_minutes_sha256"],
        }
        for run_id, row in work_items
    ]
    scored = {
        (run_id, str(row["sample_id"])): values
        for (run_id, row), values in zip(work_items, raw_scores, strict=True)
    }
    zero_metrics = {metric: 0.0 for metric in METRICS}
    row_scores: list[dict[str, Any]] = []
    rows_by_run: dict[str, list[dict[str, Any]]] = {run_id: [] for run_id in runs}
    for run_id, run in runs.items():
        for row_number, row in enumerate(run["results"], start=1):
            sample_id = str(row["sample_id"])
            gate = gate_audits[(run_id, sample_id)]
            delivery_valid = bool(gate["delivery_valid"])
            eligible = bool(gate["semantic_eligible"])
            raw = scored.get((run_id, sample_id))
            if eligible is not (raw is not None):
                raise NativeSweepSemanticError("semantic work-item/result mapping drift")
            output_row = {
                "schema_version": ROW_SCHEMA_VERSION,
                "run_id": run_id,
                "role": run["role"],
                "checkpoint_step": run["checkpoint_step"],
                "stage_id": run["stage_id"],
                "model_label": run["model_label"],
                "source_generation_row": row_number,
                "sample_id": sample_id,
                "length_bucket": row["length_bucket"],
                "analysis_reference_exact_identity": bool(
                    row["analysis_reference_exact_identity"]
                ),
                "normalized_identity": bool(row["normalized_identity"]),
                "completion_sha256": row["completion_sha256"],
                "answer_sha256": row["answer_sha256"],
                "reference_minutes_sha256": row["reference_minutes_sha256"],
                "delivery_valid": delivery_valid,
                "stored_runner_quality_valid": gate["stored_quality_valid"],
                "stored_runner_quality_failures": gate["stored_quality_failures"],
                "preregistered_core_valid": gate["preregistered_core_valid"],
                "preregistered_core_failures": gate[
                    "preregistered_core_failures"
                ],
                "extended_signed_surface_valid": gate[
                    "extended_signed_surface_valid"
                ],
                "strict_extended_core_valid": gate[
                    "strict_extended_core_valid"
                ],
                "strict_extended_core_failures": gate[
                    "strict_extended_core_failures"
                ],
                "recomputed_core_hard_valid": gate["core_hard_valid"],
                "recomputed_core_hard_failures": gate["core_hard_failures"],
                "recomputed_core_metrics": {
                    "degeneration_free": gate["degeneration_free"],
                    "numeric_multiset_preserved": gate[
                        "numeric_multiset_preserved"
                    ],
                    "date_set_preserved": gate["date_set_preserved"],
                    "native_structure_valid": gate["native_structure_valid"],
                    "corrected_signed_number_metrics": gate[
                        "corrected_signed_number_metrics"
                    ],
                },
                "stored_vs_preregistered": gate["stored_vs_preregistered"],
                "stored_vs_recomputed": gate["stored_vs_recomputed"],
                "semantic_eligibility_basis": gate[
                    "semantic_eligibility_basis"
                ],
                "semantic_eligible": eligible,
                "penalty_reasons": gate["semantic_penalty_reasons"],
                "raw_metrics": dict(raw) if raw is not None else None,
                "penalized_metrics": dict(raw) if raw is not None else dict(zero_metrics),
                "cohort_membership": {
                    "all_n12": True,
                    "non_identity_n11": not bool(row["normalized_identity"]),
                    "valid_only": eligible,
                },
            }
            row_scores.append(output_row)
            rows_by_run[run_id].append(output_row)

    summaries = {
        run_id: {
            "role": runs[run_id]["role"],
            "checkpoint_step": runs[run_id]["checkpoint_step"],
            "model_label": runs[run_id]["model_label"],
            "cohorts": _summarize_run(
                rows, scoring_policy_version=scoring_policy_version
            ),
        }
        for run_id, rows in rows_by_run.items()
    }
    contrasts: dict[str, Any] = {}
    if BASELINE_RUN_ID in rows_by_run:
        contrasts = {
            run_id: _paired_contrast(
                rows_by_run[BASELINE_RUN_ID], rows_by_run[run_id]
            )
            for run_id in (f"cp{step}" for step in steps)
        }

    run_inputs = {
        run_id: {
            "role": run["role"],
            "checkpoint_step": run["checkpoint_step"],
            "stage_id": run["stage_id"],
            "model_label": run["model_label"],
            "run_manifest": dict(run["manifest_binding"]),
            "model": run["manifest"].get("model"),
            "adapter": run["manifest"].get("adapter"),
            "generations": run["manifest"].get("artifacts", {}).get("generations"),
        }
        for run_id, run in runs.items()
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_id": evaluation_id,
        "task_contract_id": native_eval.TASK_CONTRACT_ID,
        "operation": "semantic_scoring_only_no_generation",
        "candidate_checkpoints": list(steps),
        "candidate_representation": candidate_representation,
        "baseline_included": BASELINE_RUN_ID in runs,
        "sample_manifest": dict(sample_manifest_binding),
        "scoring_contract": {
            "candidate_text": "answer",
            "reference_text": "reference_minutes",
            "semantic_eligibility": (
                "independently_recomputed_preregistered_core_valid"
                if eligibility_basis == "preregistered_core"
                else "independently_recomputed_strict_extended_core_valid"
            ),
            "semantic_eligibility_basis": eligibility_basis,
            "stored_quality_valid_is_authoritative": False,
            "signed_number_surface_policy": SIGNED_NUMBER_SURFACE_POLICY,
            "signed_number_surface_regex": _SIGNED_NUMBER_SURFACE_RE.pattern,
            "preregistered_core_components": [
                "delivery",
                "native_structure",
                "numeric_multiset_fidelity",
                "date_set_fidelity",
                "degeneration_limits",
            ],
            "extended_signed_surface_role": (
                "diagnostic_only_not_an_independent_zero_penalty_gate"
                if eligibility_basis == "preregistered_core"
                else "strict_extended_core_gate"
            ),
            "all_n12": {
                "cases": EXPECTED_SAMPLE_COUNT,
                "invalid_policy": "all semantic metrics fixed to 0",
            },
            "non_identity_n11": {
                "cases": EXPECTED_NON_IDENTITY_COUNT,
                "identity_field": "normalized_identity",
                "invalid_policy": "all semantic metrics fixed to 0",
            },
            "valid_only": {
                "cases": "variable",
                "invalid_policy": "excluded",
            },
            "policy_version": scoring_policy_version,
            "strict_extended_policy_version": SCORING_POLICY_VERSION,
            "metrics": list(METRICS),
            "statistical_inference_authorized": False,
            "weighted_composite_score_authorized": False,
        },
        "semantic_models": dict(semantic_provenance),
        "semantic_backend_audit": backend_audit,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "sources": dict(source_bindings),
        "run_inputs": run_inputs,
        "row_scores": row_scores,
        "summaries": summaries,
        "paired_contrasts_vs_chk1": contrasts,
        "limitations": [
            "N12 is a deterministic diagnostic sample, not a population inference set.",
            "The Minutes-style references are synthetic teacher targets, not official Minutes.",
            "BERTScore and MPNet measure reference similarity, not factual correctness.",
            "One identity row is excluded from the non-identity-N11 view.",
            "No weighted semantic composite is used to select a checkpoint.",
        ],
    }
    return seal_manifest(payload)


def score_sweep(
    *,
    candidate_manifests: Sequence[Path],
    baseline_manifest: Path | None,
    sample_manifest: Path,
    sample_manifest_sha256: str,
    semantic_manifest: Path,
    output: Path,
    semantic_device: str,
    semantic_batch_size: int,
    merged_finalist_manifests: Sequence[Path] | None = None,
) -> dict[str, Any]:
    output_path = output.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise NativeSweepSemanticError(
            f"refusing to overwrite semantic artifact: {output_path}"
        )
    if semantic_device.startswith("cuda"):
        visible = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if semantic_device != "cuda:0" or visible != ["0"]:
            raise NativeSweepSemanticError(
                "CUDA semantic scoring requires CUDA_VISIBLE_DEVICES=0 and "
                "--semantic-device cuda:0"
            )
    finalist_paths = tuple(merged_finalist_manifests or ())
    if finalist_paths:
        if candidate_manifests:
            raise NativeSweepSemanticError(
                "candidate and merged-finalist manifest modes are mutually exclusive"
            )
        if baseline_manifest is None:
            raise NativeSweepSemanticError(
                "merged-finalist mode requires --baseline-manifest for chk1 contrasts"
            )
        runs = load_merged_finalist_runs(
            finalist_manifests=finalist_paths,
            baseline_manifest=baseline_manifest,
            sample_manifest=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        )
        candidate_steps = MERGED_FINALIST_CHECKPOINT_STEPS
        evaluation_id = MERGED_FINALIST_EVALUATION_ID
        candidate_representation = "exact_merged"
    else:
        runs = load_sweep_runs(
            candidate_manifests=candidate_manifests,
            baseline_manifest=baseline_manifest,
            sample_manifest=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        )
        candidate_steps = REQUIRED_CHECKPOINT_STEPS
        evaluation_id = EVALUATION_ID
        candidate_representation = "adapter_overlay"
    for run_id, run in runs.items():
        manifest_binding = run.get("manifest_binding")
        if not isinstance(manifest_binding, Mapping):
            raise NativeSweepSemanticError(f"{run_id} has no run-manifest binding")
        run_parent = Path(str(manifest_binding.get("path"))).expanduser().resolve().parent
        if output_path == run_parent or run_parent in output_path.parents:
            raise NativeSweepSemanticError(
                f"semantic output must be outside generation run directory: {run_parent}"
            )
    source_bindings = {
        "semantic_scorer": _file_binding(Path(__file__)),
        "native_run_validator": _file_binding(Path(native_eval.__file__)),
        "semantic_backend_implementation": _file_binding(Path(semantic_eval.__file__)),
    }
    bert, mpnet, semantic_provenance = load_formal_semantic_backends(
        semantic_manifest_path=semantic_manifest,
        batch_size=semantic_batch_size,
        device=semantic_device,
    )
    sample_path = sample_manifest.expanduser().resolve()
    sample_value = _read_json(sample_path)
    try:
        sample_payload_sha = validate_manifest_integrity(sample_value)
    except Exception as exc:
        raise NativeSweepSemanticError(
            f"sample manifest integrity failed after run validation: {exc}"
        ) from exc
    if _sha256_file(sample_path) != sample_manifest_sha256:
        raise NativeSweepSemanticError("sample manifest hash changed before scoring")
    result = build_semantic_sweep(
        runs=runs,
        bert=bert,
        mpnet=mpnet,
        semantic_provenance=semantic_provenance,
        sample_manifest_binding={
            **_file_binding(sample_path),
            "payload_sha256": sample_payload_sha,
        },
        source_bindings=source_bindings,
        candidate_steps=candidate_steps,
        evaluation_id=evaluation_id,
        candidate_representation=candidate_representation,
    )
    for label, binding in source_bindings.items():
        _assert_file_binding_unchanged(binding, label=label)
    _assert_run_sources_unchanged(runs)
    _assert_file_binding_unchanged(
        semantic_provenance["binding"], label="semantic manifest"
    )
    if _sha256_file(sample_path) != sample_manifest_sha256:
        raise NativeSweepSemanticError("sample manifest changed during scoring")
    _write_new_sealed_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument(
        "--candidate-manifest",
        action="append",
        type=Path,
        help="Repeat exactly four times for cp170, cp200, cp230, and cp318.",
    )
    inventory.add_argument(
        "--merged-finalist-manifest",
        action="append",
        type=Path,
        help=(
            "Repeat exactly twice for the exact-merged cp200 and cp318 native runs; "
            "requires --baseline-manifest."
        ),
    )
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest-sha256", required=True)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--semantic-device", default="cuda:0")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = score_sweep(
            candidate_manifests=args.candidate_manifest or (),
            baseline_manifest=args.baseline_manifest,
            sample_manifest=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            semantic_manifest=args.semantic_manifest,
            output=args.output,
            semantic_device=args.semantic_device,
            semantic_batch_size=args.semantic_batch_size,
            merged_finalist_manifests=args.merged_finalist_manifest,
        )
    except NativeSweepSemanticError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "evaluation_id": result["evaluation_id"],
                "output": str(args.output.expanduser().resolve()),
                "rows": len(result["row_scores"]),
                "payload_sha256": result["integrity"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
