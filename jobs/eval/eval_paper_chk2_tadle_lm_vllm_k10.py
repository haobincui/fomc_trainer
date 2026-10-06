"""End-to-end orchestration for the paper chk-2 Tadle-form LM K=10 diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_tadle_form_lm_ff_futures_k10_v1_20260902"
)
COMMON_MODULE = "jobs.eval.paper_chk2_tadle_lm_k10_common"
STATISTICS_MODULE = "jobs.eval.paper_chk2_tadle_lm_k10_statistics"
REPORT_MODULE = "jobs.eval.render_paper_chk2_tadle_lm_k10"
VLLM_LAUNCHER = ROOT / "run/eval_paper_chk2_tadle_lm_vllm_k10_dual_gpu.sh"
FOMC_PYTHON = (
    Path(
        os.environ.get(
            "FOMC_TRAINER_PYTHON",
            "/home/haobin_cui/.conda/envs/fomc_trainer/bin/python",
        )
    )
    .expanduser()
    .resolve()
)
VLLM_PYTHON = (
    Path(
        os.environ.get(
            "VLLM_PYTHON_BIN",
            "/home/haobin_cui/.conda/envs/vllm_env/bin/python",
        )
    )
    .expanduser()
    .resolve()
)
PIPELINE_LOCK = Path("/tmp/fomc_trainer_paper_chk2_tadle_lm_k10.lock")
FINAL_SCHEMA = "paper-chk2-tadle-form-lm-k10-final-evaluation-manifest-v1"


class TadleOrchestratorError(RuntimeError):
    """A prerequisite, subprocess, or immutable-artifact check failed."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise TadleOrchestratorError(f"missing/unsafe artifact: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise TadleOrchestratorError(f"missing/unsafe artifact: {resolved}")
    shown: Path | str = resolved
    if relative_to is not None:
        shown = resolved.relative_to(relative_to.expanduser().resolve())
    return {
        "path": str(shown),
        "bytes": resolved.stat().st_size,
        "sha256": _sha_file(resolved),
    }


def _read_object(path: Path) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise TadleOrchestratorError(f"missing/unsafe JSON object: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise TadleOrchestratorError(f"missing/unsafe JSON object: {resolved}")
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as exc:
        raise TadleOrchestratorError(f"invalid JSON object {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise TadleOrchestratorError(f"JSON artifact is not an object: {resolved}")
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    parent_descriptor = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _jsonl_rows(path: Path) -> int:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise TadleOrchestratorError(f"missing/unsafe JSONL artifact: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise TadleOrchestratorError(f"missing/unsafe JSONL artifact: {resolved}")
    count = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                raise TadleOrchestratorError(
                    f"invalid JSONL framing: {resolved}:{line_number}"
                )
            try:
                value = json.loads(
                    raw,
                    parse_constant=_reject_json_constant,
                )
            except ValueError as exc:
                raise TadleOrchestratorError(
                    f"invalid JSONL row: {resolved}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise TadleOrchestratorError(
                    f"JSONL row is not an object: {resolved}:{line_number}"
                )
            count += 1
    return count


def _verify_bound_file(
    root: Path,
    record: Any,
    relative_path: str,
    *,
    rows: int | None = None,
    absolute_record_path: bool = False,
) -> None:
    expected_record_path = (
        str((root / relative_path).resolve()) if absolute_record_path else relative_path
    )
    if not isinstance(record, dict) or record.get("path") != expected_record_path:
        raise TadleOrchestratorError(f"artifact binding path drift: {relative_path}")
    observed = _binding(root / relative_path, relative_to=root)
    observed["path"] = expected_record_path
    for key in ("path", "bytes", "sha256"):
        if record.get(key) != observed.get(key):
            raise TadleOrchestratorError(
                f"artifact binding drift: {relative_path}/{key}"
            )
    if rows is not None:
        if record.get("rows") != rows or _jsonl_rows(root / relative_path) != rows:
            raise TadleOrchestratorError(f"artifact row-count drift: {relative_path}")


def _validated_coverage(
    root: Path, objects: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Derive the final coverage only after replaying terminal manifest bindings."""

    # Re-run the statistics module's own committed-phase validator so the
    # terminal seal closes the full documents -> sentiment -> estimate ->
    # bootstrap chain, including supplemental ledgers that are not rendered in
    # the paper tables.
    from jobs.eval import paper_chk2_tadle_lm_k10_statistics as statistics_contract

    for phase, relative_path in (
        ("sentiment", "sentiment/manifest.json"),
        ("estimate", "estimation/estimate_receipt.json"),
        ("statistics", "estimation/statistics_manifest.json"),
    ):
        try:
            statistics_contract.verify_committed_phase(
                root, root / relative_path, phase=phase
            )
        except Exception as exc:
            raise TadleOrchestratorError(
                f"{phase} committed-phase replay failed: {exc}"
            ) from exc

    evaluation_id = "paper-chk2-cp50-tadle-form-lm-ff-futures-k10-v1"
    preparation = objects["preparation"]
    generation = objects["generation"]
    documents = objects["documents"]
    sentiment = objects["sentiment"]
    estimate = objects["estimate"]
    statistics = objects["statistics"]
    report = objects["report"]
    expected_contracts = {
        "preparation": (
            "paper-chk2-tadle-lm-k10-preparation-v1",
            "prepared",
        ),
        "generation": (
            "paper-chk2-tadle-lm-k10-all-model-validation-v1",
            "complete",
        ),
        "documents": (
            "paper-chk2-tadle-lm-k10-document-manifest-v1",
            "complete",
        ),
        "sentiment": (
            "paper-chk2-tadle-lm-k10-statistics-v1:sentiment-manifest",
            "complete",
        ),
        "estimate": (
            "paper-chk2-tadle-lm-k10-statistics-v1:estimate-manifest",
            "complete",
        ),
        "statistics": (
            "paper-chk2-tadle-lm-k10-statistics-v1:statistics-manifest",
            "complete",
        ),
        "report": (
            "paper-chk2-tadle-form-lm-k10-report-v1:manifest",
            "complete",
        ),
    }
    for name, (schema, status) in expected_contracts.items():
        if (
            objects[name].get("schema_version") != schema
            or objects[name].get("status") != status
        ):
            raise TadleOrchestratorError(f"{name} terminal contract drift")
    for name in ("preparation", "generation", "statistics"):
        observed_id = objects[name].get("evaluation_id")
        if observed_id is not None and observed_id != evaluation_id:
            raise TadleOrchestratorError(f"{name} evaluation ID drift")

    population = preparation.get("population") or {}
    generation_contract = preparation.get("generation") or {}
    document_coverage = documents.get("coverage") or {}
    estimate_artifacts = estimate.get("artifacts") or {}
    statistics_artifacts = statistics.get("artifacts") or {}
    generation_rows = int(generation.get("generation_rows", -1))
    generated_documents = int(document_coverage.get("documents_total", -1))
    bootstrap_draws = int(statistics.get("bootstrap_draws", -1))

    _verify_bound_file(
        root,
        generation.get("combined_generation_rows"),
        "generation/generation_rows.jsonl",
        rows=generation_rows,
        absolute_record_path=True,
    )
    _verify_bound_file(
        root,
        documents.get("generated_documents"),
        "documents/generated_documents.jsonl",
        rows=generated_documents,
        absolute_record_path=True,
    )
    for name, relative_path, rows in (
        ("analysis_panel", "estimation/analysis_panel.jsonl", 40_700),
        ("coefficient_estimates", "estimation/coefficient_estimates.jsonl", 16),
    ):
        _verify_bound_file(root, estimate_artifacts.get(name), relative_path, rows=rows)
    for name, relative_path, rows in (
        (
            "model_minus_official_contrasts",
            "estimation/model_minus_official_contrasts.jsonl",
            36,
        ),
        ("absolute_distances", "estimation/absolute_distances.jsonl", 12),
        ("distance_gains", "estimation/distance_gains.jsonl", 8),
        ("bootstrap_draws", "estimation/bootstrap_draws.jsonl", bootstrap_draws),
    ):
        _verify_bound_file(
            root, statistics_artifacts.get(name), relative_path, rows=rows
        )

    report_artifacts = report.get("artifacts") or {}
    expected_report_names = {
        "artifact.json",
        "report.md",
        "report.html",
        "methods.tex",
        "post_beta_table.tex",
        "distance_and_gain_table.tex",
        "results.tex",
    }
    if set(report_artifacts) != expected_report_names:
        raise TadleOrchestratorError("report artifact inventory drift")
    for name in expected_report_names:
        _verify_bound_file(root, report_artifacts[name], f"report/{name}")

    coverage = {
        "generation_meetings": int(population.get("meeting_union", -1)),
        "release_meetings": int(population.get("release_meetings", -1)),
        "estimable_current_events": int(population.get("estimable_current_events", -1)),
        "lag_only_within_estimable_roster": int(
            population.get("lag_only_meetings", -1)
        ),
        "external_leading_lag_buffer_meetings": int(population.get("meeting_union", -1))
        - int(population.get("standardization_meetings", -1)),
        "topics_per_meeting": int(population.get("topics_per_meeting", -1)),
        "replicates": int(generation_contract.get("replicates", -1)),
        "models": len(generation_contract.get("models") or ()),
        "atomic_generation_rows": generation_rows,
        "generated_documents": generated_documents,
        "analysis_panel_rows": int(
            estimate_artifacts.get("analysis_panel", {}).get("rows", -1)
        ),
        "coefficient_rows": int(
            estimate_artifacts.get("coefficient_estimates", {}).get("rows", -1)
        ),
        "model_minus_official_contrast_rows": int(
            statistics_artifacts.get("model_minus_official_contrasts", {}).get(
                "rows", -1
            )
        ),
        "absolute_distance_rows": int(
            statistics_artifacts.get("absolute_distances", {}).get("rows", -1)
        ),
        "distance_gain_rows": int(
            statistics_artifacts.get("distance_gains", {}).get("rows", -1)
        ),
        "bootstrap_draws": bootstrap_draws,
    }
    expected = {
        "generation_meetings": 84,
        "release_meetings": 83,
        "estimable_current_events": 82,
        "lag_only_within_estimable_roster": 2,
        "external_leading_lag_buffer_meetings": 1,
        "topics_per_meeting": 8,
        "replicates": 10,
        "models": 3,
        "atomic_generation_rows": 20_160,
        "generated_documents": 2_520,
        "analysis_panel_rows": 40_700,
        "coefficient_rows": 16,
        "model_minus_official_contrast_rows": 36,
        "absolute_distance_rows": 12,
        "distance_gain_rows": 8,
        "bootstrap_draws": 10_000,
    }
    if coverage != expected:
        raise TadleOrchestratorError(f"terminal coverage drift: {coverage}")
    if sentiment.get("generated_document_rows") != generated_documents:
        raise TadleOrchestratorError("sentiment/document coverage drift")
    return coverage


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    prefixes = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        prefixes.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(prefixes),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _run(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), cwd=ROOT, env=_environment(), check=False)
    if completed.returncode:
        raise TadleOrchestratorError(
            f"command failed ({completed.returncode}): {' '.join(map(str, command))}"
        )


def _python_module(module: str, *arguments: str) -> list[str]:
    if not FOMC_PYTHON.is_file():
        raise TadleOrchestratorError(f"missing fomc_trainer Python: {FOMC_PYTHON}")
    return [str(FOMC_PYTHON), "-u", "-m", module, *arguments]


def _phase_common(phase: str, root: Path, *, resume: bool) -> None:
    command = _python_module(COMMON_MODULE, phase, "--output-root", str(root))
    if resume:
        command.append("--resume")
    _run(command)


def _phase_vllm_common(phase: str, root: Path, *, resume: bool) -> None:
    """Run token-deep-replay phases in their pinned generation runtime."""

    if not VLLM_PYTHON.is_file():
        raise TadleOrchestratorError(f"missing vLLM Python: {VLLM_PYTHON}")
    command = [
        str(VLLM_PYTHON),
        "-u",
        "-m",
        COMMON_MODULE,
        phase,
        "--output-root",
        str(root),
    ]
    if resume:
        command.append("--resume")
    _run(command)


def _phase_generation(phase: str, root: Path) -> None:
    manifest = root / "preparation/evaluation_manifest.json"
    _run([str(VLLM_LAUNCHER), phase, str(manifest), str(root)])


def _phase_statistics(phase: str, root: Path, *, workers: int, resume: bool) -> None:
    command = _python_module(
        STATISTICS_MODULE,
        phase,
        "--evaluation-root",
        str(root),
        "--bootstrap-workers",
        str(workers),
    )
    if resume:
        command.append("--resume")
    _run(command)


def _phase_report(root: Path, *, resume: bool) -> None:
    command = _python_module(REPORT_MODULE, "--evaluation-root", str(root))
    if resume:
        command.append("--resume")
    _run(command)


def seal_or_verify_final_manifest(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    artifacts = {
        "preparation": root / "preparation/evaluation_manifest.json",
        "generation": root / "generation/validation.json",
        "documents": root / "documents/manifest.json",
        "sentiment": root / "sentiment/manifest.json",
        "estimate": root / "estimation/estimate_receipt.json",
        "statistics": root / "estimation/statistics_manifest.json",
        "report": root / "report/report_manifest.json",
    }
    objects = {name: _read_object(path) for name, path in artifacts.items()}
    terminal_status = {
        "preparation": "prepared",
        "generation": "complete",
        "documents": "complete",
        "sentiment": "complete",
        "estimate": "complete",
        "statistics": "complete",
        "report": "complete",
    }
    if any(
        objects[name].get("status") != expected
        for name, expected in terminal_status.items()
    ):
        raise TadleOrchestratorError("a terminal phase manifest is not complete")

    coverage = _validated_coverage(root, objects)
    payload: dict[str, Any] = {
        "schema_version": FINAL_SCHEMA,
        "status": "complete",
        "immutable": True,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "evaluation_id": "paper-chk2-cp50-tadle-form-lm-ff-futures-k10-v1",
        "coverage": coverage,
        "artifacts": {
            name: _binding(path, relative_to=root) for name, path in artifacts.items()
        },
        "implementation": {
            "orchestrator": _binding(Path(__file__)),
            "common": _binding(ROOT / "jobs/eval/paper_chk2_tadle_lm_k10_common.py"),
            "generation": _binding(
                ROOT / "jobs/eval/eval_paper_chk2_tadle_lm_vllm_k10_dual_dp1.py"
            ),
            "statistics": _binding(
                ROOT / "jobs/eval/paper_chk2_tadle_lm_k10_statistics.py"
            ),
            "reporting": _binding(ROOT / "jobs/eval/render_paper_chk2_tadle_lm_k10.py"),
            "dual_gpu_launcher": _binding(VLLM_LAUNCHER),
        },
        "lineage": {
            "loo_generation_rows_reused": False,
            "loo_execution_patterns_reused": True,
            "three_models_fresh_same_prompt_k10": True,
            "chapter2_modified": False,
            "training_overlap_makes_post_period_leakage_safe": False,
            "generated_texts_were_historically_observed": False,
            "exact_tadle_2022_replication": False,
        },
        "mutation_policy": "create_once_then_full_verify_only",
    }
    payload["payload_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()
    manifest = root / "evaluation_manifest.json"
    if manifest.exists() or manifest.is_symlink():
        observed = _read_object(manifest)
        # Creation time is part of the seal; replay uses the stored value.
        payload["created_at_utc"] = observed.get("created_at_utc")
        payload["payload_sha256"] = hashlib.sha256(
            _canonical(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "payload_sha256"
                }
            ).encode("utf-8")
        ).hexdigest()
        if observed != payload:
            raise TadleOrchestratorError(
                "final evaluation manifest differs from full replay"
            )
    else:
        _write_new_json(manifest, payload)
    return {"path": str(manifest), "sha256": _sha_file(manifest), "status": "complete"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "smoke",
            "generate",
            "validate",
            "assemble",
            "sentiment",
            "estimate",
            "bootstrap",
            "report",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_root.expanduser().resolve()
    if args.bootstrap_workers < 1:
        raise TadleOrchestratorError("--bootstrap-workers must be positive")
    PIPELINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOCK.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TadleOrchestratorError(
                "another Tadle K10 pipeline owns the lock"
            ) from exc

        if args.phase in {"prepare", "all"}:
            _phase_common("prepare", root, resume=args.resume)
        if args.phase in {"smoke", "all"}:
            _phase_generation("smoke", root)
        if args.phase in {"generate", "all"}:
            _phase_generation("generate", root)
        if args.phase == "validate":
            _phase_generation("validate", root)
        if args.phase in {"assemble", "all"}:
            _phase_vllm_common("assemble", root, resume=args.resume)
        if args.phase in {"sentiment", "all"}:
            _phase_statistics(
                "sentiment", root, workers=args.bootstrap_workers, resume=args.resume
            )
        if args.phase in {"estimate", "all"}:
            _phase_statistics(
                "estimate", root, workers=args.bootstrap_workers, resume=args.resume
            )
        if args.phase in {"bootstrap", "all"}:
            _phase_statistics(
                "bootstrap", root, workers=args.bootstrap_workers, resume=args.resume
            )
        if args.phase in {"report", "all"}:
            _phase_report(root, resume=args.resume)
            result = seal_or_verify_final_manifest(root)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
