"""End-to-end CLI for the paper chk-2 cp50 Core8 LOO K=10 audit.

The CLI is intentionally a thin orchestrator.  Preparation runs under the
``fomc_trainer`` Python so the frozen tokenizer runtime materializes exact
prompt token IDs.  Generation is delegated to two independent single-GPU
vLLM workers, which consume those IDs without re-rendering the chat template.
Scoring, shared-index inference, and reporting each seal their own artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_core8_loo_vllm_k10_n128_1993_2008_"
    "t06_p095_b10000_v1_20260901"
)
VLLM_LAUNCHER = ROOT / "run/eval_paper_chk2_core8_loo_vllm_k10_dual_gpu.sh"
GENERATION_MODULE = "jobs.eval.eval_paper_chk2_core8_loo_vllm_k10_dual_dp1"
SCORING_MODULE = "jobs.eval.score_paper_chk2_core8_loo_vllm_k10"
REPORT_MODULE = "jobs.eval.render_paper_chk2_core8_loo_report"
PIPELINE_LOCK = Path("/tmp/fomc_trainer_paper_chk2_core8_loo_k10.lock")
FINAL_MANIFEST_SCHEMA = "paper-chk2-core8-loo-final-evaluation-manifest-v1"
FINAL_MANIFEST_NAME = "evaluation_manifest.json"


class Core8OrchestratorError(RuntimeError):
    """A phase failed or a required sealed artifact is unavailable."""


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


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8OrchestratorError(f"missing/unsafe {label}: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Core8OrchestratorError(f"invalid {label}: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise Core8OrchestratorError(f"{label} must be a JSON object: {resolved}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8OrchestratorError(f"cannot bind missing/unsafe file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha_file(resolved),
    }


def _verify_binding(record: Any, path: Path, *, label: str) -> None:
    if not isinstance(record, dict):
        raise Core8OrchestratorError(f"final manifest lacks {label} binding")
    expected = _binding(path)
    if record != expected:
        raise Core8OrchestratorError(f"final manifest {label} binding drift")


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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def seal_or_verify_final_manifest(output_root: Path) -> dict[str, Any]:
    """Create or verify the root manifest after every substantive phase closes."""

    root = output_root.expanduser().resolve()
    paths = {
        "preparation": root / "preparation/evaluation_manifest.json",
        "generation": root / "generation/validation.json",
        "score": root / "score/score_manifest.json",
        "report": root / "report/report_manifest.json",
    }
    preparation = _read_object(paths["preparation"], label="preparation manifest")
    generation = _read_object(paths["generation"], label="generation validation")
    score = _read_object(paths["score"], label="score manifest")
    report = _read_object(paths["report"], label="report manifest")
    bootstrap_path = root / "score/bootstrap_results.json"
    bootstrap = _read_object(bootstrap_path, label="bootstrap results")

    if not (
        preparation.get("schema_version")
        == "paper-chk2-core8-loo-preparation-manifest-v1"
        and preparation.get("status") == "complete"
        and preparation.get("sealed") is True
        and preparation.get("immutable") is True
        and preparation.get("coverage", {}).get("generation_rows") == 21_760
    ):
        raise Core8OrchestratorError("preparation final-seal contract drift")
    generation_gates = generation.get("gates")
    if not (
        generation.get("schema_version")
        == "paper-chk2-core8-loo-vllm-k10-validation-v1"
        and generation.get("status") == "complete"
        and generation.get("mode") == "formal"
        and generation.get("model_id") == "paper_chk2"
        and generation.get("model_label")
        == "paper-chk2-cp50-lora-over-chk1-cp200"
        and generation.get("generation_rows") == 21_760
        and generation.get("meetings") == 128
        and generation.get("replicates") == 10
        and isinstance(generation_gates, dict)
        and generation_gates
        and all(value is True for value in generation_gates.values())
    ):
        raise Core8OrchestratorError("generation final-seal contract drift")
    if not (
        score.get("schema_version") == "paper-chk2-core8-loo-score-manifest-v1"
        and score.get("status") == "complete"
        and score.get("immutable") is True
        and score.get("paper_chk2_scope") is True
        and score.get("historical_generation_or_score_rows_reused") is False
        and bootstrap.get("schema_version") == "paper-chk2-core8-loo-statistics-v1"
        and bootstrap.get("status") == "complete"
        and bootstrap.get("coverage", {}).get("generation_rows") == 21_760
        and bootstrap.get("coverage", {}).get("meeting_cells") == 2_048
        and bootstrap.get("coverage", {}).get("primary_topic_arm_metric_cells") == 32
        and bootstrap.get("bootstrap_contract", {}).get("draws") == 10_000
    ):
        raise Core8OrchestratorError("score/statistics final-seal contract drift")
    report_artifacts = report.get("artifacts")
    required_report_artifacts = {
        "canonical_artifact",
        "portable_html",
        "markdown_report",
        "latex_table",
        "latex_results",
    }
    if not (
        report.get("schema_version") == "paper-chk2-core8-loo-report-manifest-v1"
        and report.get("status") == "complete"
        and report.get("immutable") is True
        and isinstance(report_artifacts, dict)
        and set(report_artifacts) == required_report_artifacts
    ):
        raise Core8OrchestratorError("report final-seal contract drift")
    for role, record in report_artifacts.items():
        artifact_path = Path(str(record.get("path", ""))).expanduser().resolve()
        _verify_binding(
            {key: record.get(key) for key in ("path", "bytes", "sha256")},
            artifact_path,
            label=f"report artifact {role}",
        )

    implementation_paths = {
        "orchestrator": Path(__file__),
        "preparation": ROOT / "jobs/eval/paper_chk2_core8_loo_common.py",
        "generation": ROOT / "jobs/eval/eval_paper_chk2_core8_loo_vllm_k10_dual_dp1.py",
        "scoring": ROOT / "jobs/eval/score_paper_chk2_core8_loo_vllm_k10.py",
        "reporting": ROOT / "jobs/eval/render_paper_chk2_core8_loo_report.py",
        "dual_gpu_launcher": VLLM_LAUNCHER,
    }
    manifest_path = root / FINAL_MANIFEST_NAME
    payload: dict[str, Any] = {
        "schema_version": FINAL_MANIFEST_SCHEMA,
        "status": "complete",
        "immutable": True,
        "evaluation_id": "paper-chk2-cp50-core8-loo-vllm-k10-v1",
        "created_at_utc": report.get("created_at_utc"),
        "model": {
            "id": "paper_chk2",
            "label": "paper-chk2-cp50-lora-over-chk1-cp200",
            "checkpoint": 50,
            "dynamic_lora": True,
        },
        "coverage": {
            "meetings": 128,
            "variants_per_meeting": 17,
            "replicates": 10,
            "generation_rows": 21_760,
            "meeting_cells": 2_048,
            "statistical_cells": 32,
            "bootstrap_draws": 10_000,
        },
        "artifacts": {role: _binding(path) for role, path in paths.items()},
        "bootstrap_results": _binding(bootstrap_path),
        "implementation": {
            role: _binding(path) for role, path in implementation_paths.items()
        },
        "interpretation": {
            "target_relative_prompt_sensitivity": True,
            "causal_topic_importance": False,
            "reference_is_synthetic_not_official_minutes": True,
            "generation_gates_are_diagnostic_only": True,
            "historical_core8_panel_is_reused": True,
            "pristine_unseen_test_claim_allowed": False,
        },
        "mutation_policy": "create_once_then_full_verify_only",
    }
    payload["manifest_sha256"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()

    if manifest_path.exists() or manifest_path.is_symlink():
        observed = _read_object(manifest_path, label="final evaluation manifest")
        if observed != payload:
            raise Core8OrchestratorError("final evaluation manifest differs from replay")
    else:
        _write_new_json(manifest_path, payload)
    return {
        "status": "complete",
        "path": str(manifest_path),
        "sha256": _sha_file(manifest_path),
        "payload_sha256": payload["manifest_sha256"],
    }


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(paths),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _run(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command), cwd=ROOT, env=_python_env(), text=True, check=False
    )
    if completed.returncode != 0:
        raise Core8OrchestratorError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _manifest(root: Path) -> Path:
    return root / "preparation/evaluation_manifest.json"


def run_phase(
    phase: str,
    *,
    output_root: Path,
    resume: bool,
    bootstrap_workers: int,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    manifest = _manifest(root)

    if phase == "prepare":
        command = [
            sys.executable,
            "-m",
            GENERATION_MODULE,
            "prepare",
            "--output-root",
            str(root),
        ]
        if resume:
            command.append("--resume")
        _run(command)
        if not manifest.is_file() or manifest.is_symlink():
            raise Core8OrchestratorError("prepare did not seal evaluation_manifest.json")
        return {"status": "prepared", "manifest": str(manifest)}

    if not manifest.is_file() or manifest.is_symlink():
        raise Core8OrchestratorError("run prepare before downstream phases")

    if phase == "smoke":
        _run([str(VLLM_LAUNCHER), "smoke", str(manifest), str(root)])
        return {"status": "smoke_complete", "rows": 68}

    if phase == "generate":
        _run(
            [
                str(VLLM_LAUNCHER),
                "generate",
                str(manifest),
                str(root),
            ]
        )
        return {"status": "generation_complete", "rows": 21_760}

    if phase == "validate":
        _run(
            [
                str(VLLM_LAUNCHER),
                "validate",
                str(manifest),
                str(root),
            ]
        )
        score_manifest = root / "score/score_manifest.json"
        if score_manifest.is_file():
            _run(
                [
                    sys.executable,
                    "-m",
                    SCORING_MODULE,
                    "validate",
                    "--run-root",
                    str(root),
                ]
            )
        report_manifest = root / "report/report_manifest.json"
        if report_manifest.is_file():
            _run(
                [
                    sys.executable,
                    "-m",
                    REPORT_MODULE,
                    "validate",
                    "--run-root",
                    str(root),
                ]
            )
        final_manifest = None
        if score_manifest.is_file() and report_manifest.is_file():
            final_manifest = seal_or_verify_final_manifest(root)
        return {"status": "validated", "final_manifest": final_manifest}

    if phase == "score":
        consolidate = [
            sys.executable,
            "-m",
            SCORING_MODULE,
            "consolidate",
            "--run-root",
            str(root),
        ]
        score = [
            sys.executable,
            "-m",
            SCORING_MODULE,
            "score",
            "--run-root",
            str(root),
            "--metric",
            "all",
        ]
        if resume:
            consolidate.append("--resume")
            score.append("--resume")
        _run(consolidate)
        _run(score)
        return {"status": "scored", "rows": 21_760}

    if phase == "bootstrap":
        statistics = [
            sys.executable,
            "-m",
            SCORING_MODULE,
            "statistics",
            "--run-root",
            str(root),
            "--workers",
            str(bootstrap_workers),
        ]
        if resume:
            statistics.append("--resume")
        _run(statistics)
        return {
            "status": "bootstrapped",
            "draws": 10_000,
            "workers": bootstrap_workers,
        }

    if phase == "report":
        report = [
            sys.executable,
            "-m",
            REPORT_MODULE,
            "render",
            "--run-root",
            str(root),
        ]
        if resume:
            report.append("--resume")
        _run(report)
        return {"status": "reported", "path": str(root / "report/report.html")}

    raise Core8OrchestratorError(f"unsupported phase: {phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "smoke",
            "generate",
            "validate",
            "score",
            "bootstrap",
            "report",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=6,
        choices=range(1, 13),
        metavar="{1..12}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phases = (
        ("prepare", "smoke", "generate", "validate", "score", "bootstrap", "report", "validate")
        if args.phase == "all"
        else (args.phase,)
    )
    results: list[dict[str, Any]] = []
    lock_handle = PIPELINE_LOCK.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Core8OrchestratorError(
                f"another paper chk-2 Core8 pipeline owns {PIPELINE_LOCK}"
            ) from exc
        for phase in phases:
            results.append(
                {
                    "phase": phase,
                    **run_phase(
                        phase,
                        output_root=args.output_root,
                        resume=args.resume,
                        bootstrap_workers=args.bootstrap_workers,
                    ),
                }
            )
    except Core8OrchestratorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        lock_handle.close()
    print(json.dumps({"status": "complete", "phases": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
