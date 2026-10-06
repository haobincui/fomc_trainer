"""End-to-end CLI for the all-391 chk0/chk1/chk2-cp50 similarity audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from jobs.eval import paper_chk2_text_similarity_common as preparation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_text_similarity_all391_chk0_chk1_chk2cp50_"
    "vllm_t06_p09_k10_b1000_v1_20260901"
)
VLLM_LAUNCHER = ROOT / "run/eval_paper_chk2_text_similarity_vllm_k10_dual_gpu.sh"
SEMANTIC_MANIFEST = ROOT / "configs/main/checkpoint_eval_semantic_models.json"


class EvaluationOrchestratorError(RuntimeError):
    pass


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        list(command), cwd=ROOT, env=env, check=False, text=True
    )
    if completed.returncode != 0:
        raise EvaluationOrchestratorError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    path = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        path.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(path),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _manifest(root: Path) -> Path:
    return root / "evaluation_manifest.json"


def run_phase(
    phase: str,
    *,
    output_root: Path,
    resume: bool,
    semantic_device: str,
    semantic_batch_size: int,
    bootstrap_workers: int = 6,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    if phase == "prepare":
        return dict(preparation.prepare(root))

    manifest = _manifest(root)
    if not manifest.is_file():
        raise EvaluationOrchestratorError("run prepare before downstream phases")

    if phase == "smoke":
        _run(
            [str(VLLM_LAUNCHER), "smoke", str(manifest), str(root / "smoke")],
            env=_python_env(),
        )
        return {"status": "smoke_complete", "root": str(root / "smoke")}

    if phase == "generate":
        _run(
            [
                str(VLLM_LAUNCHER),
                "generate",
                str(manifest),
                str(root / "generation"),
            ],
            env=_python_env(),
        )
        command = [
            sys.executable,
            "-m",
            "jobs.eval.paper_chk2_text_similarity_vllm",
            "consolidate",
            "--manifest",
            str(manifest),
            "--generation-output-root",
            str(root / "generation"),
            "--destination-root",
            str(root),
        ]
        if resume:
            command.append("--resume")
        _run(command, env=_python_env())
        return {"status": "generation_complete", "rows": 11_730}

    if phase == "validate":
        _run(
            [
                sys.executable,
                "-m",
                "jobs.eval.paper_chk2_text_similarity_vllm",
                "validate",
                "--manifest",
                str(manifest),
                "--output-root",
                str(root / "generation"),
            ],
            env=_python_env(),
        )
        consolidate = [
            sys.executable,
            "-m",
            "jobs.eval.paper_chk2_text_similarity_vllm",
            "consolidate",
            "--manifest",
            str(manifest),
            "--generation-output-root",
            str(root / "generation"),
            "--destination-root",
            str(root),
            "--resume",
        ]
        _run(consolidate, env=_python_env())
        if (root / "score_manifest.json").is_file():
            _run(
                [
                    sys.executable,
                    "-m",
                    "jobs.eval.paper_chk2_text_similarity_scoring",
                    "validate",
                    "--output-dir",
                    str(root),
                ],
                env=_python_env(),
            )
        return {"status": "validated"}

    common = [
        "--generation-rows",
        str(root / "generation_rows.jsonl"),
        "--samples",
        str(root / "samples.jsonl"),
        "--evaluation-manifest",
        str(root / "complete_evaluation_manifest.json"),
        "--semantic-manifest",
        str(SEMANTIC_MANIFEST),
        "--output-dir",
        str(root),
    ]
    if phase == "score":
        command = [
            sys.executable,
            "-m",
            "jobs.eval.paper_chk2_text_similarity_scoring",
            "score",
            *common,
            "--semantic-device",
            semantic_device,
            "--semantic-batch-size",
            str(semantic_batch_size),
        ]
        if resume:
            command.append("--resume")
        _run(command, env=_python_env())
        return {"status": "scored", "rows": 11_730}
    if phase == "bootstrap":
        command = [
            sys.executable,
            "-m",
            "jobs.eval.paper_chk2_text_similarity_bootstrap_parallel",
            *common,
            "--workers",
            str(bootstrap_workers),
        ]
        if resume:
            command.append("--resume")
        _run(command, env=_python_env())
        return {
            "status": "bootstrapped",
            "draws": 1_000,
            "workers": bootstrap_workers,
        }
    if phase == "report":
        command = [
            sys.executable,
            "-m",
            "jobs.eval.render_paper_chk2_text_similarity_report",
            "--score-root",
            str(root),
            "--report-root",
            str(root / "report"),
        ]
        if resume:
            command.append("--resume")
        _run(command, env=_python_env())
        return {"status": "reported", "path": str(root / "report/report.html")}
    raise EvaluationOrchestratorError(f"unsupported phase: {phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "smoke", "generate", "validate", "score", "bootstrap", "report", "all"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--semantic-device", default="cuda:0")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=6,
        help="CPU workers for deterministic bootstrap reductions (1-12; default: 6)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phases = (
        ("prepare", "smoke", "generate", "score", "bootstrap", "report")
        if args.phase == "all"
        else (args.phase,)
    )
    results: list[dict[str, Any]] = []
    try:
        for phase in phases:
            results.append(
                {
                    "phase": phase,
                    **run_phase(
                        phase,
                        output_root=args.output_root,
                        resume=args.resume,
                        semantic_device=args.semantic_device,
                        semantic_batch_size=args.semantic_batch_size,
                        bootstrap_workers=args.bootstrap_workers,
                    ),
                }
            )
    except EvaluationOrchestratorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "phases": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
