from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

TRAINING_STAGES = {
    "analysis_sft": {
        "module": "jobs.train.train_sft",
        "config": ROOT / "configs" / "main" / "analysis_sft.yaml",
        "accelerate": ROOT / "configs" / "accelerate" / "zero3.yaml",
    },
    "analysis_grpo": {
        "module": "jobs.train.train_grpo",
        "config": ROOT / "configs" / "main" / "analysis_grpo.yaml",
        "accelerate": ROOT / "configs" / "accelerate" / "zero2.yaml",
    },
    "minutes_alignment_sft": {
        "module": "jobs.train.train_sft",
        "config": ROOT / "configs" / "main" / "minutes_alignment_sft.yaml",
        "accelerate": ROOT / "configs" / "accelerate" / "zero3.yaml",
    },
    "decision_sft": {
        "module": "jobs.train.train_sft",
        "config": ROOT / "configs" / "main" / "decision_sft.yaml",
        "accelerate": ROOT / "configs" / "accelerate" / "zero3.yaml",
    },
    "decision_grpo": {
        "module": "jobs.train.train_grpo",
        "config": ROOT / "configs" / "main" / "decision_grpo.yaml",
        "accelerate": ROOT / "configs" / "accelerate" / "zero2.yaml",
    },
}


def _run(command: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def _stage_names(selection: str) -> list[str]:
    if selection == "all":
        return list(TRAINING_STAGES)
    return [selection]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical main pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run_parent = argparse.ArgumentParser(add_help=False)
    dry_run_parent.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("build-datasets", parents=[dry_run_parent])
    subparsers.add_parser("audit", parents=[dry_run_parent])

    cleanup_parser = subparsers.add_parser("cleanup-generated", parents=[dry_run_parent])
    cleanup_parser.add_argument("--execute", action="store_true")

    train_parser = subparsers.add_parser("train", parents=[dry_run_parent])
    train_parser.add_argument("stage", choices=[*TRAINING_STAGES, "all"])

    merge_parser = subparsers.add_parser("merge", parents=[dry_run_parent])
    merge_parser.add_argument("stage", choices=[*TRAINING_STAGES, "all"])

    generate_minutes = subparsers.add_parser("generate-minutes", parents=[dry_run_parent])
    generate_minutes.add_argument("--model", required=True)
    generate_minutes.add_argument(
        "--input",
        default="dataset/processed/main/datasets/minutes_alignment/test.jsonl",
    )
    generate_minutes.add_argument(
        "--output-dir",
        default="output/evaluation/main/generated_minutes",
    )
    generate_minutes.add_argument("--start-index", type=int, default=0)
    generate_minutes.add_argument("--end-index", type=int, default=1)

    filter_mask = subparsers.add_parser("filter-mask-prompts", parents=[dry_run_parent])
    filter_mask.add_argument(
        "--input-folder",
        default="dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator/after_2009",
    )
    filter_mask.add_argument(
        "--output-folder",
        default="dataset/processed/main/evaluation_inputs/mask_prompts_test",
    )
    filter_mask.add_argument("--split", choices=["train", "eval", "test"], default="test")

    generate_mask = subparsers.add_parser("generate-masking", parents=[dry_run_parent])
    generate_mask.add_argument("--model", required=True)
    generate_mask.add_argument(
        "--input-folder",
        default="dataset/processed/main/evaluation_inputs/mask_prompts_test",
    )
    generate_mask.add_argument(
        "--output-dir",
        default="output/evaluation/main/leave_one_out_masking/generated",
    )
    generate_mask.add_argument("--simulation-step", type=int, default=5)

    similarity = subparsers.add_parser("eval-text-similarity", parents=[dry_run_parent])
    similarity.add_argument("--baseline-file", required=True)
    similarity.add_argument("--aligned-file", required=True)

    masking = subparsers.add_parser("eval-masking", parents=[dry_run_parent])
    masking.add_argument("mode", choices=["synthetic", "actual"])
    masking.add_argument("--input-folder", required=True)
    masking.add_argument("--output-file", required=True)

    decision = subparsers.add_parser("eval-decision-baselines", parents=[dry_run_parent])
    decision.add_argument("--prediction-file", action="append", default=[])
    decision.add_argument("--market-baseline")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build-datasets":
        _run([PYTHON, "-m", "jobs.main.build_datasets"], dry_run=args.dry_run)
        return

    if args.command == "audit":
        _run([PYTHON, "-m", "jobs.main.audit_splits"], dry_run=args.dry_run)
        return

    if args.command == "cleanup-generated":
        command = [PYTHON, "-m", "jobs.main.cleanup_generated"]
        if args.execute:
            command.append("--execute")
        _run(command, dry_run=args.dry_run)
        return

    if args.command == "train":
        for stage_name in _stage_names(args.stage):
            stage = TRAINING_STAGES[stage_name]
            _run(
                [
                    "accelerate",
                    "launch",
                    "--config_file",
                    str(stage["accelerate"]),
                    "-m",
                    str(stage["module"]),
                    "--config",
                    str(stage["config"]),
                ],
                dry_run=args.dry_run,
            )
        return

    if args.command == "merge":
        for stage_name in _stage_names(args.stage):
            stage = TRAINING_STAGES[stage_name]
            _run(
                [
                    PYTHON,
                    "-m",
                    "jobs.merge_model",
                    "--config",
                    str(stage["config"]),
                ],
                dry_run=args.dry_run,
            )
        return

    if args.command == "generate-minutes":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.generation.synthetic_generation",
                "stage2-full",
                "--model",
                args.model,
                "--input",
                args.input,
                "--output-dir",
                args.output_dir,
                "--start-index",
                str(args.start_index),
                "--end-index",
                str(args.end_index),
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "filter-mask-prompts":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.main.filter_prompts_by_split",
                "--input-folder",
                args.input_folder,
                "--output-folder",
                args.output_folder,
                "--split",
                args.split,
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "generate-masking":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.generation.mask_generation",
                "--input-folder",
                args.input_folder,
                "--model",
                args.model,
                "--simulation-step",
                str(args.simulation_step),
                "--output-dir",
                args.output_dir,
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "eval-text-similarity":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.main.eval_text_similarity",
                "--baseline-file",
                args.baseline_file,
                "--aligned-file",
                args.aligned_file,
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "eval-masking":
        command_name = "test3" if args.mode == "synthetic" else "test4"
        _run(
            [
                PYTHON,
                "-m",
                "jobs.eval.eval_mask",
                command_name,
                "--input-folder",
                args.input_folder,
                "--output-file",
                args.output_file,
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "eval-decision-baselines":
        command = [PYTHON, "-m", "jobs.main.eval_decision_baselines"]
        for prediction_file in args.prediction_file:
            command.extend(["--prediction-file", prediction_file])
        if args.market_baseline:
            command.extend(["--market-baseline", args.market_baseline])
        _run(command, dry_run=args.dry_run)
        return


if __name__ == "__main__":
    main()
