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

    fetch_loo_sources = subparsers.add_parser(
        "fetch-loo-sources",
        parents=[dry_run_parent],
        help="Fetch keyless ALFRED snapshots frozen at meeting D-1.",
    )
    fetch_loo_sources.add_argument(
        "--registry",
        default="configs/main/loo_indicator_sources.json",
    )
    fetch_loo_sources.add_argument(
        "--population",
        action="append",
        required=True,
    )
    fetch_loo_sources.add_argument("--output-dir", required=True)
    fetch_loo_sources.add_argument("--resume", action="store_true")
    fetch_loo_sources.add_argument(
        "--max-workers",
        type=int,
        choices=(1, 2),
        default=1,
    )
    fetch_loo_sources.add_argument(
        "--requests-per-second",
        type=float,
        default=1.0,
    )
    fetch_loo_sources.add_argument("--timeout-seconds", type=float, default=60.0)
    fetch_loo_sources.add_argument("--max-retries", type=int, default=4)
    fetch_loo_sources.add_argument(
        "--expected-vintage-count",
        type=int,
        choices=(13, 26),
    )

    build_loo_ledger = subparsers.add_parser(
        "build-loo-ledger",
        parents=[dry_run_parent],
        help="Build one immutable 13×26 D-1 indicator ledger.",
    )
    build_loo_ledger.add_argument(
        "--registry",
        default="configs/main/loo_indicator_sources.json",
    )
    build_loo_ledger.add_argument("--snapshot-manifest", required=True)
    build_loo_ledger.add_argument("--population", required=True)
    build_loo_ledger.add_argument(
        "--roster",
        default="configs/main/leave_one_out_roster.json",
    )
    build_loo_ledger.add_argument("--output-dir", required=True)

    validate_loo_ledger = subparsers.add_parser(
        "validate-loo-ledger",
        parents=[dry_run_parent],
        help="Replay raw evidence and validate a sealed D-1 ledger.",
    )
    validate_loo_ledger.add_argument("--ledger-manifest", required=True)
    validate_loo_ledger.add_argument(
        "--registry",
        default="configs/main/loo_indicator_sources.json",
    )
    validate_loo_ledger.add_argument("--snapshot-manifest", required=True)
    validate_loo_ledger.add_argument("--population", required=True)
    validate_loo_ledger.add_argument(
        "--roster",
        default="configs/main/leave_one_out_roster.json",
    )
    validate_loo_ledger.add_argument("--expected-manifest-payload-sha256")

    finalize_loo_workflow = subparsers.add_parser(
        "finalize-loo-workflow",
        parents=[dry_run_parent],
        help="Seal the complete source→ledger→generation provenance graph.",
    )
    finalize_loo_workflow.add_argument("--run-id", required=True)
    finalize_loo_workflow.add_argument(
        "--mode",
        choices=("all", "pilot", "formal"),
        required=True,
    )
    finalize_loo_workflow.add_argument("--workflow-root", required=True)
    finalize_loo_workflow.add_argument(
        "--artifact",
        action="append",
        required=True,
    )
    finalize_loo_workflow.add_argument("--output", required=True)

    cleanup_parser = subparsers.add_parser(
        "cleanup-generated", parents=[dry_run_parent]
    )
    cleanup_parser.add_argument("--execute", action="store_true")

    train_parser = subparsers.add_parser("train", parents=[dry_run_parent])
    train_parser.add_argument("stage", choices=[*TRAINING_STAGES, "all"])

    merge_parser = subparsers.add_parser("merge", parents=[dry_run_parent])
    merge_parser.add_argument("stage", choices=[*TRAINING_STAGES, "all"])

    generate_minutes = subparsers.add_parser(
        "generate-minutes", parents=[dry_run_parent]
    )
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

    generate_loo_analysis = subparsers.add_parser(
        "generate-loo-analysis",
        parents=[dry_run_parent],
        help="Generate release-safe indicator analyses with frozen model artifacts.",
    )
    generate_loo_analysis.add_argument("--input", required=True)
    generate_loo_analysis.add_argument(
        "--roster",
        default="configs/main/leave_one_out_roster.json",
    )
    generate_loo_analysis.add_argument("--population", required=True)
    generate_loo_analysis.add_argument("--ledger-manifest", required=True)
    generate_loo_analysis.add_argument("--snapshot-manifest", required=True)
    generate_loo_analysis.add_argument("--source-registry", required=True)
    generate_loo_analysis.add_argument("--model", required=True)
    generate_loo_analysis.add_argument("--tokenizer", required=True)
    generate_loo_analysis.add_argument("--output-dir", required=True)
    generate_loo_analysis.add_argument("--seed", type=int, default=20260728)
    generate_loo_analysis.add_argument("--batch-size", type=int, default=20)
    generate_loo_analysis.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
    )
    generate_loo_analysis.add_argument(
        "--requested-max-output-tokens",
        type=int,
        default=4096,
    )
    generate_loo_analysis.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
    )
    generate_loo_analysis.add_argument(
        "--max-token-limit-errors",
        type=int,
        default=0,
    )

    project_loo_analysis = subparsers.add_parser(
        "project-loo-analysis",
        parents=[dry_run_parent],
        help="Extract deterministic final answers for canonical Minutes prompts.",
    )
    project_loo_analysis.add_argument("--input", required=True)
    project_loo_analysis.add_argument("--analysis-manifest", required=True)
    project_loo_analysis.add_argument("--tokenizer", required=True)
    project_loo_analysis.add_argument("--output-dir", required=True)
    project_loo_analysis.add_argument("--source-field", default="generated")
    project_loo_analysis.add_argument(
        "--output-field",
        default="minutes_analysis",
    )

    build_loo_prompts = subparsers.add_parser(
        "build-loo-prompts",
        parents=[dry_run_parent],
        help="Build frozen full, deletion, and token-matched neutral prompts.",
    )
    build_loo_prompts.add_argument("--analysis-blocks", required=True)
    build_loo_prompts.add_argument(
        "--section-roster",
        default="configs/main/loo_sections.json",
    )
    build_loo_prompts.add_argument(
        "--indicator-roster",
        default="configs/main/leave_one_out_roster.json",
    )
    build_loo_prompts.add_argument("--population", required=True)
    build_loo_prompts.add_argument("--population-id")
    build_loo_prompts.add_argument("--tokenizer", required=True)
    build_loo_prompts.add_argument("--output-dir", required=True)
    build_loo_prompts.add_argument(
        "--analysis-field",
        default="minutes_analysis",
    )
    build_loo_prompts.add_argument("--prompt-template")
    build_loo_prompts.add_argument(
        "--minutes-max-new-tokens",
        type=int,
        default=8192,
    )
    build_loo_prompts.add_argument(
        "--minutes-max-model-len",
        type=int,
        default=16384,
    )

    build_loo_spec = subparsers.add_parser(
        "build-loo-spec",
        parents=[dry_run_parent],
        help="Fingerprint frozen generation artifacts and write an immutable spec.",
    )
    build_loo_spec.add_argument("--run-id", required=True)
    build_loo_spec.add_argument("--phase", required=True)
    build_loo_spec.add_argument("--population-id", required=True)
    build_loo_spec.add_argument("--output", required=True)
    build_loo_spec.add_argument("--model", action="append", required=True)
    build_loo_spec.add_argument("--tokenizer", action="append", required=True)
    build_loo_spec.add_argument("--source", action="append", required=True)
    build_loo_spec.add_argument("--generation-config", required=True)
    build_loo_spec.add_argument(
        "--replicate-seed",
        action="append",
        type=int,
        required=True,
    )

    finalize_loo = subparsers.add_parser(
        "finalize-loo-generation",
        parents=[dry_run_parent],
        help="Validate all generation cells and seal the release manifest.",
    )
    finalize_loo.add_argument("--run-root", required=True)
    finalize_loo.add_argument("--generation-spec", required=True)
    finalize_loo.add_argument("--generation-spec-sha256", required=True)
    finalize_loo.add_argument("--analysis-manifest", required=True)
    finalize_loo.add_argument("--projection-manifest", required=True)
    finalize_loo.add_argument("--prompt-manifest", required=True)
    finalize_loo.add_argument("--output", required=True)

    filter_mask = subparsers.add_parser("filter-mask-prompts", parents=[dry_run_parent])
    filter_mask.add_argument(
        "--input-folder",
        default="dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator/after_2009",
    )
    filter_mask.add_argument(
        "--output-folder",
        default="dataset/processed/main/evaluation_inputs/mask_prompts_test",
    )
    filter_mask.add_argument(
        "--split", choices=["train", "eval", "test"], default="test"
    )

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
    generate_mask.add_argument(
        "--roster-file",
        default="configs/main/leave_one_out_roster.json",
    )
    generate_mask.add_argument("--simulation-step", type=int, default=1)
    generate_mask.add_argument("--batch-size", type=int, default=20)
    generate_mask.add_argument("--seed", type=int, default=20260728)
    generate_mask.add_argument("--temperature", type=float, default=0.0)
    generate_mask.add_argument("--top-p", type=float, default=1.0)
    generate_mask.add_argument("--max-new-tokens", type=int, default=8192)
    generate_mask.add_argument("--max-model-len", type=int, default=16384)
    generate_mask.add_argument(
        "--masking-strategy",
        choices=[
            "indicator_block_deletion",
            "indicator_block_neutral_replacement",
        ],
        default="indicator_block_deletion",
    )
    generate_mask.add_argument(
        "--seed-policy",
        choices=["batch-seed-v1", "sample-id-sha256-v1"],
        default="sample-id-sha256-v1",
    )
    generate_mask.add_argument("--model-sha256")
    generate_mask.add_argument("--tokenizer")
    generate_mask.add_argument("--tokenizer-sha256")
    generate_mask.add_argument("--require-normal-finish", action="store_true")
    generate_mask.add_argument("--intervention-manifest")
    generate_mask.add_argument("--prompt-manifest")
    generate_mask.add_argument("--generation-spec")
    generate_mask.add_argument("--generation-spec-sha256")

    similarity = subparsers.add_parser("eval-text-similarity", parents=[dry_run_parent])
    similarity.add_argument("--baseline-file", required=True)
    similarity.add_argument("--aligned-file", required=True)

    masking = subparsers.add_parser("eval-masking", parents=[dry_run_parent])
    masking.add_argument("mode", choices=["synthetic", "actual"])
    masking.add_argument("--input-folder", required=True)
    masking.add_argument("--output-file", required=True)
    masking.add_argument("--embedding-model-path", required=True)
    masking.add_argument("--embedding-model-sha256")
    masking.add_argument("--reference-file")
    masking.add_argument(
        "--reference-key",
        action="append",
        default=[],
        help="Repeat to define a composite actual-Minutes join key.",
    )
    masking.add_argument("--reference-text-field", default="response")
    masking.add_argument(
        "--reference-duplicate-policy",
        choices=["error", "concatenate"],
        default="error",
    )
    masking.add_argument("--row-output-file")
    masking.add_argument("--exclusions-file")
    masking.add_argument("--audit-file")
    masking.add_argument("--baseline-indicator", default="None")
    masking.add_argument(
        "--allow-missing-generation-manifest",
        action="store_true",
    )
    masking.add_argument(
        "--unmatched-policy", choices=["error", "drop"], default="error"
    )
    masking.add_argument("--embedding-batch-size", type=int, default=8)
    masking.add_argument("--score-chunk-size", type=int, default=256)
    masking.add_argument("--bootstrap-samples", type=int, default=5000)
    masking.add_argument("--bootstrap-seed", type=int, default=20260728)

    decision = subparsers.add_parser(
        "eval-decision-baselines", parents=[dry_run_parent]
    )
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

    if args.command == "fetch-loo-sources":
        command = [
            PYTHON,
            "-m",
            "jobs.main.fetch_loo_source_snapshots",
            "--registry",
            args.registry,
            "--output-dir",
            args.output_dir,
            "--max-workers",
            str(args.max_workers),
            "--requests-per-second",
            str(args.requests_per_second),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--max-retries",
            str(args.max_retries),
        ]
        for population in args.population:
            command.extend(["--population", population])
        if args.resume:
            command.append("--resume")
        if args.expected_vintage_count:
            command.extend(
                [
                    "--expected-vintage-count",
                    str(args.expected_vintage_count),
                ]
            )
        _run(command, dry_run=args.dry_run)
        return

    if args.command == "build-loo-ledger":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.main.build_loo_indicator_ledger",
                "--registry",
                args.registry,
                "--snapshot-manifest",
                args.snapshot_manifest,
                "--population",
                args.population,
                "--roster",
                args.roster,
                "--output-dir",
                args.output_dir,
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "validate-loo-ledger":
        command = [
            PYTHON,
            "-m",
            "jobs.main.validate_loo_indicator_ledger",
            "--ledger-manifest",
            args.ledger_manifest,
            "--registry",
            args.registry,
            "--snapshot-manifest",
            args.snapshot_manifest,
            "--population",
            args.population,
            "--roster",
            args.roster,
        ]
        if args.expected_manifest_payload_sha256:
            command.extend(
                [
                    "--expected-manifest-payload-sha256",
                    args.expected_manifest_payload_sha256,
                ]
            )
        _run(command, dry_run=args.dry_run)
        return

    if args.command == "finalize-loo-workflow":
        command = [
            PYTHON,
            "-m",
            "jobs.main.finalize_loo_workflow",
            "--run-id",
            args.run_id,
            "--mode",
            args.mode,
            "--workflow-root",
            args.workflow_root,
            "--output",
            args.output,
        ]
        for artifact in args.artifact:
            command.extend(["--artifact", artifact])
        _run(command, dry_run=args.dry_run)
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

    if args.command == "generate-loo-analysis":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.generation.canonical_indicator_analysis",
                "--input",
                args.input,
                "--roster",
                args.roster,
                "--population",
                args.population,
                "--ledger-manifest",
                args.ledger_manifest,
                "--snapshot-manifest",
                args.snapshot_manifest,
                "--source-registry",
                args.source_registry,
                "--model",
                args.model,
                "--tokenizer",
                args.tokenizer,
                "--output-dir",
                args.output_dir,
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--requested-max-output-tokens",
                str(args.requested_max_output_tokens),
                "--max-model-len",
                str(args.max_model_len),
                "--max-token-limit-errors",
                str(args.max_token_limit_errors),
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "project-loo-analysis":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.generation.project_indicator_analysis",
                "--input",
                args.input,
                "--analysis-manifest",
                args.analysis_manifest,
                "--tokenizer",
                args.tokenizer,
                "--output-dir",
                args.output_dir,
                "--source-field",
                args.source_field,
                "--output-field",
                args.output_field,
            ],
            dry_run=args.dry_run,
        )
        return

    if args.command == "build-loo-prompts":
        command = [
            PYTHON,
            "-m",
            "jobs.generation.loo_prompt_builder",
            "--analysis-blocks",
            args.analysis_blocks,
            "--section-roster",
            args.section_roster,
            "--indicator-roster",
            args.indicator_roster,
            "--population",
            args.population,
            "--tokenizer",
            args.tokenizer,
            "--output-dir",
            args.output_dir,
            "--analysis-field",
            args.analysis_field,
            "--minutes-max-new-tokens",
            str(args.minutes_max_new_tokens),
            "--minutes-max-model-len",
            str(args.minutes_max_model_len),
        ]
        if args.population_id:
            command.extend(["--population-id", args.population_id])
        if args.prompt_template:
            command.extend(["--prompt-template", args.prompt_template])
        _run(command, dry_run=args.dry_run)
        return

    if args.command == "build-loo-spec":
        command = [
            PYTHON,
            "-m",
            "jobs.generation.build_loo_generation_spec",
            "--run-id",
            args.run_id,
            "--phase",
            args.phase,
            "--population-id",
            args.population_id,
            "--output",
            args.output,
            "--generation-config",
            args.generation_config,
        ]
        for model in args.model:
            command.extend(["--model", model])
        for tokenizer in args.tokenizer:
            command.extend(["--tokenizer", tokenizer])
        for source in args.source:
            command.extend(["--source", source])
        for replicate_seed in args.replicate_seed:
            command.extend(["--replicate-seed", str(replicate_seed)])
        _run(command, dry_run=args.dry_run)
        return

    if args.command == "finalize-loo-generation":
        _run(
            [
                PYTHON,
                "-m",
                "jobs.generation.finalize_loo_generation",
                "--run-root",
                args.run_root,
                "--generation-spec",
                args.generation_spec,
                "--generation-spec-sha256",
                args.generation_spec_sha256,
                "--analysis-manifest",
                args.analysis_manifest,
                "--projection-manifest",
                args.projection_manifest,
                "--prompt-manifest",
                args.prompt_manifest,
                "--output",
                args.output,
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
        command = [
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
            "--roster-file",
            args.roster_file,
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--masking-strategy",
            args.masking_strategy,
            "--seed-policy",
            args.seed_policy,
        ]
        if args.model_sha256:
            command.extend(["--model-sha256", args.model_sha256])
        if args.tokenizer:
            command.extend(["--tokenizer", args.tokenizer])
        if args.tokenizer_sha256:
            command.extend(["--tokenizer-sha256", args.tokenizer_sha256])
        if args.require_normal_finish:
            command.append("--require-normal-finish")
        if args.intervention_manifest:
            command.extend(["--intervention-manifest", args.intervention_manifest])
        if args.prompt_manifest:
            command.extend(["--prompt-manifest", args.prompt_manifest])
        if args.generation_spec:
            command.extend(["--generation-spec", args.generation_spec])
        if args.generation_spec_sha256:
            command.extend(
                [
                    "--generation-spec-sha256",
                    args.generation_spec_sha256,
                ]
            )
        _run(command, dry_run=args.dry_run)
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
        if args.mode == "actual" and not args.reference_file:
            raise ValueError("--reference-file is required for eval-masking actual")

        target_mode = "full-output" if args.mode == "synthetic" else "actual-minutes"
        command = [
            PYTHON,
            "-m",
            "jobs.eval.eval_leave_one_out",
            "--input-folder",
            args.input_folder,
            "--output-file",
            args.output_file,
            "--target-mode",
            target_mode,
            "--embedding-model-path",
            args.embedding_model_path,
            "--reference-text-field",
            args.reference_text_field,
            "--reference-duplicate-policy",
            args.reference_duplicate_policy,
            "--baseline-indicator",
            args.baseline_indicator,
            "--unmatched-policy",
            args.unmatched_policy,
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--score-chunk-size",
            str(args.score_chunk_size),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
        ]
        if args.reference_file:
            command.extend(["--reference-file", args.reference_file])
        if args.embedding_model_sha256:
            command.extend(["--embedding-model-sha256", args.embedding_model_sha256])
        for reference_key in args.reference_key:
            command.extend(["--reference-key", reference_key])
        if args.row_output_file:
            command.extend(["--row-output-file", args.row_output_file])
        if args.exclusions_file:
            command.extend(["--exclusions-file", args.exclusions_file])
        if args.audit_file:
            command.extend(["--audit-file", args.audit_file])
        if args.allow_missing_generation_manifest:
            command.append("--allow-missing-generation-manifest")
        _run(command, dry_run=args.dry_run)
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
