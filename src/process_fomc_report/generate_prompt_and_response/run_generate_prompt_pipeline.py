from __future__ import annotations

import argparse
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from process_fomc_report.generate_prompt_and_response.algo.assemble_analysis_training_data import assemble_analysis_datasets
from process_fomc_report.generate_prompt_and_response.algo.build_analysis_student_prompts import build_student_prompts
from process_fomc_report.generate_prompt_and_response.algo.build_analysis_teacher_prompts import build_teacher_prompts
from process_fomc_report.generate_prompt_and_response.algo.build_decision_datasets import build_decision_datasets, build_decision_prompts
from process_fomc_report.generate_prompt_and_response.algo.build_minutes_rewrite_dataset import (
    build_minutes_rewrite_dataset,
    build_minutes_rewrite_prompts,
)
from process_fomc_report.generate_prompt_and_response.algo.common.config import load_pipeline_config, resolve_path
from process_fomc_report.generate_prompt_and_response.algo.generate_analysis_teacher_responses import generate_teacher_responses
from process_fomc_report.generate_prompt_and_response.algo.generate_minutes_rewrite_teacher_responses import (
    generate_minutes_rewrite_teacher_responses,
)
from process_fomc_report.generate_prompt_and_response.algo.label_minutes_sections import label_minutes_sections
from process_fomc_report.generate_prompt_and_response.algo.merge_labeled_minutes import merge_labeled_minutes
from process_fomc_report.generate_prompt_and_response.algo.normalize_input_sources import normalize_input_sources


PUBLIC_STAGES = [
    "normalize_input_sources",
    "label_html",
    "merge_labels",
    "analysis_sft_teacher_prompts",
    "analysis_sft_teacher_responses",
    "analysis_sft_prompts",
    "analysis_sft_dataset",
    "analysis_grpo_prompts",
    "analysis_grpo_dataset",
    "minutes_alignment_prompts",
    "minutes_alignment_teacher_responses",
    "minutes_alignment_dataset",
    "decision_prompts",
    "decision_dataset",
    "all",
]

AFTER_ONLY_STAGES = {
    "normalize_input_sources",
    "analysis_sft_dataset",
    "analysis_grpo_prompts",
    "analysis_grpo_dataset",
    "minutes_alignment_prompts",
    "minutes_alignment_teacher_responses",
    "minutes_alignment_dataset",
    "decision_prompts",
    "decision_dataset",
}


def parse_binary_bool(value: str) -> bool:
    normalized = str(value).strip()
    if normalized == "0":
        return False
    if normalized == "1":
        return True
    raise argparse.ArgumentTypeError("expected 0 or 1")


def _normalize_stage(stage: str) -> str:
    return stage


def _ensure_after_2009(scope: str, stage: str) -> None:
    if scope != "after_2009":
        raise RuntimeError(f"stage={stage} is only supported for scope=after_2009.")


def _iter_label_inputs(config: dict, scope: str) -> list[Path]:
    labeling_cfg = config["labeling"]
    input_glob = str(labeling_cfg.get("input_glob", "") or "").strip()
    if input_glob:
        input_glob = input_glob.format(scope=scope)
        return sorted(Path(path) for path in glob.glob(str(resolve_path(input_glob))))

    raw_html_root = resolve_path(config["pipeline"]["raw_html_root"]) / scope
    direct_matches = sorted(raw_html_root.glob("*.xlsx"))
    if direct_matches:
        return direct_matches

    nested_root = resolve_path(config["pipeline"]["raw_html_root"])
    nested_matches = sorted(path for path in nested_root.rglob("*.xlsx") if path.parent.name == scope)
    return nested_matches


def _labeled_output_file(config: dict, scope: str) -> Path:
    analysis_cfg = config["analysis"]
    labeled_paths = analysis_cfg.get("labeled_paths", {})
    if scope in labeled_paths:
        return resolve_path(labeled_paths[scope])
    if scope == "after_2009":
        return resolve_path(analysis_cfg.get("labeled_after_2009_path"))
    return resolve_path(config["labeling"]["output_root"]) / f"merged_labeled_{scope}.xlsx"


def run_pipeline(
    *,
    config_path: str | None,
    stage: str,
    profile: str,
    scope: str,
    teacher_source: str | None,
    force_teacher_refresh: bool = False,
) -> None:
    config = load_pipeline_config(config_path)
    stage = _normalize_stage(stage)

    if stage in AFTER_ONLY_STAGES and scope != "after_2009":
        _ensure_after_2009(scope, stage)

    if stage == "label_html":
        labeling_cfg = config["labeling"]
        input_files = _iter_label_inputs(config, scope)
        if not input_files:
            raw_html_root = resolve_path(config["pipeline"]["raw_html_root"])
            raise RuntimeError(
                f"No raw minutes files found for scope={scope} under {raw_html_root}. "
                f"Checked both {raw_html_root / scope} and nested */{scope}/*.xlsx paths."
            )
        for input_file in input_files:
            label_minutes_sections(
                input_file=input_file,
                output_dir=resolve_path(labeling_cfg["output_root"]) / scope,
                indicator_file=resolve_path(labeling_cfg["indicator_file"]),
                llm_payload=labeling_cfg,
            )
        return

    if stage == "normalize_input_sources":
        normalize_input_sources(config_path)
        return

    if stage == "merge_labels":
        merge_labeled_minutes(
            input_dir=resolve_path(config["labeling"]["output_root"]) / scope,
            output_file=_labeled_output_file(config, scope),
            indicator_file=resolve_path(config["labeling"]["indicator_file"]),
            scope=scope,
        )
        return

    if stage == "analysis_sft_teacher_prompts":
        build_teacher_prompts(config_path, scope=scope)
        return

    if stage == "analysis_sft_teacher_responses":
        generate_teacher_responses(
            config_path,
            scope=scope,
            teacher_source=teacher_source,
            force_refresh=force_teacher_refresh,
        )
        return

    if stage == "analysis_sft_prompts":
        build_student_prompts(
            config_path,
            scope=scope,
            template_name="analysis_student.md",
            output_root=config["analysis"].get("analysis_sft_prompt_root"),
            audit_name="analysis_sft_prompts",
        )
        return

    if stage == "analysis_sft_dataset":
        assemble_analysis_datasets(config_path, profile=profile, dataset_kind="analysis_sft")
        return

    if stage == "analysis_grpo_prompts":
        build_student_prompts(
            config_path,
            scope=scope,
            template_name="analysis_grpo.md",
            output_root=config["analysis"]["analysis_grpo_prompt_root"],
            audit_name="analysis_grpo_prompts",
        )
        return

    if stage == "analysis_grpo_dataset":
        assemble_analysis_datasets(config_path, profile=profile, dataset_kind="analysis_grpo")
        return

    if stage == "minutes_alignment_prompts":
        build_minutes_rewrite_prompts(config_path, scope=scope)
        return

    if stage == "minutes_alignment_teacher_responses":
        generate_minutes_rewrite_teacher_responses(
            config_path,
            scope=scope,
            teacher_source=teacher_source,
            force_refresh=force_teacher_refresh,
        )
        return

    if stage == "minutes_alignment_dataset":
        build_minutes_rewrite_dataset(config_path, scope=scope)
        return

    if stage == "decision_prompts":
        build_decision_prompts(config_path, scope=scope)
        return

    if stage == "decision_dataset":
        build_decision_datasets(config_path, scope=scope)
        return

    if stage == "all":
        if scope == "before_2009":
            run_pipeline(config_path=config_path, stage="label_html", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
            run_pipeline(config_path=config_path, stage="merge_labels", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
            run_pipeline(config_path=config_path, stage="analysis_sft_teacher_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
            run_pipeline(config_path=config_path, stage="analysis_sft_teacher_responses", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
            run_pipeline(config_path=config_path, stage="analysis_sft_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
            return

        run_pipeline(config_path=config_path, stage="normalize_input_sources", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        if config["labeling"].get("run_in_all"):
            run_pipeline(config_path=config_path, stage="label_html", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="merge_labels", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="analysis_sft_teacher_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="analysis_sft_teacher_responses", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="analysis_sft_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="analysis_sft_dataset", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="analysis_grpo_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="analysis_grpo_dataset", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="minutes_alignment_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="minutes_alignment_teacher_responses", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="minutes_alignment_dataset", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="decision_prompts", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        run_pipeline(config_path=config_path, stage="decision_dataset", profile=profile, scope=scope, teacher_source=teacher_source, force_teacher_refresh=force_teacher_refresh)
        return

    raise ValueError(f"Unsupported stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main prompt and dataset pipeline."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009", "before_2009"], default="after_2009")
    parser.add_argument("--teacher-source", choices=["llm", "archived_master"], default=None)
    parser.add_argument("--force-teacher-refresh", type=parse_binary_bool, default=False)
    parser.add_argument("--profile", choices=["strict", "compat", "both"], default="both")
    parser.add_argument(
        "--stage",
        choices=PUBLIC_STAGES,
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        config_path=args.config,
        stage=args.stage,
        profile=args.profile,
        scope=args.scope,
        teacher_source=args.teacher_source,
        force_teacher_refresh=args.force_teacher_refresh,
    )
    print(
        f"Completed stage={args.stage} scope={args.scope} profile={args.profile}"
        + (f" teacher_source={args.teacher_source}" if args.teacher_source else "")
        + f" force_teacher_refresh={int(args.force_teacher_refresh)}"
    )


if __name__ == "__main__":
    main()
