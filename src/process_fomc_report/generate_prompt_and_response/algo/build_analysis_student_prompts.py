from __future__ import annotations

import argparse

from process_fomc_report.generate_prompt_and_response.algo.common.analysis import (
    build_analysis_prompt_rows,
    load_analysis_samples,
    load_analysis_samples_from_labeled,
    write_prompt_audit,
)
from process_fomc_report.generate_prompt_and_response.algo.common.config import load_pipeline_config, resolve_path
from process_fomc_report.generate_prompt_and_response.algo.common.indicators import IndicatorRepository
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import write_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.paths import MODULE_ROOT
from process_fomc_report.generate_prompt_and_response.algo.common.prompting import load_template


def _load_grouped_samples(config: dict, scope: str) -> dict[str, list[dict]]:
    analysis_cfg = config["analysis"]
    labeled_paths = analysis_cfg.get("labeled_paths", {})
    if scope == "after_2009":
        split_manifests = analysis_cfg["split_manifests"]
        if "after_2009" in split_manifests:
            split_manifests = split_manifests["after_2009"]
        return load_analysis_samples(
            master_path=resolve_path(analysis_cfg["master_path"]),
            split_manifest_paths={split: resolve_path(path) for split, path in split_manifests.items()},
            labeled_path=resolve_path(labeled_paths.get("after_2009") or analysis_cfg.get("labeled_after_2009_path")),
        )
    return load_analysis_samples_from_labeled(
        labeled_path=resolve_path(labeled_paths[scope]),
        scope=scope,
    )


def build_student_prompts(
    config_path: str | None = None,
    *,
    scope: str = "after_2009",
    template_name: str = "analysis_student.md",
    output_root: str | None = None,
    audit_name: str = "analysis_sft_prompts",
) -> dict:
    config = load_pipeline_config(config_path)
    analysis_cfg = config["analysis"]
    threshold_cfg = analysis_cfg["audit_thresholds"]
    grouped_samples = _load_grouped_samples(config, scope)
    template = load_template(MODULE_ROOT / "templates" / template_name)
    output_rows, audit_summary = build_analysis_prompt_rows(
        grouped_samples=grouped_samples,
        template=template,
        indicator_repo=IndicatorRepository(),
        max_prompt_chars=int(threshold_cfg["max_prompt_chars"]),
        max_prompt_words=int(threshold_cfg["max_prompt_words"]),
        include_reference=False,
    )
    configured_root = output_root or analysis_cfg.get("analysis_sft_prompt_root") or analysis_cfg.get("student_prompt_root")
    output_root_path = resolve_path(configured_root) / scope
    for split, rows in output_rows.items():
        write_jsonl(output_root_path / f"{split}.jsonl", rows)
    audit_root = resolve_path(config["pipeline"]["audit_root"]) / ("analysis_grpo" if "grpo" in audit_name else "analysis_sft")
    write_prompt_audit(audit_root / f"{audit_name}_{scope}.json", audit_summary)
    return {"output_root": str(output_root_path), "summary": audit_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build without-reference analysis student prompts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009", "before_2009"], default="after_2009")
    parser.add_argument("--template-name", default="analysis_student.md")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--audit-name", default="analysis_sft_prompts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_student_prompts(
        args.config,
        scope=args.scope,
        template_name=args.template_name,
        output_root=args.output_root,
        audit_name=args.audit_name,
    )
    print(f"Saved student prompts to {result['output_root']}")


if __name__ == "__main__":
    main()
