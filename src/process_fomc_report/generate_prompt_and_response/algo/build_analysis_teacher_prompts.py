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


def log(message: str) -> None:
    print(f"[build_teacher_prompts] {message}", flush=True)


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


def build_teacher_prompts(config_path: str | None = None, *, scope: str = "after_2009") -> dict:
    log(f"Loading pipeline config. scope={scope}, config={config_path or 'default'}")
    config = load_pipeline_config(config_path)
    analysis_cfg = config["analysis"]
    threshold_cfg = analysis_cfg["audit_thresholds"]
    log("Loading grouped analysis samples.")
    grouped_samples = _load_grouped_samples(config, scope)
    log(
        "Loaded grouped samples: "
        + ", ".join(f"{split}={len(rows)}" for split, rows in grouped_samples.items())
    )
    template = load_template(MODULE_ROOT / "templates" / "analysis_teacher.md")
    log("Loaded analysis_teacher.md template.")
    log("Building with-reference teacher prompts.")
    output_rows, audit_summary = build_analysis_prompt_rows(
        grouped_samples=grouped_samples,
        template=template,
        indicator_repo=IndicatorRepository(),
        max_prompt_chars=int(threshold_cfg["max_prompt_chars"]),
        max_prompt_words=int(threshold_cfg["max_prompt_words"]),
        include_reference=True,
        progress_logger=log,
    )
    output_root = resolve_path(analysis_cfg["teacher_prompt_root"]) / scope
    log(f"Writing teacher prompts to {output_root}")
    for split, rows in output_rows.items():
        write_jsonl(output_root / f"{split}.jsonl", rows)
        log(f"Wrote split={split} rows={len(rows)}")
    audit_path = resolve_path(config["pipeline"]["audit_root"]) / "analysis_sft" / f"teacher_prompts_{scope}.json"
    write_prompt_audit(audit_path, audit_summary)
    log(f"Wrote prompt audit to {audit_path}")
    return {"output_root": str(output_root), "summary": audit_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build with-reference analysis teacher prompts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009", "before_2009"], default="after_2009")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_teacher_prompts(args.config, scope=args.scope)
    print(f"Saved teacher prompts to {result['output_root']}")


if __name__ == "__main__":
    main()
