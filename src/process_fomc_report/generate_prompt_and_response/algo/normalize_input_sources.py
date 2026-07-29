from __future__ import annotations

import argparse
import glob
from pathlib import Path

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl, write_json, write_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import (
    format_response_text,
    parse_response_text,
)


def _iter_source_paths(config: dict) -> tuple[list[Path], list[str]]:
    decision_cfg = config["decision"]
    pipeline_cfg = config.get("pipeline", {})
    input_root = resolve_path(pipeline_cfg.get("input_root", "dataset/processed/input_sources"))

    paths: list[Path] = [input_root / "fomc_qa.jsonl"]
    paths.extend(resolve_path(path) for path in decision_cfg["sft_source_files"].values())

    glob_matches = sorted(glob.glob(str(resolve_path(decision_cfg["grpo_source_pattern"]))))
    paths.extend(Path(path) for path in glob_matches)

    unique_paths: list[Path] = []
    missing_paths: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            missing_paths.append(str(resolved))
            seen.add(resolved)
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    if not glob_matches:
        missing_paths.append(
            f"glob:{resolve_path(decision_cfg['grpo_source_pattern'])}"
        )

    return unique_paths, missing_paths


def get_normalize_input_sources_audit_path(config: dict) -> Path:
    pipeline_cfg = config.get("pipeline", {})
    audit_root = resolve_path(pipeline_cfg.get("audit_root", "dataset/processed/pipeline/audit"))
    return audit_root / "input_sources" / "normalize_input_sources.json"


def _count_rows(path: Path) -> int:
    return len(load_jsonl(path))


def build_normalize_input_sources_marker(config_path: str | None = None, *, config: dict | None = None) -> dict:
    if config is None:
        config = load_pipeline_config(config_path)

    response_template = get_response_template(config)
    source_paths, missing_paths = _iter_source_paths(config)
    return {
        "response_template": response_template,
        "files": {
            str(path): {
                "rows": _count_rows(path),
            }
            for path in source_paths
        },
        "missing_files": missing_paths,
    }


def normalize_input_sources(config_path: str | None = None) -> dict:
    config = load_pipeline_config(config_path)
    response_template = get_response_template(config)
    summary = {
        "response_template": response_template,
        "files": {},
        "missing_files": [],
    }

    source_paths, missing_paths = _iter_source_paths(config)
    summary["missing_files"] = missing_paths

    for path in source_paths:
        rows = load_jsonl(path)
        changed_rows = 0
        structured_rows = 0
        normalized_rows = []

        for row in rows:
            normalized = dict(row)
            original_response = str(row.get("response", ""))
            parsed = parse_response_text(original_response)
            if parsed.format_name != "plain":
                structured_rows += 1
            normalized_response = format_response_text(
                original_response,
                response_template=response_template,
            )
            if normalized_response != original_response:
                normalized["response"] = normalized_response
                changed_rows += 1
            normalized_rows.append(normalized)

        if changed_rows:
            write_jsonl(path, normalized_rows)

        summary["files"][str(path)] = {
            "rows": len(rows),
            "structured_rows": structured_rows,
            "changed_rows": changed_rows,
        }

    marker = build_normalize_input_sources_marker(config=config)
    audit_path = get_normalize_input_sources_audit_path(config)
    write_json(audit_path, marker)
    summary["marker"] = marker
    summary["audit_path"] = str(audit_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize reusable input sources to the configured response template.")
    parser.add_argument("--config", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = normalize_input_sources(args.config)
    print(
        f"Normalized {len(summary['files'])} input source files "
        f"to response_template={summary['response_template']}"
    )


if __name__ == "__main__":
    main()
