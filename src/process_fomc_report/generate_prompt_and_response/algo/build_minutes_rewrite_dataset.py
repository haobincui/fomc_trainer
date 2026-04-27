from __future__ import annotations

import argparse
from collections import Counter

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl, training_view, write_json, write_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.paths import MODULE_ROOT
from process_fomc_report.generate_prompt_and_response.algo.common.prompting import (
    load_template,
    prompt_hash,
    render_rewrite_prompt,
)
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import format_response_text, parse_response_text


def _load_split_rows(root, split: str) -> list[dict]:
    return load_jsonl(root / f"{split}.jsonl")


def build_minutes_rewrite_prompts(config_path: str | None = None, *, scope: str = "after_2009") -> dict:
    if scope != "after_2009":
        raise RuntimeError("minutes_alignment prompts are only supported for scope=after_2009.")

    config = load_pipeline_config(config_path)
    rewrite_cfg = config["rewrite"]
    template = load_template(MODULE_ROOT / "templates" / "minutes_rewrite.md")
    prompt_root = resolve_path(rewrite_cfg["prompt_root"]) / scope
    source_prompt_root = resolve_path(rewrite_cfg["source_prompt_root"]) / scope
    source_teacher_response_root = resolve_path(rewrite_cfg["source_teacher_response_root"]) / scope
    summary: dict[str, dict] = {}

    for split in ("train", "eval", "test"):
        source_prompt_rows = _load_split_rows(source_prompt_root, split)
        source_teacher_rows = {
            row["sample_id"]: row for row in _load_split_rows(source_teacher_response_root, split)
        }
        rows = []
        flag_counter: Counter[str] = Counter()
        teacher_status_counter: Counter[str] = Counter()
        for source_prompt_row in source_prompt_rows:
            teacher_row = source_teacher_rows.get(source_prompt_row["sample_id"])
            teacher_status = teacher_row.get("status", "missing") if teacher_row else "missing"
            teacher_status_counter[teacher_status] += 1

            quality_flags = list(source_prompt_row.get("quality_flags", []))
            target_section = str(source_prompt_row.get("section_name", "")).strip()
            reference_excerpt = str(source_prompt_row.get("reference_excerpt", "")).strip()
            raw_analysis = ""
            if teacher_row is None:
                quality_flags.append("missing_analysis_teacher_response")
            else:
                if teacher_row.get("prompt_hash") != source_prompt_row.get("prompt_hash"):
                    quality_flags.append("analysis_teacher_prompt_hash_mismatch")
                if teacher_status not in {"success", "archived_master"}:
                    quality_flags.append("analysis_teacher_response_unusable")
                raw_analysis = parse_response_text(teacher_row.get("response", "")).answer
                if not raw_analysis:
                    quality_flags.append("missing_raw_analysis")
            if not target_section:
                quality_flags.append("missing_target_section")
            if not reference_excerpt:
                quality_flags.append("missing_reference_excerpt")
            deduped_flags = list(dict.fromkeys(quality_flags))
            flag_counter.update(deduped_flags)
            prompt = render_rewrite_prompt(
                template=template,
                meeting_date=source_prompt_row.get("meeting_date", ""),
                target_section=target_section,
                raw_analysis=raw_analysis,
            )
            rows.append(
                {
                    "sample_id": source_prompt_row["sample_id"],
                    "split": split,
                    "meeting_date": source_prompt_row.get("meeting_date", ""),
                    "section_name": target_section,
                    "topic": source_prompt_row.get("topic", ""),
                    "source_row_index": source_prompt_row.get("source_row_index"),
                    "quality_flags": deduped_flags,
                    "raw_analysis": raw_analysis,
                    "reference_excerpt": reference_excerpt,
                    "prompt": prompt,
                    "prompt_hash": prompt_hash(prompt),
                    "provided_data": source_prompt_row.get("provided_data", ""),
                    "analysis_teacher_status": teacher_status,
                }
            )
        write_jsonl(prompt_root / f"{split}.jsonl", rows)
        summary[split] = {
            "rows": len(rows),
            "meetings": len({row["meeting_date"] for row in rows if row["meeting_date"]}),
            "flag_counts": dict(flag_counter),
            "analysis_teacher_status_counts": dict(teacher_status_counter),
        }

    write_json(resolve_path(config["pipeline"]["audit_root"]) / "minutes_alignment" / f"prompts_{scope}.json", summary)
    return {"output_root": str(prompt_root), "summary": summary}


def build_minutes_rewrite_dataset(config_path: str | None = None, *, scope: str = "after_2009") -> dict:
    if scope != "after_2009":
        raise RuntimeError("minutes_alignment dataset assembly is only supported for scope=after_2009.")

    config = load_pipeline_config(config_path)
    rewrite_cfg = config["rewrite"]
    response_template = get_response_template(config)
    prompt_root = resolve_path(rewrite_cfg["prompt_root"]) / scope
    teacher_root = resolve_path(rewrite_cfg["teacher_response_root"]) / scope
    output_root = resolve_path(rewrite_cfg["train_root"])
    normalized_target_root = resolve_path(rewrite_cfg["normalized_target_root"])
    summary: dict[str, dict] = {}

    for split in ("train", "eval", "test"):
        prompt_rows = _load_split_rows(prompt_root, split)
        teacher_rows = {row["sample_id"]: row for row in _load_split_rows(teacher_root, split)}

        normalized_rows = []
        normalized_targets = []
        drop_counter: Counter[str] = Counter()
        for prompt_row in prompt_rows:
            reference_excerpt = str(prompt_row.get("reference_excerpt", "")).strip()
            raw_analysis = str(prompt_row.get("raw_analysis", "")).strip()
            teacher_row = teacher_rows.get(prompt_row["sample_id"])
            if not reference_excerpt:
                drop_counter["missing_reference_excerpt"] += 1
                continue
            if not raw_analysis:
                drop_counter["missing_raw_analysis"] += 1
                continue
            if teacher_row is None:
                drop_counter["teacher_response_missing"] += 1
                continue
            if teacher_row.get("prompt_hash") != prompt_row.get("prompt_hash"):
                drop_counter["teacher_prompt_hash_mismatch"] += 1
                continue
            if teacher_row.get("status") not in {"success", "archived_master"}:
                drop_counter["teacher_response_missing"] += 1
                continue

            parsed_rewrite_response = parse_response_text(teacher_row.get("response", ""))
            teacher_rewrite_response = parsed_rewrite_response.answer
            teacher_rewrite_reasoning = teacher_row.get("reasoning", "") or parsed_rewrite_response.reasoning
            if not teacher_rewrite_response:
                drop_counter["teacher_rewrite_empty_response"] += 1
                continue

            normalized_response = format_response_text(
                reference_excerpt,
                response_template=response_template,
                reasoning_text=teacher_rewrite_reasoning,
            )
            normalized_rows.append(
                {
                    "sample_id": prompt_row["sample_id"],
                    "split": split,
                    "meeting_date": prompt_row["meeting_date"],
                    "section_name": prompt_row["section_name"],
                    "topic": prompt_row.get("topic", ""),
                    "source_row_index": prompt_row["source_row_index"],
                    "quality_flags": prompt_row.get("quality_flags", []),
                    "prompt_hash": prompt_row["prompt_hash"],
                    "prompt": prompt_row["prompt"],
                    "response": normalized_response,
                    "provided_data": prompt_row.get("provided_data", ""),
                    "reference_excerpt": reference_excerpt,
                    "raw_analysis": raw_analysis,
                    "teacher_rewrite_response": teacher_rewrite_response,
                    "teacher_rewrite_reasoning": teacher_rewrite_reasoning,
                    "teacher_model": teacher_row.get("teacher_model", ""),
                    "teacher_status": teacher_row.get("status", "missing"),
                }
            )
            normalized_targets.append(
                {
                    "sample_id": prompt_row["sample_id"],
                    "split": split,
                    "meeting_date": prompt_row["meeting_date"],
                    "section_name": prompt_row["section_name"],
                    "source_row_index": prompt_row["source_row_index"],
                    "response": normalized_response,
                    "provided_data": prompt_row.get("provided_data", ""),
                    "reference_excerpt": reference_excerpt,
                    "teacher_rewrite_reasoning": teacher_rewrite_reasoning,
                }
            )

        write_jsonl(output_root / f"{split}.jsonl", [training_view(row) for row in normalized_rows])
        write_jsonl(output_root / f"{split}_manifest.jsonl", normalized_rows)
        write_jsonl(normalized_target_root / f"{split}.jsonl", normalized_targets)
        summary[split] = {
            "rows": len(normalized_rows),
            "meetings": len({row["meeting_date"] for row in normalized_rows if row["meeting_date"]}),
            "drop_reasons": dict(drop_counter),
        }

    write_json(resolve_path(config["pipeline"]["audit_root"]) / "minutes_alignment" / f"dataset_{scope}.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build minutes_alignment prompts and datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009"], default="after_2009")
    parser.add_argument("--build", choices=["prompts", "dataset", "all"], default="all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.build in {"prompts", "all"}:
        build_minutes_rewrite_prompts(args.config, scope=args.scope)
    if args.build in {"dataset", "all"}:
        summary = build_minutes_rewrite_dataset(args.config, scope=args.scope)
        print(f"Saved rewrite dataset to {summary}")


if __name__ == "__main__":
    main()
