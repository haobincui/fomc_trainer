from __future__ import annotations

import argparse
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl, write_json, write_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.paths import MODULE_ROOT
from process_fomc_report.generate_prompt_and_response.algo.common.prompting import (
    extract_current_rate,
    extract_meeting_date,
    extract_policy_options,
    extract_raw_analysis,
    load_template,
    prompt_hash,
    render_decision_prompt,
)
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import format_response_text


def _coarse_vote(label: str) -> str:
    label = str(label or "").strip()
    if label.startswith("Raise"):
        return "Raise"
    if label.startswith("Cut"):
        return "Cut"
    if label.startswith("No change"):
        return "No change"
    return ""


def _write_split(root: Path, split_name: str, rows: list[dict]) -> None:
    write_jsonl(
        root / f"{split_name}.jsonl",
        [
            {
                "prompt": row.get("prompt", ""),
                "response": row.get("response", ""),
                "provided_data": row.get("provided_data", ""),
                "rate_change": row.get("rate_change", ""),
            }
            for row in rows
        ],
    )
    write_jsonl(root / f"{split_name}_manifest.jsonl", rows)


def _scope_filter(meeting_date: str, scope: str) -> bool:
    if scope != "after_2009":
        return True
    return str(meeting_date or "") >= "2009-01-01"


def _normalize_decision_sft_rows(config: dict, *, scope: str) -> dict[str, list[dict]]:
    decision_cfg = config["decision"]
    response_template = get_response_template(config)
    grouped: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for split, source_path in decision_cfg["sft_source_files"].items():
        rows = []
        for index, row in enumerate(load_jsonl(resolve_path(source_path))):
            prompt = row.get("prompt", "")
            meeting_date = extract_meeting_date(prompt)
            if not _scope_filter(meeting_date, scope):
                continue
            rows.append(
                {
                    "sample_id": f"decision-sft-{split}-{index:05d}",
                    "split": split,
                    "meeting_date": meeting_date,
                    "section_name": "",
                    "topic": "",
                    "source_row_index": row.get("index", index),
                    "quality_flags": [],
                    "rate_change": row.get("rate_change", ""),
                    "current_rate": extract_current_rate(prompt, row.get("current_rate")),
                    "policy_options": extract_policy_options(prompt),
                    "analysis_text": extract_raw_analysis(prompt),
                    "response": format_response_text(
                        row.get("response", ""),
                        response_template=response_template,
                    ),
                    "provided_data": row.get("provided_data", ""),
                }
            )
        grouped[split] = rows
    return grouped


def _normalize_decision_grpo_rows(config: dict, *, scope: str) -> dict[str, list[dict]]:
    decision_cfg = config["decision"]
    response_template = get_response_template(config)
    all_grpo_rows = []
    for source_file in sorted(glob.glob(str(resolve_path(decision_cfg["grpo_source_pattern"])))):
        for index, row in enumerate(load_jsonl(resolve_path(source_file))):
            prompt = row.get("prompt", "")
            meeting_date = extract_meeting_date(prompt)
            if not _scope_filter(meeting_date, scope):
                continue
            all_grpo_rows.append(
                {
                    "sample_id": "",
                    "meeting_date": meeting_date,
                    "section_name": "",
                    "topic": "",
                    "source_row_index": row.get("index", index),
                    "quality_flags": [],
                    "rate_change": row.get("rate_change", ""),
                    "current_rate": extract_current_rate(prompt, row.get("current_rate")),
                    "policy_options": extract_policy_options(prompt),
                    "analysis_text": extract_raw_analysis(prompt),
                    "response": format_response_text(
                        row.get("response", ""),
                        response_template=response_template,
                    ),
                    "provided_data": row.get("provided_data", ""),
                    "source_file": source_file,
                }
            )

    grouped_by_meeting = defaultdict(list)
    for row in all_grpo_rows:
        grouped_by_meeting[row["meeting_date"]].append(row)

    deduplicated_rows = []
    for meeting_date, rows in sorted(grouped_by_meeting.items()):
        ordered = sorted(
            rows,
            key=lambda item: (
                0 if str(item["response"]).strip() else 1,
                int(item["source_row_index"]),
                item["source_file"],
            ),
        )
        canonical = dict(ordered[0])
        canonical["deduplicated_from"] = len(rows)
        deduplicated_rows.append(canonical)

    rng = random.Random(int(decision_cfg["split_seed"]))
    meeting_dates = sorted({row["meeting_date"] for row in deduplicated_rows if row["meeting_date"]})
    shuffled = meeting_dates[:]
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * 0.8)
    remainder = len(shuffled) - train_count
    eval_count = remainder // 2
    split_lookup = {}
    for meeting_date in shuffled[:train_count]:
        split_lookup[meeting_date] = "train"
    for meeting_date in shuffled[train_count : train_count + eval_count]:
        split_lookup[meeting_date] = "eval"
    for meeting_date in shuffled[train_count + eval_count :]:
        split_lookup[meeting_date] = "test"

    grouped: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for index, row in enumerate(deduplicated_rows):
        split = split_lookup[row["meeting_date"]]
        normalized = dict(row)
        normalized["sample_id"] = f"decision-grpo-{split}-{index:05d}"
        normalized["split"] = split
        normalized["coarse_vote"] = _coarse_vote(normalized["rate_change"])
        grouped[split].append(normalized)
    return grouped


def _build_decision_prompt_rows(rows: dict[str, list[dict]], template_name: str) -> tuple[dict[str, list[dict]], dict]:
    template = load_template(MODULE_ROOT / "templates" / template_name)
    grouped_output: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    summary: dict[str, dict] = {}

    for split, split_rows in rows.items():
        flag_counter: Counter[str] = Counter()
        built_rows = []
        for row in split_rows:
            quality_flags = list(row.get("quality_flags", []))
            if not row.get("analysis_text", "").strip():
                quality_flags.append("missing_analysis_text")
            if not row.get("policy_options", "").strip():
                quality_flags.append("missing_policy_options")
            flag_counter.update(quality_flags)
            prompt = render_decision_prompt(
                template=template,
                meeting_date=row["meeting_date"],
                current_rate=row.get("current_rate"),
                analysis_text=row.get("analysis_text", ""),
                policy_options=row.get("policy_options", ""),
            )
            built_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "meeting_date": row["meeting_date"],
                    "section_name": row.get("section_name", ""),
                    "topic": row.get("topic", ""),
                    "source_row_index": row["source_row_index"],
                    "quality_flags": quality_flags,
                    "rate_change": row.get("rate_change", ""),
                    "current_rate": row.get("current_rate"),
                    "coarse_vote": row.get("coarse_vote", ""),
                    "policy_options": row.get("policy_options", ""),
                    "prompt": prompt,
                    "prompt_hash": prompt_hash(prompt),
                    "provided_data": row.get("provided_data", ""),
                }
            )
        grouped_output[split] = built_rows
        summary[split] = {
            "rows": len(built_rows),
            "meetings": len({row["meeting_date"] for row in built_rows if row["meeting_date"]}),
            "flag_counts": dict(flag_counter),
        }

    return grouped_output, summary


def build_decision_prompts(config_path: str | None = None, *, scope: str = "after_2009") -> dict:
    if scope != "after_2009":
        raise RuntimeError("decision prompts are only supported for scope=after_2009.")

    config = load_pipeline_config(config_path)
    decision_cfg = config["decision"]
    sft_rows = _normalize_decision_sft_rows(config, scope=scope)
    grpo_rows = _normalize_decision_grpo_rows(config, scope=scope)
    sft_prompts, sft_summary = _build_decision_prompt_rows(sft_rows, "decision_sft.md")
    grpo_prompts, grpo_summary = _build_decision_prompt_rows(grpo_rows, "decision_grpo.md")

    sft_root = resolve_path(decision_cfg["prompt_roots"]["decision_sft"]) / scope
    grpo_root = resolve_path(decision_cfg["prompt_roots"]["decision_grpo"]) / scope
    for split, rows in sft_prompts.items():
        write_jsonl(sft_root / f"{split}.jsonl", rows)
    for split, rows in grpo_prompts.items():
        write_jsonl(grpo_root / f"{split}.jsonl", rows)

    summary = {"decision_sft": sft_summary, "decision_grpo": grpo_summary}
    write_json(resolve_path(config["pipeline"]["audit_root"]) / "decision" / f"prompts_{scope}.json", summary)
    return {
        "decision_sft_root": str(sft_root),
        "decision_grpo_root": str(grpo_root),
        "summary": summary,
    }


def build_decision_datasets(config_path: str | None = None, *, scope: str = "after_2009") -> dict:
    if scope != "after_2009":
        raise RuntimeError("decision dataset assembly is only supported for scope=after_2009.")

    config = load_pipeline_config(config_path)
    decision_cfg = config["decision"]
    decision_sft_root = resolve_path(decision_cfg["train_roots"]["decision_sft"])
    decision_grpo_root = resolve_path(decision_cfg["train_roots"]["decision_grpo"])

    sft_rows = _normalize_decision_sft_rows(config, scope=scope)
    grpo_rows = _normalize_decision_grpo_rows(config, scope=scope)
    sft_prompt_lookup = {
        split: {row["sample_id"]: row for row in load_jsonl(resolve_path(decision_cfg["prompt_roots"]["decision_sft"]) / scope / f"{split}.jsonl")}
        for split in ("train", "eval", "test")
    }
    grpo_prompt_lookup = {
        split: {row["sample_id"]: row for row in load_jsonl(resolve_path(decision_cfg["prompt_roots"]["decision_grpo"]) / scope / f"{split}.jsonl")}
        for split in ("train", "eval", "test")
    }

    summary: dict[str, dict] = {}
    for split, rows in sft_rows.items():
        normalized_rows = []
        for row in rows:
            prompt_row = sft_prompt_lookup[split][row["sample_id"]]
            normalized_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "meeting_date": row["meeting_date"],
                    "section_name": "",
                    "topic": "",
                    "source_row_index": row["source_row_index"],
                    "quality_flags": prompt_row.get("quality_flags", []),
                    "rate_change": row.get("rate_change", ""),
                    "current_rate": row.get("current_rate"),
                    "prompt_hash": prompt_row["prompt_hash"],
                    "prompt": prompt_row["prompt"],
                    "response": row.get("response", ""),
                    "provided_data": row.get("provided_data", ""),
                }
            )
        _write_split(decision_sft_root, split, normalized_rows)
        summary[f"decision_sft_{split}"] = {
            "rows": len(normalized_rows),
            "meetings": len({row["meeting_date"] for row in normalized_rows if row["meeting_date"]}),
        }

    for split, rows in grpo_rows.items():
        normalized_rows = []
        for row in rows:
            prompt_row = grpo_prompt_lookup[split][row["sample_id"]]
            normalized_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "meeting_date": row["meeting_date"],
                    "section_name": "",
                    "topic": "",
                    "source_row_index": row["source_row_index"],
                    "quality_flags": prompt_row.get("quality_flags", []),
                    "rate_change": row.get("rate_change", ""),
                    "current_rate": row.get("current_rate"),
                    "coarse_vote": row.get("coarse_vote", ""),
                    "prompt_hash": prompt_row["prompt_hash"],
                    "prompt": prompt_row["prompt"],
                    "response": row.get("response", ""),
                    "provided_data": row.get("provided_data", ""),
                }
            )
        _write_split(decision_grpo_root, split, normalized_rows)
        summary[f"decision_grpo_{split}"] = {
            "rows": len(normalized_rows),
            "meetings": len({row["meeting_date"] for row in normalized_rows if row["meeting_date"]}),
        }

    write_json(resolve_path(config["pipeline"]["audit_root"]) / "decision" / f"dataset_{scope}.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build decision prompts and datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009"], default="after_2009")
    parser.add_argument("--build", choices=["prompts", "dataset", "all"], default="all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.build in {"prompts", "all"}:
        build_decision_prompts(args.config, scope=args.scope)
    if args.build in {"dataset", "all"}:
        summary = build_decision_datasets(args.config, scope=args.scope)
        print(f"Saved decision datasets to {summary}")


if __name__ == "__main__":
    main()
