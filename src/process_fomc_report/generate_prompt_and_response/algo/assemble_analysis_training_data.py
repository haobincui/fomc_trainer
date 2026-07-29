from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl, training_view, write_json, write_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import format_response_text


def _profile_keep(row: dict, profile_cfg: dict) -> tuple[bool, list[str]]:
    flags = set(row.get("quality_flags", []))
    drop_reasons: list[str] = []
    if profile_cfg["require_nonempty_table"] and not row.get("has_nonempty_table", False):
        drop_reasons.append("header_only_table")
    if profile_cfg["drop_abnormal_section"] and "abnormal_section" in flags:
        drop_reasons.append("abnormal_section")
    if profile_cfg["drop_abnormal_topic"] and "abnormal_topic" in flags:
        drop_reasons.append("abnormal_topic")
    if profile_cfg["drop_missing_reference"] and "missing_reference_excerpt" in flags:
        drop_reasons.append("missing_reference_excerpt")
    if profile_cfg["drop_too_long_prompt"]:
        if "prompt_too_long_chars" in flags:
            drop_reasons.append("prompt_too_long_chars")
        if "prompt_too_long_words" in flags:
            drop_reasons.append("prompt_too_long_words")
    if profile_cfg["drop_missing_indicator_data"] and "missing_indicator_data" in flags:
        drop_reasons.append("missing_indicator_data")
    if row.get("teacher_status") not in {"success", "archived_master"}:
        drop_reasons.append("teacher_response_missing")
    return not drop_reasons, drop_reasons


def _write_dataset_dir(root: Path, grouped_rows: dict[str, list[dict]]) -> None:
    for split, rows in grouped_rows.items():
        write_jsonl(root / f"{split}.jsonl", [training_view(row) for row in rows])
        write_jsonl(root / f"{split}_manifest.jsonl", rows)


def _load_split_rows(root: Path) -> dict[str, list[dict]]:
    return {
        split: load_jsonl(root / f"{split}.jsonl")
        for split in ("train", "eval", "test")
    }


def _merge_prompt_and_teacher_rows(
    *,
    prompt_rows: dict[str, list[dict]],
    teacher_rows: dict[str, dict[str, dict]],
    profile_cfg: dict,
    response_template: str,
) -> tuple[dict[str, list[dict]], Counter[str]]:
    grouped_rows = {"train": [], "eval": [], "test": []}
    drop_counter: Counter[str] = Counter()

    for split, rows in prompt_rows.items():
        for row in rows:
            teacher_row = teacher_rows[split].get(row["sample_id"])
            merged = dict(row)
            merged["response"] = (
                format_response_text(
                    teacher_row.get("response", ""),
                    response_template=response_template,
                    reasoning_text=teacher_row.get("reasoning", ""),
                )
                if teacher_row
                else ""
            )
            merged["reasoning"] = teacher_row.get("reasoning", "") if teacher_row else ""
            merged["teacher_model"] = teacher_row.get("teacher_model", "") if teacher_row else ""
            merged["teacher_status"] = teacher_row.get("status", "missing") if teacher_row else "missing"
            keep, drop_reasons = _profile_keep(merged, profile_cfg)
            if not keep:
                drop_counter.update(drop_reasons)
                continue
            grouped_rows[split].append(merged)

    return grouped_rows, drop_counter


def _select_sft_rows(train_rows: list[dict], compat_mode: bool, compat_sft_train_rows: int) -> tuple[list[dict], dict]:
    ordered = sorted(train_rows, key=lambda item: (item["prompt_hash"], item["sample_id"]))
    if compat_mode:
        sft_train_size = min(compat_sft_train_rows, len(ordered))
    else:
        sft_train_size = math.floor(len(ordered) * 0.8)
    selected = ordered[:sft_train_size]
    return selected, {"sft_train_rows": len(selected)}


def _select_grpo_rows(
    train_rows: list[dict],
    compat_mode: bool,
    compat_sft_train_rows: int,
    compat_grpo_core_rows: int,
    replay_stride: int,
) -> tuple[list[dict], dict]:
    ordered = sorted(train_rows, key=lambda item: (item["prompt_hash"], item["sample_id"]))
    if compat_mode:
        sft_train_size = min(compat_sft_train_rows, len(ordered))
    else:
        sft_train_size = math.floor(len(ordered) * 0.8)
    sft_rows = ordered[:sft_train_size]
    grpo_core_rows = ordered[sft_train_size:]
    if compat_mode:
        grpo_core_rows = grpo_core_rows[:compat_grpo_core_rows]

    replay_target = min(math.floor(len(sft_rows) * 0.1), len(list(range(0, len(sft_rows), replay_stride))))
    replay_indices = list(range(0, len(sft_rows), replay_stride))[:replay_target]
    replay_rows = [dict(sft_rows[index]) for index in replay_indices]
    combined_grpo = [dict(row, train_source="grpo_core") for row in grpo_core_rows]
    combined_grpo.extend(dict(row, train_source="sft_mix") for row in replay_rows)
    audit = {
        "sft_train_rows": len(sft_rows),
        "grpo_core_rows": len(grpo_core_rows),
        "sft_mix_rows": len(replay_rows),
        "train_overlap_rows": len({row["sample_id"] for row in replay_rows}),
    }
    return combined_grpo, audit


def _summarize_splits(grouped_rows: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "rows": len(rows),
            "meetings": len({row["meeting_date"] for row in rows if row.get("meeting_date")}),
        }
        for split, rows in grouped_rows.items()
    }


def _profile_output_root(base_root: Path, *, requested_profile: str, profile_name: str) -> Path:
    if requested_profile == "both" and profile_name == "strict":
        return base_root.parent / f"{base_root.name}_strict"
    return base_root


def _compute_analysis_dataset_plan(
    config_path: str | None = None,
    *,
    profile: str = "both",
    dataset_kind: str = "both",
) -> dict:
    if dataset_kind not in {"analysis_sft", "analysis_grpo", "both"}:
        raise ValueError(f"Unsupported dataset_kind: {dataset_kind}")

    config = load_pipeline_config(config_path)
    analysis_cfg = config["analysis"]
    profiles = config["profiles"]
    response_template = get_response_template(config)
    profile_names = ("strict", "compat") if profile == "both" else (profile,)

    analysis_sft_prompt_root = resolve_path(analysis_cfg.get("analysis_sft_prompt_root") or analysis_cfg.get("student_prompt_root")) / "after_2009"
    analysis_grpo_prompt_root = resolve_path(analysis_cfg["analysis_grpo_prompt_root"]) / "after_2009"
    teacher_root = resolve_path(analysis_cfg["teacher_response_root"]) / "after_2009"
    analysis_sft_train_root = resolve_path(analysis_cfg["train_roots"]["analysis_sft"])
    analysis_grpo_train_root = resolve_path(analysis_cfg["train_roots"]["analysis_grpo"])
    audit_root = resolve_path(config["pipeline"]["audit_root"])

    teacher_rows = {
        split: {row["sample_id"]: row for row in load_jsonl(teacher_root / f"{split}.jsonl")}
        for split in ("train", "eval", "test")
    }
    analysis_sft_prompt_rows = _load_split_rows(analysis_sft_prompt_root) if dataset_kind in {"analysis_sft", "both"} else {}
    analysis_grpo_prompt_rows = _load_split_rows(analysis_grpo_prompt_root) if dataset_kind in {"analysis_grpo", "both"} else {}

    summary: dict[str, dict] = {}
    outputs: dict[str, dict[str, dict]] = {}
    analysis_sft_audit_profiles: dict[str, dict] = {}
    analysis_grpo_audit_profiles: dict[str, dict] = {}
    for profile_name in profile_names:
        compat_mode = profile_name == "compat"
        profile_cfg = profiles[profile_name]
        profile_summary: dict[str, dict] = {}
        profile_outputs: dict[str, dict] = {}

        if dataset_kind in {"analysis_sft", "both"}:
            analysis_sft_rows, analysis_sft_drop_counter = _merge_prompt_and_teacher_rows(
                prompt_rows=analysis_sft_prompt_rows,
                teacher_rows=teacher_rows,
                profile_cfg=profile_cfg,
                response_template=response_template,
            )
            analysis_sft_train_rows, analysis_sft_selection = _select_sft_rows(
                analysis_sft_rows["train"],
                compat_mode=compat_mode,
                compat_sft_train_rows=int(analysis_cfg["compat_sft_train_rows"]),
            )
            analysis_sft_grouped = {
                "train": analysis_sft_train_rows,
                "eval": analysis_sft_rows["eval"],
                "test": analysis_sft_rows["test"],
            }
            output_root = _profile_output_root(
                analysis_sft_train_root,
                requested_profile=profile,
                profile_name=profile_name,
            )
            _write_dataset_dir(output_root, analysis_sft_grouped)
            dataset_summary = _summarize_splits(analysis_sft_grouped)
            profile_summary["analysis_sft"] = dataset_summary
            profile_summary["analysis_sft_drop_reasons"] = dict(analysis_sft_drop_counter)
            profile_summary["analysis_sft_selection"] = analysis_sft_selection
            profile_outputs["analysis_sft"] = {
                "output_root": output_root,
                "grouped_rows": analysis_sft_grouped,
            }
            analysis_sft_audit_profiles[profile_name] = {
                "dataset_root": str(output_root),
                "splits": dataset_summary,
                "drop_reasons": dict(analysis_sft_drop_counter),
                "selection": analysis_sft_selection,
            }

        if dataset_kind in {"analysis_grpo", "both"}:
            analysis_grpo_rows, analysis_grpo_drop_counter = _merge_prompt_and_teacher_rows(
                prompt_rows=analysis_grpo_prompt_rows,
                teacher_rows=teacher_rows,
                profile_cfg=profile_cfg,
                response_template=response_template,
            )
            analysis_grpo_train_rows, analysis_grpo_selection = _select_grpo_rows(
                analysis_grpo_rows["train"],
                compat_mode=compat_mode,
                compat_sft_train_rows=int(analysis_cfg["compat_sft_train_rows"]),
                compat_grpo_core_rows=int(analysis_cfg["compat_grpo_core_rows"]),
                replay_stride=int(analysis_cfg["replay_stride"]),
            )
            analysis_grpo_grouped = {
                "train": analysis_grpo_train_rows,
                "eval": analysis_grpo_rows["eval"],
                "test": analysis_grpo_rows["test"],
            }
            output_root = _profile_output_root(
                analysis_grpo_train_root,
                requested_profile=profile,
                profile_name=profile_name,
            )
            _write_dataset_dir(output_root, analysis_grpo_grouped)
            dataset_summary = _summarize_splits(analysis_grpo_grouped)
            profile_summary["analysis_grpo"] = dataset_summary
            profile_summary["analysis_grpo_drop_reasons"] = dict(analysis_grpo_drop_counter)
            profile_summary["analysis_grpo_selection"] = analysis_grpo_selection
            profile_outputs["analysis_grpo"] = {
                "output_root": output_root,
                "grouped_rows": analysis_grpo_grouped,
            }
            analysis_grpo_audit_profiles[profile_name] = {
                "dataset_root": str(output_root),
                "splits": dataset_summary,
                "drop_reasons": dict(analysis_grpo_drop_counter),
                "selection": analysis_grpo_selection,
            }

        summary[profile_name] = profile_summary
        outputs[profile_name] = profile_outputs

    return {
        "summary": summary,
        "outputs": outputs,
        "analysis_sft_audit_profiles": analysis_sft_audit_profiles,
        "analysis_grpo_audit_profiles": analysis_grpo_audit_profiles,
        "audit_root": audit_root,
    }


def build_analysis_dataset_expectations(
    config_path: str | None = None,
    *,
    profile: str = "both",
    dataset_kind: str = "both",
) -> dict:
    plan = _compute_analysis_dataset_plan(
        config_path,
        profile=profile,
        dataset_kind=dataset_kind,
    )
    expectations: dict[str, dict[str, dict]] = {}
    for profile_name, datasets in plan["outputs"].items():
        expectations[profile_name] = {}
        for dataset_name, payload in datasets.items():
            grouped_rows = payload["grouped_rows"]
            expectations[profile_name][dataset_name] = {
                "output_root": str(payload["output_root"]),
                "splits": {
                    split: {
                        "rows": len(rows),
                        "manifest_rows": len(rows),
                    }
                    for split, rows in grouped_rows.items()
                },
            }
    return expectations


def assemble_analysis_datasets(
    config_path: str | None = None,
    *,
    profile: str = "both",
    dataset_kind: str = "both",
) -> dict:
    plan = _compute_analysis_dataset_plan(
        config_path,
        profile=profile,
        dataset_kind=dataset_kind,
    )

    for datasets in plan["outputs"].values():
        for payload in datasets.values():
            _write_dataset_dir(payload["output_root"], payload["grouped_rows"])

    if plan["analysis_sft_audit_profiles"]:
        write_json(plan["audit_root"] / "analysis_sft" / "dataset_profiles.json", plan["analysis_sft_audit_profiles"])
    if plan["analysis_grpo_audit_profiles"]:
        write_json(plan["audit_root"] / "analysis_grpo" / "dataset_profiles.json", plan["analysis_grpo_audit_profiles"])
    return plan["summary"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble analysis_sft and analysis_grpo training datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", choices=["strict", "compat", "both"], default="both")
    parser.add_argument("--dataset-kind", choices=["analysis_sft", "analysis_grpo", "both"], default="both")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = assemble_analysis_datasets(
        args.config,
        profile=args.profile,
        dataset_kind=args.dataset_kind,
    )
    print("Saved analysis datasets:")
    for profile_name, payload in summary.items():
        print(profile_name, sorted(payload))


if __name__ == "__main__":
    main()
