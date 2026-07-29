from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

from .indicators import IndicatorRepository
from .io_utils import load_jsonl, write_json
from .prompting import extract_reference_excerpt, prompt_hash, render_analysis_prompt
from .quality import (
    canonicalize_section_name,
    canonicalize_topic,
    is_abnormal_section,
    is_abnormal_topic,
    prompt_length_metrics,
    table_metrics,
)


def load_analysis_samples(
    *,
    master_path: Path,
    split_manifest_paths: dict[str, Path],
    labeled_path: Path,
) -> dict[str, list[dict]]:
    master_rows = {row["sample_id"]: row for row in load_jsonl(master_path)}
    labeled_records = pd.read_excel(labeled_path).to_dict("records")
    grouped: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}

    for split, manifest_path in split_manifest_paths.items():
        for manifest_row in load_jsonl(manifest_path):
            sample_id = manifest_row["sample_id"]
            master_row = dict(master_rows[sample_id])
            source_row_index = int(master_row["source_row_index"])
            reference_excerpt = extract_reference_excerpt(master_row.get("prompt", ""))
            reference_source_status = "legacy_prompt" if reference_excerpt else "missing"
            reference_flags: list[str] = []
            if 0 <= source_row_index < len(labeled_records):
                candidate = labeled_records[source_row_index]
                candidate_excerpt = str(candidate.get("raw_text", "")).strip()
                candidate_date = str(candidate.get("date", ""))[:10]
                candidate_topic = canonicalize_topic(candidate.get("relabel") or candidate.get("new_label") or candidate.get("label", ""))
                topic_matches = candidate_topic == canonicalize_topic(master_row["topic"])
                date_matches = not candidate_date or candidate_date == str(master_row["meeting_date"])[:10]
                if candidate_excerpt and date_matches and topic_matches:
                    reference_excerpt = candidate_excerpt
                    reference_source_status = "labeled_index"
                elif candidate_excerpt and not reference_excerpt:
                    reference_excerpt = candidate_excerpt
                    reference_source_status = "labeled_index_fallback"
            record = {
                "sample_id": sample_id,
                "meeting_date": str(master_row["meeting_date"])[:10],
                "section_name": canonicalize_section_name(master_row["section_name"]),
                "topic": canonicalize_topic(master_row["topic"]),
                "archived_prompt": master_row["prompt"],
                "archived_response": master_row["response"],
                "provided_data": master_row.get("provided_data", ""),
                "rate_change": master_row.get("rate_change"),
                "current_rate": master_row.get("current_rate"),
                "source_row_index": source_row_index,
                "response_origin": master_row.get("response_origin", ""),
                "reference_excerpt": reference_excerpt,
                "reference_source_status": reference_source_status,
                "split": split,
                "pre_quality_flags": reference_flags,
            }
            grouped[split].append(record)

    return grouped


def load_analysis_samples_from_labeled(
    *,
    labeled_path: Path,
    scope: str,
) -> dict[str, list[dict]]:
    labeled_records = pd.read_excel(labeled_path).to_dict("records")
    grouped: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}

    for index, record in enumerate(labeled_records):
        meeting_date = str(record.get("date", "") or "")[:10]
        topic = canonicalize_topic(record.get("relabel") or record.get("new_label") or record.get("label", ""))
        section_name = canonicalize_section_name(record.get("section_name", ""))
        reference_excerpt = str(record.get("raw_text", "") or "").strip()
        grouped["train"].append(
            {
                "sample_id": f"{scope}-analysis-{index:05d}",
                "meeting_date": meeting_date,
                "section_name": section_name,
                "topic": topic,
                "archived_prompt": "",
                "archived_response": "",
                "provided_data": "",
                "rate_change": None,
                "current_rate": None,
                "source_row_index": index,
                "response_origin": "",
                "reference_excerpt": reference_excerpt,
                "reference_source_status": "labeled_raw_text" if reference_excerpt else "missing",
                "split": "train",
                "pre_quality_flags": [],
            }
        )

    return grouped


def build_analysis_prompt_rows(
    *,
    grouped_samples: dict[str, list[dict]],
    template: str,
    indicator_repo: IndicatorRepository,
    max_prompt_chars: int,
    max_prompt_words: int,
    include_reference: bool,
    progress_logger: Callable[[str], None] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    output: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    audit_summary: dict[str, dict] = {}

    for split, rows in grouped_samples.items():
        if progress_logger is not None:
            progress_logger(f"Starting split={split} rows={len(rows)}")
        flag_counter: Counter[str] = Counter()
        built_rows: list[dict] = []
        for index, row in enumerate(rows, start=1):
            bundle = indicator_repo.build_indicator_bundle(row["topic"], row["meeting_date"])
            prompt = render_analysis_prompt(
                template=template,
                meeting_date=row["meeting_date"],
                topic=row["topic"],
                section_style=row["section_name"],
                data_label=bundle.data_label or row["topic"],
                table_str=bundle.provided_data,
                reference_excerpt=row["reference_excerpt"] if include_reference else "",
            )
            metrics = prompt_length_metrics(prompt)
            table_info = table_metrics(bundle.provided_data)
            quality_flags = list(row["pre_quality_flags"])
            if not table_info["has_nonempty_table"]:
                quality_flags.append("header_only_table")
            if is_abnormal_section(row["section_name"]):
                quality_flags.append("abnormal_section")
            if is_abnormal_topic(row["topic"]):
                quality_flags.append("abnormal_topic")
            if not row["reference_excerpt"].strip():
                quality_flags.append("missing_reference_excerpt")
            if metrics["prompt_length_chars"] > max_prompt_chars:
                quality_flags.append("prompt_too_long_chars")
            if metrics["prompt_length_words"] > max_prompt_words:
                quality_flags.append("prompt_too_long_words")
            if bundle.missing_indicators:
                quality_flags.append("missing_indicator_data")

            deduped_flags = list(dict.fromkeys(quality_flags))
            flag_counter.update(deduped_flags)
            built_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "meeting_date": row["meeting_date"],
                    "section_name": row["section_name"],
                    "topic": row["topic"],
                    "source_row_index": row["source_row_index"],
                    "rate_change": row["rate_change"],
                    "current_rate": row["current_rate"],
                    "prompt": prompt,
                    "prompt_hash": prompt_hash(prompt),
                    "provided_data": bundle.provided_data,
                    "data_label": bundle.data_label,
                    "reference_excerpt": row["reference_excerpt"] if include_reference else "",
                    "reference_source_status": row["reference_source_status"],
                    "has_nonempty_table": table_info["has_nonempty_table"],
                    "table_count": table_info["table_count"],
                    "data_table_count": table_info["data_table_count"],
                    "header_only_table_count": table_info["header_only_table_count"],
                    "prompt_length_chars": metrics["prompt_length_chars"],
                    "prompt_length_words": metrics["prompt_length_words"],
                    "quality_flags": deduped_flags,
                    "missing_indicators": bundle.missing_indicators,
                    "source_files": bundle.source_files,
                    "archived_response": row["archived_response"],
                    "response_origin": row["response_origin"],
                }
            )
            if progress_logger is not None and (index == len(rows) or index % 500 == 0):
                progress_logger(f"Processed split={split} rows={index}/{len(rows)}")

        output[split] = sorted(built_rows, key=lambda item: (item["meeting_date"], item["sample_id"]))
        audit_summary[split] = {
            "rows": len(built_rows),
            "meetings": len({item["meeting_date"] for item in built_rows}),
            "flag_counts": dict(flag_counter),
        }
        if progress_logger is not None:
            progress_logger(
                f"Finished split={split} rows={len(built_rows)} "
                f"meetings={audit_summary[split]['meetings']}"
            )

    return output, audit_summary


def write_prompt_audit(path: Path, summary: dict) -> None:
    write_json(path, summary)
