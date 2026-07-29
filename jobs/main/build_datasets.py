from __future__ import annotations

import argparse
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import (
    MAIN_SPLIT_SEED,
    attach_split,
    build_meeting_random_split,
    build_row_manifest,
    deduplicate_decision_rows,
    extract_meeting_date,
    extract_provided_data,
    extract_reference_excerpt,
    extract_section_name,
    extract_topic,
    group_by_split,
    load_jsonl,
    sha1_text,
    sort_records_for_reproducibility,
    split_analysis_train_for_grpo,
    write_json,
    write_jsonl,
)


POST_2009_CUTOFF = "2009-01-01"
ANALYSIS_MINUTES_EXPECTED_COUNTS = {"train": 102, "eval": 13, "test": 13}
DECISION_EXPECTED_COUNTS = {"train": 189, "eval": 24, "test": 24}


def _write_dataset_dir(dataset_dir: Path, grouped_rows: dict[str, list[dict]]) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    summary = {"dataset_dir": str(dataset_dir), "splits": {}}
    for split, rows in grouped_rows.items():
        ordered_rows = sort_records_for_reproducibility(rows)
        write_jsonl(dataset_dir / f"{split}.jsonl", ordered_rows)
        summary["splits"][split] = {
            "rows": len(ordered_rows),
            "meetings": len({row["meeting_date"] for row in ordered_rows}),
        }
    return summary


def _write_row_manifests(manifest_root: Path, dataset_name: str, grouped_rows: dict[str, list[dict]], extra_keys: list[str] | None = None) -> None:
    for split, rows in grouped_rows.items():
        manifest_rows = build_row_manifest(sort_records_for_reproducibility(rows), extra_keys=extra_keys)
        write_jsonl(manifest_root / f"{dataset_name}_{split}_manifest.jsonl", manifest_rows)


def _load_analysis_rows(raw_path: Path) -> list[dict]:
    rows = []
    for index, raw_row in enumerate(load_jsonl(raw_path)):
        prompt = raw_row.get("prompt", "")
        meeting_date = extract_meeting_date(prompt, raw_row.get("meeting_date"))
        if not meeting_date or meeting_date < POST_2009_CUTOFF:
            continue

        rows.append(
            {
                "index": index,
                "prompt": prompt,
                "response": raw_row.get("response", ""),
                "meeting_date": meeting_date,
                "section_name": extract_section_name(prompt, raw_row.get("section_name")),
                "topic": extract_topic(prompt, raw_row.get("topic")),
                "provided_data": extract_provided_data(prompt),
                "reference_excerpt": extract_reference_excerpt(prompt),
                "source_dataset": "fomc_qa_raw",
                "prompt_hash": sha1_text(prompt),
            }
        )
    return rows


def _load_minutes_rows(raw_path: Path) -> list[dict]:
    rows = []
    for raw_row in load_jsonl(raw_path):
        meeting_date = str(raw_row.get("meeting_date", ""))[:10]
        if not meeting_date or meeting_date < POST_2009_CUTOFF:
            continue

        prompt = raw_row.get("prompt", "")
        rows.append(
            {
                "index": int(raw_row.get("index", len(rows))),
                "prompt": prompt,
                "response": raw_row.get("response", ""),
                "meeting_date": meeting_date,
                "section_name": str(raw_row.get("section_name", "")).strip(),
                "topic": str(raw_row.get("topic", "")).strip(),
                "rate_change": raw_row.get("rate_change"),
                "current_rate": raw_row.get("current_rate"),
                "source_dataset": "synthetic_text_raw",
                "prompt_hash": sha1_text(prompt),
            }
        )
    return rows


def _load_decision_rows(pattern: str) -> tuple[list[dict], dict]:
    loaded_rows = []
    for source_file in sorted(glob.glob(pattern)):
        for raw_row in load_jsonl(Path(source_file)):
            prompt = raw_row.get("prompt", "")
            loaded_rows.append(
                {
                    "index": int(raw_row.get("index", len(loaded_rows))),
                    "prompt": prompt,
                    "response": raw_row.get("response", ""),
                    "meeting_date": extract_meeting_date(prompt, raw_row.get("meeting_date")),
                    "rate_change": raw_row.get("rate_change"),
                    "current_rate": raw_row.get("current_rate"),
                    "source_file": source_file,
                    "source_dataset": "decision_grpo_raw",
                    "prompt_hash": sha1_text(prompt),
                }
            )
    return deduplicate_decision_rows(loaded_rows)


def _assert_expected_split_counts(split_map: dict[str, list[str]], expected_counts: dict[str, int], label: str) -> None:
    actual_counts = {split: len(meetings) for split, meetings in split_map.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"{label} split counts mismatch: expected {expected_counts}, got {actual_counts}")


def _meeting_split_payload(split_map: dict[str, list[str]], source_label: str) -> dict:
    return {
        "split_strategy": "meeting_random_80_10_10",
        "split_seed": MAIN_SPLIT_SEED,
        "source_population": source_label,
        "source_population_meetings": sorted(set(sum(split_map.values(), []))),
        "meeting_count_by_split": {split: len(meetings) for split, meetings in split_map.items()},
        "meeting_dates_by_split": split_map,
    }


def build_all(
    *,
    analysis_raw: Path,
    minutes_raw: Path,
    decision_pattern: str,
    dataset_root: Path,
    manifest_root: Path,
) -> dict:
    analysis_rows = _load_analysis_rows(analysis_raw)
    minutes_rows = _load_minutes_rows(minutes_raw)
    decision_rows, decision_audit = _load_decision_rows(decision_pattern)

    analysis_meetings = sorted({row["meeting_date"] for row in analysis_rows})
    minutes_meetings = sorted({row["meeting_date"] for row in minutes_rows})

    if analysis_meetings != minutes_meetings:
        raise ValueError("Analysis and Minutes source populations do not cover the same post-2009 meetings.")
    if len(analysis_meetings) != 128:
        raise ValueError(f"Expected 128 post-2009 meetings for analysis/minutes, found {len(analysis_meetings)}.")
    if len({row['meeting_date'] for row in decision_rows}) != 237:
        raise ValueError("Expected 237 unique meetings for deduplicated decision prompts.")

    analysis_minutes_split = build_meeting_random_split(analysis_meetings, seed=MAIN_SPLIT_SEED)
    decision_split = build_meeting_random_split(
        sorted({row["meeting_date"] for row in decision_rows}),
        seed=MAIN_SPLIT_SEED,
    )

    _assert_expected_split_counts(analysis_minutes_split, ANALYSIS_MINUTES_EXPECTED_COUNTS, "analysis/minutes")
    _assert_expected_split_counts(decision_split, DECISION_EXPECTED_COUNTS, "decision")

    analysis_rows = attach_split(analysis_rows, analysis_minutes_split)
    minutes_rows = attach_split(minutes_rows, analysis_minutes_split)
    decision_rows = attach_split(decision_rows, decision_split)

    analysis_grouped = group_by_split(analysis_rows)
    minutes_grouped = group_by_split(minutes_rows)
    decision_grouped = group_by_split(decision_rows)

    analysis_grpo_train, analysis_grpo_selection = split_analysis_train_for_grpo(analysis_grouped["train"])
    analysis_grpo_grouped = {
        "train": analysis_grpo_train,
        "eval": analysis_grouped["eval"],
        "test": analysis_grouped["test"],
    }

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    analysis_sft_summary = _write_dataset_dir(dataset_root / "analysis_sft", analysis_grouped)
    analysis_grpo_summary = _write_dataset_dir(dataset_root / "analysis_grpo", analysis_grpo_grouped)
    minutes_summary = _write_dataset_dir(dataset_root / "minutes_alignment", minutes_grouped)
    decision_summary = _write_dataset_dir(dataset_root / "decision_grpo", decision_grouped)

    _write_row_manifests(manifest_root, "analysis_sft", analysis_grouped, extra_keys=["section_name", "topic"])
    _write_row_manifests(manifest_root, "analysis_grpo", analysis_grpo_grouped, extra_keys=["section_name", "topic", "analysis_grpo_role"])
    _write_row_manifests(manifest_root, "minutes_alignment", minutes_grouped, extra_keys=["section_name"])
    _write_row_manifests(manifest_root, "decision_grpo", decision_grouped, extra_keys=["rate_change", "current_rate"])

    write_json(
        manifest_root / "analysis_minutes_split.json",
        _meeting_split_payload(analysis_minutes_split, "post-2009 meetings from fomc_qa.jsonl and synthetic_text_20250520.jsonl"),
    )
    write_json(
        manifest_root / "decision_split.json",
        _meeting_split_payload(decision_split, "deduplicated union of dataset/processed/main/input_sources/decision_grpo/*.jsonl"),
    )
    write_json(manifest_root / "analysis_grpo_selection.json", analysis_grpo_selection)
    write_json(manifest_root / "decision_dedup_audit.json", decision_audit)

    summary = {
        "split_seed": MAIN_SPLIT_SEED,
        "analysis_sft": analysis_sft_summary,
        "analysis_grpo": analysis_grpo_summary,
        "minutes_alignment": minutes_summary,
        "decision_grpo": decision_summary,
        "source_populations": {
            "analysis_minutes_unique_meetings": len(analysis_meetings),
            "decision_unique_meetings": len({row["meeting_date"] for row in decision_rows}),
        },
    }
    write_json(manifest_root / "dataset_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the canonical main datasets.")
    parser.add_argument(
        "--analysis-raw",
        type=Path,
        default=ROOT / "dataset" / "processed" / "main" / "input_sources" / "fomc_qa.jsonl",
    )
    parser.add_argument(
        "--minutes-raw",
        type=Path,
        default=ROOT / "dataset" / "processed" / "main" / "input_sources" / "synthetic_text" / "source_20250520.jsonl",
    )
    parser.add_argument(
        "--decision-pattern",
        default=str(ROOT / "dataset" / "processed" / "main" / "input_sources" / "decision_grpo" / "*.jsonl"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dataset" / "processed" / "main" / "datasets",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=ROOT / "dataset" / "processed" / "main" / "manifests",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_all(
        analysis_raw=args.analysis_raw,
        minutes_raw=args.minutes_raw,
        decision_pattern=args.decision_pattern,
        dataset_root=args.dataset_root,
        manifest_root=args.manifest_root,
    )
    print("✅ Rebuilt main datasets")
    for dataset_name, dataset_summary in summary.items():
        if isinstance(dataset_summary, dict) and "splits" in dataset_summary:
            split_summary = ", ".join(
                f"{split}={details['rows']} rows/{details['meetings']} meetings"
                for split, details in dataset_summary["splits"].items()
            )
            print(f"- {dataset_name}: {split_summary}")


if __name__ == "__main__":
    main()
