from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
)
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import format_response_text


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent

ARCHIVED_LABELED_SOURCE = REPO_ROOT / "archive" / "data" / "process_fomc_report" / "generate_prompt_output" / "labeled_text" / "merged_labeled_after_2009.xlsx"
ARCHIVED_PROMPT_SOURCE = REPO_ROOT / "archive" / "data" / "process_fomc_report" / "generate_prompt_output" / "data_to_analysis_prompt_no_ref" / "prompt_after_2009_20251016_no_reference.jsonl"
ARCHIVED_RESPONSE_SOURCE = REPO_ROOT / "archive" / "data" / "dataset" / "raw_data" / "generate_prompt" / "data_to_analysis" / "merged_response_20250512.jsonl"
PROCESSED_INPUT_ROOT = REPO_ROOT / "dataset" / "processed" / "input_sources"
RAW_BACKFILL_CANDIDATES = [
    PROCESSED_INPUT_ROOT / "fomc_qa.jsonl",
    REPO_ROOT / "archive" / "data" / "process_fomc_report" / "output" / "chapter2_qa" / "master" / "chapter2_qa_master.jsonl",
    REPO_ROOT / "archive" / "data" / "dataset" / "raw_data" / "generate_prompt" / "chapter2_qa" / "master" / "chapter2_qa_master.jsonl",
]
LEGACY_SPLIT_ROOT = REPO_ROOT / "archive" / "data" / "dataset" / "training_data" / "fomc_qa"
CANONICAL_MANIFEST_ROOT = REPO_ROOT / "dataset" / "processed" / "manifests"
EXPECTED_AFTER_2009_MASTER_ROWS = 4887
EXPECTED_AFTER_2009_SPLIT_ROWS = {"train": 3890, "eval": 535, "test": 462}

EXPECTED_DROPPED_ROW = {
    "source_row_index": 1085,
    "meeting_date": "2012-06-20",
    "section_name": "Participants' Views on Current Conditions and the Economic Outlook",
    "topic": "GDP Growth",
}


def log(message: str) -> None:
    print(f"[build_qa_master] {message}", flush=True)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_prompt_metadata(prompt: str) -> dict[str, str]:
    meeting_match = re.search(r"meeting on \*\*(.*?)\*\*", prompt)
    topic_match = re.search(r"recent trends in \*\*(.*?)\*\* and related indicators", prompt)
    if not topic_match:
        topic_match = re.search(r"developments in \*\*(.*?)\*\*, focusing on the trends", prompt)
    section_match = re.search(r"tone and style of the \*\*(.*?)\*\* section", prompt)

    return {
        "meeting_date": (meeting_match.group(1).split()[0] if meeting_match else ""),
        "topic": topic_match.group(1) if topic_match else "",
        "section_name": section_match.group(1) if section_match else "",
    }


def minimal_training_view(row: dict) -> dict:
    return {
        "prompt": row["prompt"],
        "response": row["response"],
        "provided_data": row["provided_data"],
    }


def _load_existing_backfill_sources() -> list[tuple[Path, list[dict]]]:
    return [
        (path, load_jsonl(path))
        for path in RAW_BACKFILL_CANDIDATES
        if path.exists()
    ]


def build_canonical_master(*, response_template: str) -> tuple[list[dict], dict]:
    log("Loading archived labeled/prompt/response/backfill sources.")
    labeled_frame = pd.read_excel(ARCHIVED_LABELED_SOURCE)
    prompt_rows = load_jsonl(ARCHIVED_PROMPT_SOURCE)
    response_rows = load_jsonl(ARCHIVED_RESPONSE_SOURCE)
    backfill_sources = _load_existing_backfill_sources()
    if not backfill_sources:
        raise FileNotFoundError(
            "No raw backfill source found. Checked: "
            + ", ".join(str(path) for path in RAW_BACKFILL_CANDIDATES)
        )
    raw_backfill_source, raw_backfill_rows = backfill_sources[0]
    log(
        "Loaded source rows: "
        f"labeled={len(labeled_frame)}, "
        f"prompt={len(prompt_rows)}, "
        f"response={len(response_rows)}, "
        f"backfill={len(raw_backfill_rows)}"
    )

    prompt_by_index = {int(row["index"]): row for row in prompt_rows}
    response_by_index = {int(row["index"]): row for row in response_rows}
    raw_backfill_by_prompt = [
        (path, {row["prompt"]: row for row in rows if "prompt" in row})
        for path, rows in backfill_sources
    ]

    prompt_only = sorted(set(prompt_by_index) - set(response_by_index))
    response_only = sorted(set(response_by_index) - set(prompt_by_index))
    metadata_mismatch = []
    for index in sorted(set(prompt_by_index) & set(response_by_index)):
        prompt_row = prompt_by_index[index]
        response_row = response_by_index[index]
        prompt_meta = (prompt_row["meeting_date"], prompt_row["section_name"], prompt_row["topic"])
        response_meta = (response_row["meeting_date"], response_row["section_name"], response_row["topic"])
        if prompt_meta != response_meta:
            metadata_mismatch.append(
                {
                    "source_row_index": index,
                    "prompt_meta": prompt_meta,
                    "response_meta": response_meta,
                }
            )

    canonical_rows: list[dict] = []
    backfilled_rows: list[dict] = []
    dropped_rows: list[dict] = []

    log("Reconciling prompt/response rows into canonical master.")
    for index in sorted(response_by_index):
        prompt_row = prompt_by_index[index]
        response_row = response_by_index[index]
        prompt = response_row["prompt"]
        response_text = str(response_row.get("response", "")).strip()
        response_origin = "archived_response"
        backfill_source = ""

        if not response_text:
            for backfill_path, prompt_map in raw_backfill_by_prompt:
                raw_backfill_row = prompt_map.get(prompt)
                if raw_backfill_row and str(raw_backfill_row.get("response", "")).strip():
                    response_text = str(raw_backfill_row["response"]).strip()
                    response_origin = "backfilled_from_dataset_raw"
                    backfill_source = str(backfill_path)
                    break
            if not response_text:
                dropped_rows.append(
                    {
                        "source_row_index": int(prompt_row["index"]),
                        "meeting_date": prompt_row["meeting_date"],
                        "section_name": prompt_row["section_name"],
                        "topic": prompt_row["topic"],
                        "prompt_hash": prompt_hash(prompt),
                        "reason": "empty_response_without_backfill",
                    }
                )
                continue

        response_text = format_response_text(
            response_text,
            response_template=response_template,
        )

        row = {
            "sample_id": "",
            "meeting_date": prompt_row["meeting_date"],
            "section_name": prompt_row["section_name"],
            "topic": prompt_row["topic"],
            "prompt": prompt,
            "response": response_text,
            "provided_data": prompt_row["provided_data"],
            "rate_change": prompt_row.get("rate_change", ""),
            "current_rate": prompt_row.get("current_rate", ""),
            "source_row_index": int(prompt_row["index"]),
            "response_origin": response_origin,
            "backfill_source": backfill_source if response_origin != "archived_response" else "",
        }
        canonical_rows.append(row)
        if response_origin != "archived_response":
            backfilled_rows.append(
                {
                    "source_row_index": row["source_row_index"],
                    "meeting_date": row["meeting_date"],
                    "section_name": row["section_name"],
                    "topic": row["topic"],
                    "prompt_hash": prompt_hash(row["prompt"]),
                }
            )

    for idx, row in enumerate(canonical_rows):
        row["sample_id"] = f"qa-after-2009-{idx:05d}"

    log(
        "Canonical reconciliation finished: "
        f"rows={len(canonical_rows)}, "
        f"backfilled={len(backfilled_rows)}, "
        f"dropped={len(dropped_rows)}"
    )

    if len(canonical_rows) != EXPECTED_AFTER_2009_MASTER_ROWS:
        raise RuntimeError(
            f"Canonical master row count mismatch: expected {EXPECTED_AFTER_2009_MASTER_ROWS}, got {len(canonical_rows)}"
        )
    if len({row["meeting_date"] for row in canonical_rows}) != 128:
        raise RuntimeError("Canonical master unique meeting count mismatch: expected 128")
    if len(backfilled_rows) != 2:
        raise RuntimeError(f"Backfilled row count mismatch: expected 2, got {len(backfilled_rows)}")
    if len(dropped_rows) != 1:
        raise RuntimeError(f"Dropped row count mismatch: expected 1, got {len(dropped_rows)}")

    dropped_row = dropped_rows[0]
    for key, expected_value in EXPECTED_DROPPED_ROW.items():
        if dropped_row[key] != expected_value:
            raise RuntimeError(f"Dropped row mismatch for {key}: expected {expected_value}, got {dropped_row[key]}")

    audit = {
        "source_counts": {
            "archived_labeled_rows": len(labeled_frame),
            "archived_prompt_rows": len(prompt_rows),
            "archived_response_rows": len(response_rows),
            "archived_response_empty_rows": sum(1 for row in response_rows if not str(row.get("response", "")).strip()),
            "dataset_raw_backfill_rows": len(raw_backfill_rows),
            "backfill_candidate_files": len(backfill_sources),
            "prompt_only_index_rows": len(prompt_only),
            "response_only_index_rows": len(response_only),
            "prompt_response_metadata_mismatch_rows": len(metadata_mismatch),
            "prompt_response_prompt_text_match_rows": sum(
                1
                for index in sorted(set(prompt_by_index) & set(response_by_index))
                if prompt_by_index[index]["prompt"] == response_by_index[index]["prompt"]
            ),
        },
        "canonical_counts": {
            "rows": len(canonical_rows),
            "unique_meeting_dates": len({row["meeting_date"] for row in canonical_rows}),
            "backfilled_rows": len(backfilled_rows),
            "dropped_rows": len(dropped_rows),
            "empty_responses": sum(1 for row in canonical_rows if not row["response"].strip()),
        },
        "join_strategy": {
            "prompt_source_used_for_alignment": True,
            "alignment_key": "source_row_index",
            "exported_prompt_source": "archived_response_rows",
            "backfill_lookup_key": "exported_prompt",
            "raw_backfill_source": str(raw_backfill_source),
            "raw_backfill_candidates": [str(path) for path, _ in backfill_sources],
        },
    }
    return canonical_rows, {"source_reconciliation": audit, "backfilled_rows": backfilled_rows, "dropped_rows": dropped_rows}


def build_meeting_level_split(master_rows: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    log("Building chronological meeting-level split.")
    meeting_dates = sorted({row["meeting_date"] for row in master_rows})
    train_meetings = set(meeting_dates[:102])
    eval_meetings = set(meeting_dates[102:115])
    test_meetings = set(meeting_dates[115:])

    if not (len(train_meetings), len(eval_meetings), len(test_meetings)) == (102, 13, 13):
        raise RuntimeError("Meeting split sizes are not 102/13/13")
    if train_meetings & eval_meetings or train_meetings & test_meetings or eval_meetings & test_meetings:
        raise RuntimeError("Meeting split overlap detected")

    split_lookup = {}
    for meeting_date in train_meetings:
        split_lookup[meeting_date] = "train"
    for meeting_date in eval_meetings:
        split_lookup[meeting_date] = "eval"
    for meeting_date in test_meetings:
        split_lookup[meeting_date] = "test"

    split_rows = {"train": [], "eval": [], "test": []}
    for row in master_rows:
        split_rows[split_lookup[row["meeting_date"]]].append(row)

    meeting_sample_counts = Counter(row["meeting_date"] for row in master_rows)
    split_manifest = []
    for meeting_date in meeting_dates:
        split_manifest.append(
            {
                "meeting_date": meeting_date,
                "split": split_lookup[meeting_date],
                "sample_count": meeting_sample_counts[meeting_date],
            }
        )

    return split_rows, split_manifest


def build_legacy_split_manifests(master_rows: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    log("Building sample-level legacy split manifests.")
    master_by_prompt = {row["prompt"]: row for row in master_rows}
    master_by_composite: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in master_rows:
        key = (row["meeting_date"], row["section_name"], row["topic"], row["provided_data"])
        master_by_composite.setdefault(key, []).append(row)

    split_rows: dict[str, list[dict]] = {}
    split_manifest: list[dict] = []
    used_sample_ids: set[str] = set()

    for split_name, filename in [("train", "fomc_qa_sft_train.jsonl"), ("eval", "fomc_qa_eval.jsonl"), ("test", "fomc_qa_test.jsonl")]:
        source_rows = load_jsonl(LEGACY_SPLIT_ROOT / filename)
        manifest_rows = []
        matched = 0
        fallback_matched = 0
        prompt_digest = hashlib.sha256()
        for order_idx, row in enumerate(source_rows):
            prompt_digest.update(row["prompt"].encode("utf-8"))
            prompt_digest.update(b"\n")
            master_row = master_by_prompt.get(row["prompt"])
            metadata = parse_prompt_metadata(row["prompt"])
            if not master_row:
                key = (metadata["meeting_date"], metadata["section_name"], metadata["topic"], row.get("provided_data", ""))
                for candidate in master_by_composite.get(key, []):
                    if candidate["sample_id"] not in used_sample_ids:
                        master_row = candidate
                        fallback_matched += 1
                        break
            if master_row:
                matched += 1
                metadata = {
                    "meeting_date": master_row["meeting_date"],
                    "section_name": master_row["section_name"],
                    "topic": master_row["topic"],
                }
                sample_id = master_row["sample_id"]
                used_sample_ids.add(sample_id)
            else:
                sample_id = ""

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split_name,
                    "order_in_split": order_idx,
                    "meeting_date": metadata["meeting_date"],
                    "section_name": metadata["section_name"],
                    "topic": metadata["topic"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "provided_data": row["provided_data"],
                    "prompt_hash": prompt_hash(row["prompt"]),
                }
            )

        if matched != len(source_rows):
            raise RuntimeError(f"Legacy split {split_name} has unmatched prompts: {len(source_rows) - matched}")

        split_rows[split_name] = source_rows
        split_manifest.append(
            {
                "split": split_name,
                "rows": len(source_rows),
                "matched_master_rows": matched,
                "fallback_composite_matches": fallback_matched,
                "prompt_sha256": prompt_digest.hexdigest(),
            }
        )
        split_rows[f"{split_name}_manifest"] = manifest_rows

    return split_rows, split_manifest


def export_outputs(export_root: Path, master_rows: list[dict], meeting_split_rows: dict[str, list[dict]], meeting_split_manifest: list[dict], legacy_split_rows: dict[str, list[dict]], legacy_split_manifest: list[dict], audit: dict) -> None:
    log(f"Exporting Chapter 2 QA outputs to {export_root}")
    for managed_dir in ("master", "meeting_level", "sample_level_legacy", "audit"):
        target = export_root / managed_dir
        if target.exists():
            shutil.rmtree(target)

    master_path = export_root / "master" / "qa_master.jsonl"
    write_jsonl(master_path, master_rows)
    write_jsonl(PROCESSED_INPUT_ROOT / "fomc_qa.jsonl", [minimal_training_view(row) for row in master_rows])

    for split_name in ["train", "eval", "test"]:
        write_jsonl(
            export_root / "meeting_level" / f"{split_name}.jsonl",
            [minimal_training_view(row) for row in meeting_split_rows[split_name]],
        )
        write_jsonl(
            export_root / "meeting_level" / f"{split_name}_manifest.jsonl",
            [
                {
                    **row,
                    "split": split_name,
                    "prompt_hash": prompt_hash(row["prompt"]),
                }
                for row in meeting_split_rows[split_name]
            ],
        )
        write_jsonl(
            CANONICAL_MANIFEST_ROOT / f"qa_{split_name}_manifest.jsonl",
            [
                {
                    **row,
                    "split": split_name,
                    "prompt_hash": prompt_hash(row["prompt"]),
                }
                for row in meeting_split_rows[split_name]
            ],
        )

        write_jsonl(
            export_root / "sample_level_legacy" / f"{split_name}.jsonl",
            [minimal_training_view(row) for row in meeting_split_rows[split_name]],
        )
        write_jsonl(
            export_root / "sample_level_legacy" / f"{split_name}_manifest.jsonl",
            [
                {
                    **row,
                    "split": split_name,
                    "prompt_hash": prompt_hash(row["prompt"]),
                }
                for row in meeting_split_rows[split_name]
            ],
        )

    write_json(export_root / "audit" / "source_reconciliation.json", audit["source_reconciliation"])
    write_json(export_root / "audit" / "backfilled_rows.json", audit["backfilled_rows"])
    write_json(export_root / "audit" / "dropped_rows.json", audit["dropped_rows"])
    write_json(export_root / "audit" / "meeting_split_manifest.json", meeting_split_manifest)
    write_json(CANONICAL_MANIFEST_ROOT / "qa_meeting_split_manifest.json", meeting_split_manifest)
    write_json(export_root / "audit" / "legacy_split_manifest.json", legacy_split_manifest)
    log("Export complete.")


def inspect_qa_master_export(
    export_root: Path,
    *,
    input_root: Path = PROCESSED_INPUT_ROOT,
    split_manifest_paths: dict[str, Path] | None = None,
    expected_master_rows: int = EXPECTED_AFTER_2009_MASTER_ROWS,
    expected_split_counts: dict[str, int] | None = None,
) -> dict:
    expected_split_counts = dict(expected_split_counts or EXPECTED_AFTER_2009_SPLIT_ROWS)
    split_manifest_paths = split_manifest_paths or {
        split: CANONICAL_MANIFEST_ROOT / f"qa_{split}_manifest.jsonl"
        for split in ("train", "eval", "test")
    }

    def _row_count(path: Path) -> int | None:
        return len(load_jsonl(path)) if path.exists() else None

    audit_paths = {
        "source_reconciliation": export_root / "audit" / "source_reconciliation.json",
        "backfilled_rows": export_root / "audit" / "backfilled_rows.json",
        "dropped_rows": export_root / "audit" / "dropped_rows.json",
        "meeting_split_manifest": export_root / "audit" / "meeting_split_manifest.json",
        "legacy_split_manifest": export_root / "audit" / "legacy_split_manifest.json",
    }
    actual_counts = {
        "qa_master": _row_count(export_root / "master" / "qa_master.jsonl"),
        "fomc_qa": _row_count(input_root / "fomc_qa.jsonl"),
        "canonical_manifests": {
            split: _row_count(path)
            for split, path in split_manifest_paths.items()
        },
        "meeting_level": {
            split: _row_count(export_root / "meeting_level" / f"{split}.jsonl")
            for split in ("train", "eval", "test")
        },
        "sample_level_legacy": {
            split: _row_count(export_root / "sample_level_legacy" / f"{split}.jsonl")
            for split in ("train", "eval", "test")
        },
        "audit_files_present": {
            name: path.exists()
            for name, path in audit_paths.items()
        },
    }
    expected_counts = {
        "qa_master": expected_master_rows,
        "fomc_qa": expected_master_rows,
        "canonical_manifests": expected_split_counts,
        "meeting_level": expected_split_counts,
        "sample_level_legacy": expected_split_counts,
    }

    reason_parts: list[str] = []
    if actual_counts["qa_master"] != expected_master_rows:
        reason_parts.append("qa_master_count_mismatch")
    if actual_counts["fomc_qa"] != expected_master_rows:
        reason_parts.append("fomc_qa_count_mismatch")
    for bucket_name in ("canonical_manifests", "meeting_level", "sample_level_legacy"):
        for split, expected in expected_split_counts.items():
            if actual_counts[bucket_name][split] != expected:
                reason_parts.append(f"{bucket_name}_{split}_count_mismatch")
    if not all(actual_counts["audit_files_present"].values()):
        reason_parts.append("audit_files_missing")

    return {
        "complete": not reason_parts,
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def build_qa_master(
    source: str,
    scope: str,
    export_root: Path,
    *,
    config_path: str | None = None,
) -> Path:
    if source != "archived":
        raise ValueError("Only --source archived is supported in this build.")
    if scope != "after_2009":
        raise ValueError("Only --scope after_2009 is supported in this build.")

    log(f"Starting build: source={source}, scope={scope}, export_root={export_root}")
    config = load_pipeline_config(config_path)
    master_rows, audit = build_canonical_master(
        response_template=get_response_template(config),
    )
    meeting_split_rows, meeting_split_manifest = build_meeting_level_split(master_rows)
    legacy_split_rows, legacy_split_manifest = build_legacy_split_manifests(master_rows)
    export_outputs(export_root, master_rows, meeting_split_rows, meeting_split_manifest, legacy_split_rows, legacy_split_manifest, audit)
    log("Build finished successfully.")
    return export_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build QA master datasets from archived prompt-generation artifacts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default="archived")
    parser.add_argument("--scope", default="after_2009")
    parser.add_argument(
        "--export-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "processed" / "pipeline" / "analysis_sft",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output_root = build_qa_master(
        args.source,
        args.scope,
        args.export_root,
        config_path=args.config,
    )
    print(f"QA master data exported to {output_root}")
