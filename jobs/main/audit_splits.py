from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import load_json, load_jsonl, write_json


EXPECTED = {
    "analysis_minutes": {"source_meetings": 128, "split_counts": {"train": 102, "eval": 13, "test": 13}},
    "decision": {"source_meetings": 237, "split_counts": {"train": 189, "eval": 24, "test": 24}},
}


def _read_json(path: Path) -> dict:
    return load_json(path)  # type: ignore[arg-type]


def _audit_dataset_dir(dataset_dir: Path) -> dict:
    audit = {"dataset_dir": str(dataset_dir), "splits": {}, "meeting_overlaps": {}}
    meeting_sets: dict[str, set[str]] = {}
    for split in ("train", "eval", "test"):
        rows = load_jsonl(dataset_dir / f"{split}.jsonl")
        meetings = {str(row["meeting_date"])[:10] for row in rows}
        meeting_sets[split] = meetings
        audit["splits"][split] = {"rows": len(rows), "meetings": len(meetings)}

    pairs = [("train", "eval"), ("train", "test"), ("eval", "test")]
    for left, right in pairs:
        overlap = sorted(meeting_sets[left] & meeting_sets[right])
        audit["meeting_overlaps"][f"{left}__{right}"] = {"count": len(overlap), "sample": overlap[:10]}
    return audit


def _check_manifest_counts(payload: dict, expected: dict, label: str) -> None:
    meeting_count_by_split = payload["meeting_count_by_split"]
    if payload["split_seed"] != 42:
        raise ValueError(f"{label} split_seed must be 42, got {payload['split_seed']}")
    if len(payload["source_population_meetings"]) != expected["source_meetings"]:
        raise ValueError(
            f"{label} source population mismatch: expected {expected['source_meetings']}, "
            f"got {len(payload['source_population_meetings'])}"
        )
    if meeting_count_by_split != expected["split_counts"]:
        raise ValueError(f"{label} split counts mismatch: expected {expected['split_counts']}, got {meeting_count_by_split}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit canonical main-task splits.")
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "dataset" / "processed" / "main" / "manifests" / "audit_report.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    analysis_manifest = _read_json(args.manifest_root / "analysis_minutes_split.json")
    decision_manifest = _read_json(args.manifest_root / "decision_split.json")

    _check_manifest_counts(analysis_manifest, EXPECTED["analysis_minutes"], "analysis/minutes")
    _check_manifest_counts(decision_manifest, EXPECTED["decision"], "decision")

    audit_report = {
        "analysis_minutes_split_manifest": analysis_manifest,
        "decision_split_manifest": decision_manifest,
        "datasets": {
            "analysis_sft": _audit_dataset_dir(args.dataset_root / "analysis_sft"),
            "analysis_grpo": _audit_dataset_dir(args.dataset_root / "analysis_grpo"),
            "minutes_alignment": _audit_dataset_dir(args.dataset_root / "minutes_alignment"),
            "decision_grpo": _audit_dataset_dir(args.dataset_root / "decision_grpo"),
        },
    }

    for dataset_name, dataset_audit in audit_report["datasets"].items():
        for pair_name, overlap in dataset_audit["meeting_overlaps"].items():
            if overlap["count"] != 0:
                raise ValueError(f"{dataset_name} has meeting leakage across {pair_name}: {overlap}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, audit_report)

    print("✅ Split audit passed")
    for dataset_name, dataset_audit in audit_report["datasets"].items():
        split_summary = ", ".join(
            f"{split}={summary['rows']} rows/{summary['meetings']} meetings"
            for split, summary in dataset_audit["splits"].items()
        )
        print(f"- {dataset_name}: {split_summary}")


if __name__ == "__main__":
    main()
