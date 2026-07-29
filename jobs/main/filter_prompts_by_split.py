from __future__ import annotations

import argparse
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import load_json, load_jsonl, write_jsonl


DEFAULT_MANIFEST = ROOT / "dataset" / "processed" / "main" / "manifests" / "analysis_minutes_split.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter JSONL prompt files to the meetings assigned to a canonical main split.")
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=["train", "eval", "test"], default="test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_manifest = load_json(args.split_manifest)
    allowed_meetings = set(split_manifest["meeting_dates_by_split"][args.split])

    args.output_folder.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_files = 0
    for input_file in sorted(glob.glob(str(args.input_folder / "*.jsonl"))):
        rows = [row for row in load_jsonl(Path(input_file)) if str(row.get("meeting_date", ""))[:10] in allowed_meetings]
        output_file = args.output_folder / Path(input_file).name
        write_jsonl(output_file, rows)
        total_rows += len(rows)
        total_files += 1

    print("✅ Prompt filtering finished")
    print(f"- files: {total_files}")
    print(f"- rows: {total_rows}")
    print(f"- split: {args.split}")
    print(f"- output: {args.output_folder}")


if __name__ == "__main__":
    main()
