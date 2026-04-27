from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from get_response_from_llm.get_response import get_response


BASE_DIR = Path(__file__).resolve().parent


def run_response(input_jsonl: Path, output_file: Path, model_name: str | None = None) -> Path:
    rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed_index: list[int | str] = []
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = row.get("prompt", "")
            index = row.get("index")
            if not prompt:
                failed_index.append(index)
                continue

            try:
                response, reason = get_response(prompt, model_name=model_name)
            except Exception:
                failed_index.append(index)
                continue

            row = dict(row)
            row["response"] = response
            row["reasoning"] = reason
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if failed_index:
        pd.DataFrame({"failed_index": failed_index}).to_csv(
            output_file.with_name(f"{output_file.stem}_failed_index.csv"),
            index=False,
        )
    return output_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher responses for FOMC prompt JSONL files.")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--model-name", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output_path = run_response(args.input_jsonl, args.output_jsonl, model_name=args.model_name)
    print(f"Saved responses to {output_path}")
