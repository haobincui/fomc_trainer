"""Generate one task-aligned SFT sample from one checkpoint on one GPU.

The caller pins the physical GPU with ``CUDA_VISIBLE_DEVICES``.  This wrapper
keeps the selected dataset row, system prompt, seed, and decoding metadata in
the resulting JSONL so checkpoint comparisons remain auditable.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from generate_new_response import generate_new_response


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible) != 1:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must expose exactly one physical GPU; "
            f"observed={visible!r}"
        )
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.dataset.is_file() or not args.config.is_file():
        raise FileNotFoundError("dataset or config is missing")
    if args.output_file.exists():
        raise FileExistsError(args.output_file)
    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    if args.max_new_tokens <= 0 or args.max_model_len <= args.max_new_tokens:
        raise ValueError("invalid generation token budget")

    with args.dataset.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.sample_index >= len(rows):
        raise IndexError(
            f"sample-index {args.sample_index} is outside {len(rows)} rows"
        )
    row = dict(rows[args.sample_index])
    row["sample_id"] = args.sample_id

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("config must contain a non-empty system_prompt")

    outputs = generate_new_response(
        [row],
        str(args.model),
        output_file=str(args.output_file),
        batch_size=1,
        seed=args.seed,
        replicate_id=f"temperature-{args.temperature:g}",
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        tokenizer_path=str(args.model),
        system_prompt=system_prompt,
        generation_metadata={
            "probe_schema_version": "task-aligned-single-checkpoint-v1",
            "probe_created_at": utc_now(),
            "physical_gpu": visible[0],
            "source_dataset": str(args.dataset),
            "source_config": str(args.config),
        },
        fail_closed=True,
    )
    if len(outputs) != 1:
        raise RuntimeError(f"expected one output, observed {len(outputs)}")


if __name__ == "__main__":
    main()
