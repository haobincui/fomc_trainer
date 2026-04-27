from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import (
    load_jsonl,
    training_view,
    write_jsonl,
)
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import (
    GEMMA4_THOUGHT_CHANNEL_TEMPLATE,
    parse_response_text,
)


DEFAULT_MODEL_PATH = Path("models/DeepSeek-R1-Distill-Llama-8B")
DEFAULT_SOURCE_ROOT = Path("dataset/processed/train/analysis_sft")
DEFAULT_OUTPUT_ROOT = Path("dataset/processed_llama/train/analysis_sft")
EXPECTED_GENERATION_SUFFIX = "<｜Assistant｜><think>\n"


def load_deepseek_tokenizer(model_path: str | Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def render_deepseek_generation_prompt(tokenizer: PreTrainedTokenizerBase) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def validate_deepseek_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    rendered_prompt = render_deepseek_generation_prompt(tokenizer)
    if not rendered_prompt.endswith(EXPECTED_GENERATION_SUFFIX):
        raise ValueError(
            "DeepSeek tokenizer chat template does not end with the expected generation suffix. "
            f"Expected tail={EXPECTED_GENERATION_SUFFIX!r}, actual tail={rendered_prompt[-120:]!r}"
        )
    return rendered_prompt


def convert_gemma_response_to_deepseek_completion(response_text: str | None) -> str:
    parsed = parse_response_text(response_text)
    if parsed.format_name != GEMMA4_THOUGHT_CHANNEL_TEMPLATE:
        raise ValueError(
            "Expected a Gemma thought-channel response, "
            f"got format={parsed.format_name!r} text={str(response_text or '')[:160]!r}"
        )

    reasoning_text = parsed.reasoning.strip()
    answer_text = parsed.answer.strip()
    if not reasoning_text:
        raise ValueError("Gemma thought-channel response is missing reasoning text.")
    if not answer_text:
        raise ValueError("Gemma thought-channel response is missing answer text.")
    return f"{reasoning_text}\n</think>\n{answer_text}"


def _convert_rows(rows: list[dict[str, Any]], *, split: str, label: str) -> list[dict[str, Any]]:
    converted_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        if "response" not in row:
            raise ValueError(f"Missing 'response' field in {label} row {row_index} for split={split}.")
        converted_row = dict(row)
        try:
            converted_row["response"] = convert_gemma_response_to_deepseek_completion(row.get("response"))
        except ValueError as exc:
            raise ValueError(
                f"Failed converting {label} row {row_index} for split={split}: {exc}"
            ) from exc
        converted_rows.append(converted_row)
    return converted_rows


def convert_analysis_sft_to_deepseek_llama(
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)

    tokenizer = load_deepseek_tokenizer(model_path)
    rendered_prompt = validate_deepseek_chat_template(tokenizer)

    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "expected_generation_suffix": EXPECTED_GENERATION_SUFFIX,
        "rendered_prompt_tail": rendered_prompt[-len(EXPECTED_GENERATION_SUFFIX) :],
        "splits": {},
    }

    for split in ("train", "eval", "test"):
        train_path = source_root / f"{split}.jsonl"
        manifest_path = source_root / f"{split}_manifest.jsonl"

        train_rows = load_jsonl(train_path)
        manifest_rows = load_jsonl(manifest_path)
        if len(train_rows) != len(manifest_rows):
            raise ValueError(
                f"Split={split} row count mismatch between training and manifest files: "
                f"{len(train_rows)} != {len(manifest_rows)}"
            )

        converted_train_rows = _convert_rows(train_rows, split=split, label="training")
        converted_manifest_rows = _convert_rows(manifest_rows, split=split, label="manifest")

        write_jsonl(output_root / f"{split}.jsonl", [training_view(row) for row in converted_train_rows])
        write_jsonl(output_root / f"{split}_manifest.jsonl", converted_manifest_rows)

        summary["splits"][split] = {
            "rows_read": len(train_rows),
            "rows_written": len(converted_train_rows),
            "parse_failures": 0,
        }

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert analysis_sft responses from Gemma thought-channel format to DeepSeek-R1-Llama completions."
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_analysis_sft_to_deepseek_llama(
        model_path=args.model_path,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
