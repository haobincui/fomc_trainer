import argparse
import json
import re
from pathlib import Path

import pandas as pd


COUNT_TARGETS = {
    "fomc_qa_train": Path("dataset/training_data/fomc_qa/fomc_qa_sft_train.jsonl"),
    "fomc_qa_eval": Path("dataset/training_data/fomc_qa/fomc_qa_eval.jsonl"),
    "fomc_qa_test": Path("dataset/training_data/fomc_qa/fomc_qa_test.jsonl"),
    "synthetic_train": Path("dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_train.jsonl"),
    "synthetic_eval": Path("dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_eval.jsonl"),
    "synthetic_test": Path("dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_test.jsonl"),
    "decision_making_train": Path("dataset/training_data/decision_making/decision_making_20250526_reason_train.jsonl"),
    "decision_making_eval": Path("dataset/training_data/decision_making/decision_making_20250526_reason_eval.jsonl"),
    "decision_grpo_train": Path("dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_train.jsonl"),
    "decision_grpo_eval": Path("dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_eval.jsonl"),
    "decision_archive_train_result": Path("output/validation/generation_stage2_decision/eval_result/archive/decision_making_grpo_20250528_train_generated_result.xlsx"),
    "decision_archive_eval_result": Path("output/validation/generation_stage2_decision/eval_result/archive/decision_making_grpo_20250528_generated_result.xlsx"),
    "decision_20250531_backbone_train": Path("output/validation/generation_stage2_decision/eval_result/20250531/DeepSeek-R1-Distill-Llama-8B/decision_grpo_train_result.xlsx"),
    "decision_20250531_backbone_eval": Path("output/validation/generation_stage2_decision/eval_result/20250531/DeepSeek-R1-Distill-Llama-8B/decision_grpo_eval_result.xlsx"),
    "decision_20250531_checkpoint4_train": Path("output/validation/generation_stage2_decision/eval_result/20250531/llama_grpo_decision_cp1100_20250530/decision_grpo_train_result.xlsx"),
    "decision_20250531_checkpoint4_eval": Path("output/validation/generation_stage2_decision/eval_result/20250531/llama_grpo_decision_cp1100_20250530/decision_grpo_eval_result.xlsx"),
    "decision_summary_workbook": Path("process_decsion_output/summary.xlsx"),
    "raw_prompt_summary": Path("dataset/raw_data/prompt_summary_20250512.jsonl"),
}

DATE_PATTERNS = [
    re.compile(r"meeting on \*\*(\d{4}-\d{2}-\d{2})(?: \d{2}:\d{2}:\d{2})?\*\*"),
    re.compile(r"meeting scheduled for \*\*(\d{4}-\d{2}-\d{2})\*\*"),
    re.compile(r"for the \*\*(\d{4}-\d{2}-\d{2})\*\* meeting"),
]


def _count_rows(path: Path) -> int:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.suffix == ".xlsx":
        if path.name == "summary.xlsx":
            workbook = pd.ExcelFile(path)
            return int(sum(len(pd.read_excel(path, sheet_name=sheet)) for sheet in workbook.sheet_names))
        return len(pd.read_excel(path))
    raise ValueError(f"Unsupported file type: {path}")


def _extract_unique_meetings_from_prompt_summary(path: Path) -> int:
    meeting_dates = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("meeting_date"):
                meeting_dates.add(str(row["meeting_date"])[:10])
                continue
            prompt = row.get("prompt", "")
            for pattern in DATE_PATTERNS:
                match = pattern.search(prompt)
                if match:
                    meeting_dates.add(match.group(1))
                    break
    return len(meeting_dates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Chapter 2 dataset and result counts.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the count summary as JSON.",
    )
    args = parser.parse_args()

    summary = {}
    print("\n## Chapter 2 Count Reconciliation")
    for label, path in COUNT_TARGETS.items():
        rows = _count_rows(path)
        summary[label] = {"path": str(path), "rows": rows}
        print(f"- {label}: rows={rows} ({path})")

    raw_meeting_count = _extract_unique_meetings_from_prompt_summary(COUNT_TARGETS["raw_prompt_summary"])
    summary["raw_prompt_summary"]["unique_meeting_dates"] = raw_meeting_count
    print(f"- raw_prompt_summary: unique_meeting_dates={raw_meeting_count}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
