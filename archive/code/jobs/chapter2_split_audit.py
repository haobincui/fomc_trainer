import argparse
import json
import re
from collections import Counter
from pathlib import Path


DATE_PATTERNS = [
    re.compile(r"meeting on \*\*(\d{4}-\d{2}-\d{2})(?: \d{2}:\d{2}:\d{2})?\*\*"),
    re.compile(r"meeting scheduled for \*\*(\d{4}-\d{2}-\d{2})\*\*"),
    re.compile(r"for the \*\*(\d{4}-\d{2}-\d{2})\*\* meeting"),
]

SECTION_PATTERNS = [
    re.compile(r"style of the \*\*(.*?)\*\* section", re.DOTALL),
    re.compile(r"aligned with the tone and structure of the \*\*(.*?)\*\* section", re.DOTALL),
    re.compile(r"Here is the Analysis for Section \*\*(.*?)\*\*:", re.DOTALL),
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_with_patterns(text: str, patterns: list[re.Pattern[str]]) -> str | None:
    if not isinstance(text, str):
        return None
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _extract_meeting_date(row: dict) -> str | None:
    if row.get("meeting_date"):
        return str(row["meeting_date"])[:10]
    prompt = row.get("prompt", "")
    return _extract_with_patterns(prompt, DATE_PATTERNS)


def _extract_section_name(row: dict) -> str | None:
    if row.get("section_name"):
        return str(row["section_name"]).strip()
    prompt = row.get("prompt", "")
    return _extract_with_patterns(prompt, SECTION_PATTERNS)


def _summarize_split(path: Path) -> dict:
    rows = _load_jsonl(path)
    meeting_dates = [_extract_meeting_date(row) for row in rows]
    section_names = [_extract_section_name(row) for row in rows]

    meeting_counter = Counter(date for date in meeting_dates if date)
    section_counter = Counter(name for name in section_names if name)

    return {
        "path": str(path),
        "rows": len(rows),
        "parseable_meeting_dates": sum(date is not None for date in meeting_dates),
        "unique_meetings": len(meeting_counter),
        "max_samples_per_meeting": max(meeting_counter.values()) if meeting_counter else 0,
        "parseable_sections": sum(name is not None for name in section_names),
        "unique_sections": len(section_counter),
        "top_sections": section_counter.most_common(5),
        "meeting_set": set(meeting_counter),
    }


def _group_summary(name: str, paths: list[Path]) -> dict:
    splits = {}
    for path in paths:
        split_name = path.stem
        splits[split_name] = _summarize_split(path)

    split_names = list(splits)
    overlaps = {}
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            left_set = splits[left]["meeting_set"]
            right_set = splits[right]["meeting_set"]
            overlap = sorted(left_set & right_set)
            overlaps[f"{left}__{right}"] = {
                "count": len(overlap),
                "sample": overlap[:10],
            }

    return {"group": name, "splits": splits, "overlaps": overlaps}


def _default_groups() -> dict[str, list[Path]]:
    return {
        "fomc_qa": [
            Path("dataset/training_data/fomc_qa/fomc_qa_sft_train.jsonl"),
            Path("dataset/training_data/fomc_qa/fomc_qa_eval.jsonl"),
            Path("dataset/training_data/fomc_qa/fomc_qa_test.jsonl"),
        ],
        "synthetic_text": [
            Path("dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_train.jsonl"),
            Path("dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_eval.jsonl"),
            Path("dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_test.jsonl"),
        ],
        "decision_making": [
            Path("dataset/training_data/decision_making/decision_making_20250526_reason_train.jsonl"),
            Path("dataset/training_data/decision_making/decision_making_20250526_reason_eval.jsonl"),
        ],
        "decision_grpo": [
            Path("dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_train.jsonl"),
            Path("dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_eval.jsonl"),
        ],
    }


def _print_report(report: dict) -> None:
    for group_name, group in report.items():
        print(f"\n## {group_name}")
        for split_name, summary in group["splits"].items():
            print(
                f"- {split_name}: rows={summary['rows']}, "
                f"parseable_meeting_dates={summary['parseable_meeting_dates']}, "
                f"unique_meetings={summary['unique_meetings']}, "
                f"max_samples_per_meeting={summary['max_samples_per_meeting']}, "
                f"parseable_sections={summary['parseable_sections']}, "
                f"unique_sections={summary['unique_sections']}"
            )
            if summary["top_sections"]:
                print(f"  top_sections={summary['top_sections']}")
        for pair_name, overlap in group["overlaps"].items():
            print(
                f"  overlap[{pair_name}]={overlap['count']} "
                f"sample={overlap['sample']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Chapter 2 dataset splits.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the audit report as JSON.",
    )
    args = parser.parse_args()

    report = {}
    for group_name, paths in _default_groups().items():
        report[group_name] = _group_summary(group_name, paths)
        for summary in report[group_name]["splits"].values():
            summary["meeting_set"] = sorted(summary["meeting_set"])

    _print_report(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
