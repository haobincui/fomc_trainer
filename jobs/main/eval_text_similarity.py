from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import write_json
from open_r1.validator.cos.cos_calc import cosine_similarity_calc
from open_r1.validator.cos.embedding_model import get_cached_embedding_model

try:
    from bert_score import score as bertscore_score
except ImportError:  # pragma: no cover - handled through CLI error
    bertscore_score = None


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _load_generation_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported generation file: {path}")


def _extract_segment(text: str, segment: str) -> str:
    if not isinstance(text, str):
        return ""
    if segment == "full":
        return text
    if segment == "think":
        match = THINK_PATTERN.search(text)
        return match.group(1).strip() if match else ""
    if segment == "answer":
        match = ANSWER_PATTERN.search(text)
        return match.group(1).strip() if match else text.split("</think>")[-1].strip()
    raise ValueError(f"Unsupported segment: {segment}")


def _build_aligned_frame(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = reference.merge(
        candidate[["meeting_date", "section_name", "generated"]],
        on=["meeting_date", "section_name"],
        how="inner",
        suffixes=("", "_candidate"),
        validate="one_to_one",
    )
    merged = merged.rename(columns={"generated": f"generated_{label}"})
    return merged


def _segment_metrics(df: pd.DataFrame, label: str, segment: str, embedding_model_path: str, bertscore_model: str) -> dict:
    if bertscore_score is None:
        raise ImportError("bert-score is required for text similarity evaluation. Please install the project requirements.")

    references = df["response"].map(lambda text: _extract_segment(text, segment)).tolist()
    generations = df[f"generated_{label}"].map(lambda text: _extract_segment(text, segment)).tolist()
    embedding_model = get_cached_embedding_model(embedding_model_path)
    cosine_values = [
        cosine_similarity_calc(reference, generated, embedding_model)
        for reference, generated in zip(references, generations)
    ]
    precision, recall, f1 = bertscore_score(
        generations,
        references,
        model_type=bertscore_model,
        lang="en",
        verbose=False,
    )
    return {
        "n_samples": len(df),
        "cosine_mean": float(sum(cosine_values) / len(cosine_values)),
        "bertscore_precision_mean": float(precision.mean().item()),
        "bertscore_recall_mean": float(recall.mean().item()),
        "bertscore_f1_mean": float(f1.mean().item()),
    }


def evaluate_text_similarity(
    *,
    baseline_file: Path,
    aligned_file: Path,
    embedding_model_path: str,
    bertscore_model: str,
    output_json: Path,
) -> dict:
    baseline_df = _load_generation_file(baseline_file)
    aligned_df = _load_generation_file(aligned_file)

    required_columns = {"meeting_date", "section_name", "response", "generated"}
    if not required_columns.issubset(baseline_df.columns) or not required_columns.issubset(aligned_df.columns):
        raise ValueError("Both input files must contain meeting_date, section_name, response, and generated columns.")

    baseline_df["meeting_date"] = baseline_df["meeting_date"].astype(str).str[:10]
    aligned_df["meeting_date"] = aligned_df["meeting_date"].astype(str).str[:10]

    reference = baseline_df[["meeting_date", "section_name", "response", "generated"]].copy()
    reference = reference.rename(columns={"generated": "generated_baseline"})
    comparison = reference.merge(
        aligned_df[["meeting_date", "section_name", "generated"]],
        on=["meeting_date", "section_name"],
        how="inner",
        suffixes=("", "_aligned"),
        validate="one_to_one",
    )
    comparison = comparison.rename(columns={"generated": "generated_aligned"})

    results = {
        "baseline_file": str(baseline_file),
        "aligned_file": str(aligned_file),
        "embedding_model_path": embedding_model_path,
        "bertscore_model": bertscore_model,
        "segments": {},
    }
    for segment in ("full", "think", "answer"):
        baseline_metrics = _segment_metrics(comparison, "baseline", segment, embedding_model_path, bertscore_model)
        aligned_metrics = _segment_metrics(comparison, "aligned", segment, embedding_model_path, bertscore_model)
        results["segments"][segment] = {
            "baseline": baseline_metrics,
            "aligned": aligned_metrics,
            "delta": {
                key: aligned_metrics[key] - baseline_metrics[key]
                for key in baseline_metrics
                if key.endswith("_mean")
            },
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate text similarity on held-out minutes meetings.")
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--aligned-file", type=Path, required=True)
    parser.add_argument(
        "--embedding-model-path",
        default="models/DeepSeek-R1-Distill-Llama-8B",
    )
    parser.add_argument("--bertscore-model", default="bert-base-uncased")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "output" / "evaluation" / "main" / "text_similarity.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate_text_similarity(
        baseline_file=args.baseline_file,
        aligned_file=args.aligned_file,
        embedding_model_path=args.embedding_model_path,
        bertscore_model=args.bertscore_model,
        output_json=args.output_json,
    )
    print("✅ Text similarity evaluation finished")
    for segment, payload in results["segments"].items():
        print(
            f"- {segment}: cosine_delta={payload['delta']['cosine_mean']:.4f}, "
            f"bertscore_f1_delta={payload['delta']['bertscore_f1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
