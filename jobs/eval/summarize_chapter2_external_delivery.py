#!/usr/bin/env python3
"""Summarise frozen external decision outputs; no model inference is performed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RELEASES = (
    "chk0_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v1_20260824",
    "chk4_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v3_20260824",
)
MODELS = ("model_chk0", "model_chk1_cp200", "model_chk3_sft_cp38", "model_chk3_grpo_cp450")
PANELS = {"historical_n19": 19, "postcutoff_n12": 12}
DIRECTIONS = ("cut", "hold", "hike")
EXPECTED_CORRECT = {
    "historical_n19": (53, 41, 56, 48),
    "postcutoff_n12": (16, 17, 21, 22),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(evaluation_root: Path) -> tuple[dict, list[dict], list[dict]]:
    rows, samples, sources = [], {}, []
    for release in RELEASES:
        root = evaluation_root / release
        source_rows = read_jsonl(root / "run/results.jsonl")
        rows.extend(source_rows)
        sources.append({"results": str(root / "run/results.jsonl"), "rows": len(source_rows)})
        for sample in read_jsonl(root / "inputs/panel_samples.jsonl"):
            key = sample["sample_id"]
            identity = {k: sample[k] for k in ("panel", "meeting_id", "direction", "magnitude_bp", "prompt")}
            if key in samples:
                assert samples[key] == identity, f"cross-model sample mismatch: {key}"
            samples[key] = identity
    assert len(rows) == 1240 and len(samples) == 31
    keyed = {(r["model_label"], r["sample_id"], r["replicate_id"]): r for r in rows}
    expected = {(m, s, k) for m in MODELS for s in samples for k in range(10)}
    assert len(keyed) == len(rows) and set(keyed) == expected, "missing or duplicate prediction"
    for sample_id, sample in samples.items():
        for k in range(10):
            aligned = [keyed[m, sample_id, k] for m in MODELS]
            assert len({r["row_seed"] for r in aligned}) == 1
            for row in aligned:
                assert row["panel"] == sample["panel"]
                assert row["target_direction"] == sample["direction"]
                assert row["target_magnitude_bp"] == sample["magnitude_bp"]
                pred = (row["decision_prediction"] or {}).get("direction", "invalid")
                assert pred in (*DIRECTIONS, "invalid")
                assert bool(row["decision_direction_correct"]) == (pred == sample["direction"])
                # These properties are verified for these releases, not assumed generally.
                assert bool(row["strict_json"]) == bool(row["delivery_valid"])
                assert bool(row["hit_eos"]) != bool(row["cap_reached"])
                if pred != "invalid" and not row["strict_json"]:
                    assert row["response_format"] == "fenced_json"

    delivery, confusion, modal, baselines = [], [], [], []
    for panel, n_meetings in PANELS.items():
        panel_samples = {s: v for s, v in samples.items() if v["panel"] == panel}
        assert len(panel_samples) == n_meetings
        classes = Counter(v["direction"] for v in panel_samples.values())
        baselines.append({"panel": panel, "meetings": n_meetings,
                          "cut": classes["cut"], "hold": classes["hold"], "hike": classes["hike"],
                          "hold_accuracy": classes["hold"] / n_meetings,
                          "hold_balanced_accuracy_observed_classes": 1 / len(classes)})
        for model_index, model in enumerate(MODELS):
            block = [r for r in rows if r["model_label"] == model and r["panel"] == panel]
            counts, matrix, meeting_labels = Counter(), Counter(), defaultdict(list)
            for row in block:
                pred = (row["decision_prediction"] or {}).get("direction", "invalid")
                branch = "strict" if row["strict_json"] else ("fenced" if pred != "invalid" else "invalid")
                suffix = "correct" if row["decision_direction_correct"] else "wrong"
                counts[f"{branch}_{suffix}"] += 1
                matrix[row["target_direction"], pred] += 1
                meeting_labels[row["sample_id"]].append(pred)
            correct = counts["strict_correct"] + counts["fenced_correct"]
            assert correct == EXPECTED_CORRECT[panel][model_index], "published accuracy mismatch"
            assert sum(counts.values()) == 10 * n_meetings and not counts["invalid_correct"]
            valid = len(block) - counts["invalid_wrong"]
            delivery.append({"panel": panel, "model": model, "meetings": n_meetings, "outputs": len(block),
                             **{k: counts[k] for k in ("strict_correct", "strict_wrong", "fenced_correct", "fenced_wrong")},
                             "invalid": counts["invalid_wrong"], "eos": sum(r["hit_eos"] for r in block),
                             "token_cap": sum(r["cap_reached"] for r in block), "correct": correct,
                             "direction_accuracy": correct / len(block), "parseable": valid,
                             "conditional_direction_accuracy": correct / valid})
            for true in DIRECTIONS:
                support = 10 * classes[true]
                confusion.append({"panel": panel, "model": model, "true_direction": true, "outputs": support,
                                  **{f"pred_{pred}": matrix[true, pred] for pred in (*DIRECTIONS, "invalid")},
                                  "recall": matrix[true, true] / support if support else None})
                assert sum(matrix[true, p] for p in (*DIRECTIONS, "invalid")) == support
            modal_counts = Counter()
            for sample_id, predictions in meeting_labels.items():
                c = Counter(predictions)
                winners = [label for label, n in c.items() if n == max(c.values())]
                category = "tie" if len(winners) > 1 else ("invalid_mode" if winners[0] == "invalid" else
                            ("correct_mode" if winners[0] == samples[sample_id]["direction"] else "wrong_valid_mode"))
                modal_counts[category] += 1
            modal.append({"panel": panel, "model": model,
                          **{k: modal_counts[k] for k in ("correct_mode", "wrong_valid_mode", "invalid_mode", "tie")}})
    summary = {"script": "fomc_trainer/jobs/eval/summarize_chapter2_external_delivery.py",
               "sources": sources, "method": "Saved parser outputs; invalid predictions retained as errors. Ten paired replicates per meeting.",
               "validated_rows": len(rows), "validated_meetings": len(samples), "delivery": delivery,
               "modal": modal, "constant_hold": baselines,
               "pooled_constant_hold": {"accuracy": 19 / 31, "balanced_accuracy_three_class": 1 / 3}}
    return summary, delivery, confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=REPO / "output/evaluation/retrain_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO.parent / "PhdThesis/Chapter2/results/external_delivery_20261011")
    args = parser.parse_args()
    summary, delivery, confusion = summarise(args.evaluation_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "delivery_correctness.csv", delivery)
    write_csv(args.output_dir / "confusion_matrices.csv", confusion)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {summary['validated_rows']} outputs across {summary['validated_meetings']} meetings; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
