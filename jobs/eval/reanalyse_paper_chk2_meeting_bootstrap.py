"""Reanalyse frozen CHK2 scores without rescoring, generating, or fitting models.

Primary intervals resample whole meetings within each original data partition.
Each selected meeting retains all its input texts and ten scored generations;
meetings have equal weight.  A separate conditional analysis holds the corpus
fixed and resamples paired generations independently for each input text.

The September scorer and its sealed outputs are historical records and are not
overwritten.  This CLI requires a new output directory, writes all contrasts
irrespective of their intervals, and depends only on NumPy and the standard
library.  Example (from the repository root)::

    python jobs/eval/reanalyse_paper_chk2_meeting_bootstrap.py \
      --source-dir output/evaluation/retrain_v2/paper_chk2_text_similarity_all391_chk0_chk1_chk2cp50_vllm_t06_p09_k10_b1000_v1_20260901 \
      --output-dir output/evaluation/retrain_v2/paper_chk2_meeting_bootstrap_reanalysis_b10000_20261019
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

MODELS = ("chk0", "chk1", "chk2")
METRICS = ("mpnet_cosine", "bertscore_f1")
POLICIES = ("raw_best_effort", "delivery_penalized")
SPLITS = ("train", "validation", "test")
CONTRASTS = (("chk1_minus_chk0", 0, 1), ("chk2_minus_chk1", 1, 2), ("chk2_minus_chk0", 0, 2))
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20_261_019
EXPECTED_SPLIT_ROWS = {"train": 305, "validation": 42, "test": 44}
EXPECTED_SPLIT_MEETINGS = {"train": 98, "validation": 13, "test": 13}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_seal(manifest: dict[str, Any]) -> None:
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    expected = hashlib.sha256(canonical(payload)).hexdigest()
    if manifest.get("integrity", {}).get("payload_sha256") != expected:
        raise ValueError("Source manifest integrity mismatch")


@dataclass
class Panel:
    # Scores: policy, model, metric, input text, replicate.
    scores: np.ndarray
    samples: list[dict[str, Any]]
    meetings: list[str]
    meeting_rows: list[np.ndarray]
    split_positions: dict[str, np.ndarray]
    audit: dict[str, Any]


def build_panel(samples: list[dict[str, Any]], rows: list[dict[str, Any]]) -> Panel:
    samples = sorted(samples, key=lambda row: row["sample_id"])
    sample_index = {row["sample_id"]: i for i, row in enumerate(samples)}
    if len(sample_index) != len(samples):
        raise ValueError("Duplicate input sample_id")
    replicate_ids = sorted({row["replicate_id"] for row in rows})
    if replicate_ids != list(range(len(replicate_ids))) or not replicate_ids:
        raise ValueError("Replicate IDs must be contiguous from zero")
    k = len(replicate_ids)
    scores = np.full((len(POLICIES), len(MODELS), len(METRICS), len(samples), k), np.nan)
    seen: set[tuple[str, str, int]] = set()
    paired_seeds: dict[tuple[str, int], tuple[int, int]] = {}
    score_contracts: set[str] = set()
    for row in rows:
        sample_id, model, replicate = row["sample_id"], row["model_id"], row["replicate_id"]
        key = (sample_id, model, replicate)
        if key in seen:
            raise ValueError(f"Duplicate score row: {key}")
        seen.add(key)
        if sample_id not in sample_index or model not in MODELS:
            raise ValueError(f"Unexpected score row: {key}")
        i = sample_index[sample_id]
        sample = samples[i]
        for field in ("meeting_id", "split", "prompt_sha256", "reference_sha256"):
            if row[field] != sample[field]:
                raise ValueError(f"Sample binding mismatch: {sample_id}, {field}")
        pair_key = (sample_id, replicate)
        seeds = (int(row["replicate_seed"]), int(row["row_seed"]))
        if paired_seeds.setdefault(pair_key, seeds) != seeds:
            raise ValueError(f"Cross-model seed pairing mismatch: {pair_key}")
        payload = f"{seeds[0]}|{sample_id}|{replicate}".encode()
        expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)
        if seeds[1] != expected_seed:
            raise ValueError(f"Request seed derivation mismatch: {pair_key}")
        score_contracts.add(row["score_contract_sha256"])
        for p, policy in enumerate(POLICIES):
            for t, metric in enumerate(METRICS):
                value = float(row[policy][metric])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite score: {key}, {policy}, {metric}")
                scores[p, MODELS.index(model), t, i, replicate] = value
    if np.isnan(scores).any():
        raise ValueError("Incomplete model/input/replicate score matrix")
    if len(score_contracts) != 1:
        raise ValueError("Multiple semantic scoring contracts")
    meetings = sorted({row["meeting_id"] for row in samples})
    meeting_rows = [np.array([i for i, row in enumerate(samples) if row["meeting_id"] == meeting],
                             dtype=np.int64) for meeting in meetings]
    split_by_meeting = []
    for indices in meeting_rows:
        meeting_splits = {samples[i]["split"] for i in indices}
        if len(meeting_splits) != 1 or not meeting_splits.issubset(SPLITS):
            raise ValueError("Meeting belongs to multiple or unknown data partitions")
        split_by_meeting.append(next(iter(meeting_splits)))
    split_positions = {split: np.array([i for i, value in enumerate(split_by_meeting) if value == split],
                                        dtype=np.int64) for split in SPLITS}
    if any(len(indices) == 0 for indices in split_positions.values()):
        raise ValueError("Every original data partition must contain a meeting")
    audit = {
        "score_rows": len(rows), "samples": len(samples), "meetings": len(meetings), "replicates": k,
        "split_samples": {split: sum(row["split"] == split for row in samples) for split in SPLITS},
        "split_meetings": {split: len(indices) for split, indices in split_positions.items()},
        "paired_sample_replicates": len(paired_seeds),
        "unique_request_seeds": len({seeds[1] for seeds in paired_seeds.values()}),
        "cross_model_seed_pairing_violations": 0, "seed_derivation_violations": 0,
        "sample_binding_violations": 0, "score_contract_sha256": next(iter(score_contracts)),
    }
    return Panel(scores, samples, meetings, meeting_rows, split_positions, audit)


def meeting_values(panel: Panel) -> np.ndarray:
    # Keep the historical estimand: within-meeting row means, then K means.
    return np.stack([panel.scores[..., indices, :].mean(axis=-2).mean(axis=-1)
                     for indices in panel.meeting_rows], axis=-1)


def meeting_bootstrap(panel: Panel, *, draws: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = meeting_values(panel)
    rng = np.random.default_rng(seed)
    plans = {}
    distributions = {}
    for split in SPLITS:
        positions = panel.split_positions[split]
        local = rng.integers(0, len(positions), size=(draws, len(positions)), dtype=np.int64)
        plans[split] = positions[local]
        distributions[split] = values[..., plans[split]].mean(axis=-1)
    distributions["pooled"] = sum(distributions[split] * (len(panel.split_positions[split]) / len(panel.meetings))
                                  for split in SPLITS)
    return distributions, plans


def row_replicate_plan(rng: np.random.Generator, *, draws: int, k: int) -> np.ndarray:
    """A fresh plan for one input text, shared only across its paired models."""
    return rng.integers(0, k, size=(draws, k), dtype=np.int64)


def generation_bootstrap(panel: Panel, *, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = panel.scores.shape[-1]
    weights = np.empty(len(panel.samples))
    for indices in panel.meeting_rows:
        weights[indices] = 1 / (len(panel.meetings) * len(indices))
    distribution = np.zeros((*panel.scores.shape[:3], draws))
    for i, weight in enumerate(weights):
        plan = row_replicate_plan(rng, draws=draws, k=k)
        distribution += panel.scores[..., i, plan].mean(axis=-1) * weight
    return distribution


def summarize(panel: Panel, distributions: dict[str, dict[str, np.ndarray]], *, draws: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = meeting_values(panel)
    points = {split: values[..., panel.split_positions[split]].mean(axis=-1) for split in SPLITS}
    points["pooled"] = values.mean(axis=-1)
    model_rows, contrast_rows = [], []
    for scheme, scopes in distributions.items():
        for scope, distribution in scopes.items():
            for p, policy in enumerate(POLICIES):
                for t, metric in enumerate(METRICS):
                    common = {"bootstrap_scheme": scheme, "scoring_policy": policy, "scope": scope,
                              "metric": metric, "confidence": .95, "bootstrap_draws": draws}
                    for m, model in enumerate(MODELS):
                        low, high = np.quantile(distribution[p, m, t], [.025, .975], method="linear")
                        model_rows.append({**common, "model_id": model, "estimate": float(points[scope][p, m, t]),
                                           "ci_lower": float(low), "ci_upper": float(high)})
                    for name, before, after in CONTRASTS:
                        delta = distribution[p, after, t] - distribution[p, before, t]
                        low, high = np.quantile(delta, [.025, .975], method="linear")
                        contrast_rows.append({**common, "contrast_id": name, "before_model": MODELS[before],
                                              "after_model": MODELS[after],
                                              "estimate": float(points[scope][p, after, t] - points[scope][p, before, t]),
                                              "ci_lower": float(low), "ci_upper": float(high)})
    return model_rows, contrast_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, output: Path, *, draws: int, seed: int) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if draws < 2 or seed < 0:
        raise ValueError("Require at least two bootstrap draws and a nonnegative seed")
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Output must be a new directory outside the historical source directory")
    score_manifest = json.loads((source / "score_manifest.json").read_text(encoding="utf-8"))
    complete_manifest = json.loads((source / "complete_evaluation_manifest.json").read_text(encoding="utf-8"))
    check_seal(score_manifest)
    check_seal(complete_manifest)
    bindings = {
        "row_scores.jsonl": score_manifest["artifacts"]["row_scores"]["sha256"],
        "samples.jsonl": score_manifest["inputs"]["samples"]["sha256"],
        "complete_evaluation_manifest.json": score_manifest["inputs"]["evaluation_manifest"]["sha256"],
    }
    for name, expected in bindings.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    panel = build_panel(read_jsonl(source / "samples.jsonl"), read_jsonl(source / "row_scores.jsonl"))
    if (panel.audit["split_samples"] != EXPECTED_SPLIT_ROWS or
            panel.audit["split_meetings"] != EXPECTED_SPLIT_MEETINGS or panel.audit["replicates"] != 10):
        raise ValueError("Source is not the original 391-input, 124-meeting K10 evaluation")
    main_draws, plans = meeting_bootstrap(panel, draws=draws, seed=seed)
    generation_draws = generation_bootstrap(panel, draws=draws, seed=seed + 1)
    distributions = {"whole_meeting_primary": main_draws,
                     "fixed_corpus_paired_generation_sensitivity": {"pooled": generation_draws}}
    model_rows, contrast_rows = summarize(panel, distributions, draws=draws)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "model_summary.csv", model_rows)
    write_csv(output / "pairwise_contrasts.csv", contrast_rows)
    # Preserve draws and actual meeting plans to permit a direct numerical audit.
    arrays = {f"meeting_plan_{split}": plan for split, plan in plans.items()}
    arrays.update({f"meeting_draws_{scope}": value for scope, value in main_draws.items()})
    arrays["generation_draws_pooled"] = generation_draws
    np.savez_compressed(output / "bootstrap_draws.npz", **arrays)
    manifest = {
        "schema_version": "paper-chk2-meeting-bootstrap-reanalysis-v1", "status": "complete",
        "operation": "reanalysis_of_frozen_scores", "source_directory": str(source),
        "reanalysis_does_not_modify_source": True, "new_model_fitting": False,
        "new_generation": False, "new_semantic_scoring": False,
        "inferential_scope": "internal reconstruction with fixed fitted models; not held-out generalization",
        "design_rationale": "meeting is the clustering unit; request-level generation seeds do not justify a common cross-meeting replicate shock",
        "primary": {
            "scheme": "whole_meeting_primary", "draws": draws, "seed": seed,
            "estimand": "equal-meeting mean similarity (or paired model difference), retaining observed input composition and K10 scores",
            "sampling": "sample n_s complete meetings with replacement within each original partition; hold partition proportions n_s/N fixed",
            "meeting_contents": "all original input rows and all K10 paired scores retained in every selected meeting",
            "pairing": "same meeting plan for every model, metric, and scoring policy",
            "assumptions": ["meetings are exchangeable independent clusters within each original partition",
                            "original partition proportions are fixed",
                            "model weights, prompts, references, scoring configuration, and K are fixed"],
            "uncertainty": "resampling observed meeting means includes their finite-K variability; there is no added nested generation resample",
            "excludes": ["model-fitting uncertainty", "serial dependence between meetings", "claims of external generalization"],
        },
        "generation_sensitivity": {
            "scheme": "fixed_corpus_paired_generation_sensitivity", "draws": draws, "seed": seed + 1,
            "estimand": "equal-meeting mean similarity conditional on the observed corpus and fitted models",
            "sampling": "hold every meeting and input row fixed; independently sample K of its K paired generations with replacement for each input row",
            "pairing": "same within-input replicate plan across models, metrics, and scoring policies; fresh plan for every input row",
            "assumptions": ["replicates are exchangeable within each input", "generation noise is independent across input requests conditional on the fixed setup"],
            "scope": "conditional decoding stability; not a combined meeting-plus-generation confidence interval",
        },
        "intervals": {"confidence": .95, "type": "percentile", "quantile_method": "linear",
                      "multiplicity_adjusted": False, "selection_by_significance": False},
        "ordering": {"models": MODELS, "policies": POLICIES, "metrics": METRICS, "splits": SPLITS,
                     "input_rows": "lexicographic sample_id", "meetings": "lexicographic meeting_id",
                     "rng": "numpy.random.default_rng (PCG64)", "integer_dtype": "int64",
                     "main_rng_calls": "sequential split order, shape (draws, n_split)",
                     "generation_rng_calls": "sequential input order, shape (draws, K) for each input",
                     "draw_array_axes": ["policy", "model", "metric", "bootstrap_draw"]},
        "validation": panel.audit,
        "source_files": {name: {"sha256": sha256(source / name), "bytes": (source / name).stat().st_size}
                         for name in (*bindings, "score_manifest.json", "evaluation_manifest.json", "pairwise_contrasts.csv", "model_summary.csv")},
        "implementation": {"filename": Path(__file__).name, "sha256": sha256(Path(__file__)),
                           "python": platform.python_version(), "numpy": np.__version__},
        "artifacts": {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                      for name in ("model_summary.csv", "pairwise_contrasts.csv", "bootstrap_draws.npz")},
    }
    manifest["integrity"] = {"algorithm": "sha256(canonical-json-without-integrity)",
                             "payload_sha256": hashlib.sha256(canonical(manifest)).hexdigest()}
    (output / "reanalysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False,
                                                               indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = run(args.source_dir, args.output_dir, draws=args.draws, seed=args.seed)
    print(json.dumps({"status": manifest["status"], "output": str(args.output_dir.resolve()),
                      "draws": args.draws, "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
