"""Score the current chk0/chk1/chk2 chain on the frozen Chapter 2 test.

This entry point deliberately uses the generic evaluator contract because the
historical formal evaluator is sealed to a different four-artifact archive.
It still validates the frozen test, checkpoint, generation, and semantic-model
manifests before running the same row metrics, meeting-level aggregation,
bootstrap, and parent-child contrasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.eval.eval_checkpoint_generation import (
    BERTScoreBackend,
    MPNetCosineBackend,
    SCORING_POLICIES,
    STRICT_SCORING_POLICY,
    run_checkpoint_generation_evaluation,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


ARTIFACT_IDS = (
    "eval-chk0-base",
    "eval-chk1-compressed-sft",
    "eval-chk2-reward-v3-cp150",
)


def _read_json(path: Path, *, label: str, sealed: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if sealed:
        validate_manifest_integrity(value)
    return value


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _resolve_local_path(value: object, *, root: Path, label: str) -> Path:
    raw = Path(str(value or "")).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not resolve to a directory: {value}")
    return resolved


def validate_inputs(
    *,
    generations: Sequence[Path],
    generation_manifests: Sequence[Path],
    prompts: Path,
    references: Path,
    config: Path,
    test_manifest: Path,
    checkpoint_manifest: Path,
    semantic_manifest: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if len(generations) != len(ARTIFACT_IDS):
        raise ValueError(f"Exactly {len(ARTIFACT_IDS)} generation files are required")
    if len(generation_manifests) != len(ARTIFACT_IDS):
        raise ValueError(
            f"Exactly {len(ARTIFACT_IDS)} generation manifests are required"
        )

    checkpoint = _read_json(
        checkpoint_manifest, label="checkpoint manifest", sealed=True
    )
    checkpoint_ids = {
        str(row.get("artifact_id") or "")
        for row in checkpoint.get("artifacts", [])
        if isinstance(row, Mapping)
    }
    if checkpoint_ids != set(ARTIFACT_IDS):
        raise ValueError(
            "Checkpoint manifest artifact inventory differs from the current chain: "
            f"{sorted(checkpoint_ids)}"
        )

    test = _read_json(test_manifest, label="test manifest", sealed=True)
    expected_outputs = {
        "prompts": (prompts, 33),
        "references": (references, 33),
    }
    for name, (path, expected_rows) in expected_outputs.items():
        record = test.get("outputs", {}).get(name, {})
        observed_sha = sha256_file(path)
        observed_rows = _row_count(path)
        if record.get("sha256") != observed_sha or record.get("row_count") != observed_rows:
            raise ValueError(f"Frozen {name} no longer matches the test manifest")
        if observed_rows != expected_rows:
            raise ValueError(f"Frozen {name} must contain {expected_rows} rows")

    config_payload = _read_json(config, label="evaluation config")
    if config_payload.get("evaluation_id") != test.get("evaluation_id"):
        raise ValueError("Evaluation config and frozen test manifest IDs differ")

    checkpoint_sha = sha256_file(checkpoint_manifest)
    prompt_sha = sha256_file(prompts)
    seen_ids: set[str] = set()
    for generation, manifest_path in zip(
        generations, generation_manifests, strict=True
    ):
        manifest = _read_json(
            manifest_path, label="generation manifest", sealed=True
        )
        artifact_id = str(manifest.get("artifact_id") or "")
        if artifact_id not in ARTIFACT_IDS or artifact_id in seen_ids:
            raise ValueError(f"Unexpected or duplicate generation artifact: {artifact_id}")
        seen_ids.add(artifact_id)
        output = manifest.get("output", {})
        if output.get("sha256") != sha256_file(generation):
            raise ValueError(f"{artifact_id}: generation SHA differs from its manifest")
        if output.get("row_count") != _row_count(generation):
            raise ValueError(f"{artifact_id}: generation row count differs from manifest")
        if manifest.get("sample_count") != 33 or output.get("row_count") != 33:
            raise ValueError(f"{artifact_id}: generation must contain exactly 33 rows")
        if manifest.get("checkpoint_manifest", {}).get("sha256") != checkpoint_sha:
            raise ValueError(f"{artifact_id}: checkpoint-manifest binding changed")
        if manifest.get("prompts", {}).get("sha256") != prompt_sha:
            raise ValueError(f"{artifact_id}: prompt binding changed")
    if seen_ids != set(ARTIFACT_IDS):
        raise ValueError(f"Generation inventory is incomplete: {sorted(seen_ids)}")

    semantic = _read_json(
        semantic_manifest, label="semantic model manifest", sealed=True
    )
    return config_payload, test, semantic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", action="append", required=True, type=Path)
    parser.add_argument(
        "--generation-manifest", action="append", required=True, type=Path
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--lineage-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--semantic-device", default="cuda:0")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument(
        "--scoring-policy", choices=SCORING_POLICIES, default=STRICT_SCORING_POLICY
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    config, test, semantic = validate_inputs(
        generations=args.generations,
        generation_manifests=args.generation_manifest,
        prompts=args.prompts,
        references=args.references,
        config=args.config,
        test_manifest=args.test_manifest,
        checkpoint_manifest=args.checkpoint_manifest,
        semantic_manifest=args.semantic_manifest,
        repo_root=root,
    )

    models = semantic.get("models", {})
    bert = models.get("bertscore", {})
    mpnet = models.get("embedding_cosine", {})
    bert_path = _resolve_local_path(
        bert.get("local_path"), root=root, label="BERTScore model"
    )
    mpnet_path = _resolve_local_path(
        mpnet.get("local_path"), root=root, label="MPNet model"
    )
    scorers = [
        BERTScoreBackend(
            bert_path,
            str(bert.get("directory_sha256") or ""),
            num_layers=int(bert.get("num_layers")),
            batch_size=args.semantic_batch_size,
            device=args.semantic_device,
        ),
        MPNetCosineBackend(
            mpnet_path,
            str(mpnet.get("directory_sha256") or ""),
            batch_size=args.semantic_batch_size,
            device=args.semantic_device,
        ),
    ]
    subsets = {
        "all_11_meetings": None,
        "prospective_only_9_meetings": test.get(
            "prospective_only_meeting_dates", []
        ),
    }
    result = run_checkpoint_generation_evaluation(
        args.generations,
        args.references,
        args.prompts,
        args.output_dir,
        require_formal_manifests=False,
        semantic_scorers=scorers,
        lineage_manifest_path=args.lineage_manifest,
        expected_artifact_count=3,
        required_artifact_ids=ARTIFACT_IDS,
        missing_policy="error",
        require_provenance=True,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        scoring_policy=args.scoring_policy,
        evaluation_subsets=subsets,
    )
    print(
        json.dumps(
            {
                "status": result["audit"]["status"],
                "scoring_policy": args.scoring_policy,
                "reused_existing": result["reused_existing"],
                "paths": result["paths"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
