"""Pinned CLI for dry-running and generating canonical retrain-v2 chk1 targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .generation_pipeline import (
    PILOT_SAMPLE_COUNT,
    SMOKE_SAMPLE_COUNT,
    run_generation_pipeline,
)
from .pipeline import build_minutes_resolver


PREPARE_HANDOFF = Path(
    "output/data/retrain_v2/chk1/prepared_sparse_v3/prepare_handoff.json"
)
STUDENT_PROMPTS = Path(
    "dataset/processed/pipeline/analysis_sft/student_prompts/after_2009"
)
MINUTES_REFERENCES = Path(
    "dataset/processed/pipeline/minutes_alignment/prompts/after_2009"
)
OUTPUTS = {
    "smoke": Path("output/data/retrain_v2/chk1/generation_smoke_v7"),
    "pilot": Path("output/data/retrain_v2/chk1/generation_pilot_v7"),
    "full": Path("output/data/retrain_v2/chk1/generation_full_v7"),
}
MODEL_CACHE = Path("output/data/retrain_v2/chk1/deepseek_teacher_cache_v5")
EXPECTED_COUNTS = {
    "smoke": SMOKE_SAMPLE_COUNT,
    "pilot": PILOT_SAMPLE_COUNT,
    "full": 2117,
}
class Chk1WorkflowError(RuntimeError):
    """Raised when a chk1 acquisition phase is structurally incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Chk1WorkflowError(message)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Chk1WorkflowError(f"unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(line.strip() != "", f"{label} has a blank row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Chk1WorkflowError(
                    f"{label} has invalid JSON at line {line_number}"
                ) from exc
            _require(isinstance(row, dict), f"{label} row must be an object")
            rows.append(row)
    return rows


def assess_completed_generation(output_dir: str | Path, *, mode: str) -> dict[str, Any]:
    """Reconcile a completed acquisition handoff without applying quality gates."""

    _require(mode in OUTPUTS, f"unsupported chk1 generation mode: {mode}")
    output = Path(output_dir).resolve()
    handoff = _load_json(output / "generation_handoff.json", label="generation handoff")
    selection = _load_json(output / "selection_manifest.json", label="selection manifest")
    _require(handoff.get("status") == "complete", "generation handoff is incomplete")
    expected_count = EXPECTED_COUNTS[mode]
    _require(handoff.get("mode") == mode, "generation handoff mode drifted")
    _require(
        handoff.get("selected_count") == expected_count
        and selection.get("selected_count") == expected_count,
        f"{mode} generation must bind exactly {expected_count} selected rows",
    )
    selected_ids = selection.get("selected_sample_ids")
    _require(
        isinstance(selected_ids, list)
        and len(selected_ids) == expected_count
        and len(set(selected_ids)) == expected_count,
        "generation selection IDs are invalid",
    )
    selected_topics = selection.get("selected_by_topic")
    _require(
        isinstance(selected_topics, dict)
        and selected_topics
        and sum(selected_topics.values()) == expected_count
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in selected_topics.values()
        ),
        "generation selection topic counts are invalid",
    )
    manifests: list[dict[str, Any]] = []
    for split in ("train", "eval", "test"):
        manifests.extend(
            _load_jsonl(
                output / "manifests" / f"{split}.jsonl",
                label=f"{split} generation manifests",
            )
        )
    exclusions = _load_jsonl(output / "audit/exclusions.jsonl", label="exclusions")
    terminal = manifests + exclusions
    terminal_ids = [row.get("sample_id") for row in terminal]
    _require(
        len(terminal_ids) == expected_count
        and len(set(terminal_ids)) == expected_count
        and set(terminal_ids) == set(selected_ids),
        "generation outputs do not cover the exact selection once",
    )
    accepted_count = len(manifests)
    _require(
        handoff.get("accepted_count") == accepted_count
        and handoff.get("excluded_count") == len(exclusions),
        "generation handoff counts do not reconcile",
    )
    return {
        "status": "complete",
        "mode": mode,
        "selected_count": expected_count,
        "accepted_count": accepted_count,
        "excluded_count": len(exclusions),
        "retrieval_rate": accepted_count / expected_count,
        "generation_provenance_sha256": handoff.get("generation_provenance", {}).get(
            "payload_sha256"
        ),
        "handoff_payload_sha256": handoff.get("payload_sha256"),
    }


def _paths(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    return {
        "root": root,
        "prepare_handoff": root / PREPARE_HANDOFF,
        "student_prompts": root / STUDENT_PROMPTS,
        "minutes_references": root / MINUTES_REFERENCES,
        "outputs": {mode: root / path for mode, path in OUTPUTS.items()},
        "model_cache": root / MODEL_CACHE,
    }


def _validate_prior_phase(paths: Mapping[str, Any], mode: str) -> None:
    prior = {"pilot": "smoke", "full": "pilot"}.get(mode)
    if prior is None:
        return
    assessment = assess_completed_generation(paths["outputs"][prior], mode=prior)
    _require(
        assessment["status"] == "complete",
        f"{mode} generation is blocked until {prior} acquisition is complete",
    )


def _progress(event: Mapping[str, Any]) -> None:
    print(json.dumps(dict(event), ensure_ascii=False, sort_keys=True), flush=True)


def dry_run(repo_root: str | Path) -> dict[str, Any]:
    paths = _paths(Path(repo_root))
    result = run_generation_pipeline(
        prepare_handoff_path=paths["prepare_handoff"],
        output_dir=paths["outputs"]["full"],
        repo_root=paths["root"],
        mode="full",
        dry_run=True,
    )
    assert isinstance(result, dict)
    return {
        "status": result["status"],
        "mode": result["mode"],
        "population_count": result["selection"]["population_count"],
        "selected_count": result["selection"]["selected_count"],
        "selected_by_split": result["selection"]["selected_by_split"],
        "planned_prepared_exclusions": result["planned_prepared_exclusions"],
        "generation_provenance_sha256": result["generation_provenance"][
            "payload_sha256"
        ],
        "preparation_binding_sha256": result["preparation_binding"][
            "binding_sha256"
        ],
    }


def generate(repo_root: str | Path, *, mode: str, resume: bool) -> dict[str, Any]:
    paths = _paths(Path(repo_root))
    _require(mode in OUTPUTS, f"unsupported chk1 generation mode: {mode}")
    _validate_prior_phase(paths, mode)
    resolver = build_minutes_resolver(
        student_prompt_dir=paths["student_prompts"],
        minutes_reference_dir=paths["minutes_references"],
    )
    handoff = run_generation_pipeline(
        prepare_handoff_path=paths["prepare_handoff"],
        output_dir=paths["outputs"][mode],
        repo_root=paths["root"],
        mode=mode,
        minutes_resolver=resolver,
        model_cache_dir=paths["model_cache"],
        resume=resume,
        progress_callback=_progress,
    )
    _require(isinstance(handoff, Path), "generation did not publish a handoff")
    assessment = assess_completed_generation(paths["outputs"][mode], mode=mode)
    return {"status": "complete", "handoff": str(handoff), "acquisition": assessment}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    generation = commands.add_parser("generate")
    generation.add_argument("--mode", choices=tuple(OUTPUTS), required=True)
    generation.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--mode", choices=tuple(OUTPUTS), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            result = dry_run(args.repo_root)
        elif args.command == "generate":
            result = generate(args.repo_root, mode=args.mode, resume=args.resume)
        else:
            paths = _paths(args.repo_root)
            result = assess_completed_generation(
                paths["outputs"][args.mode], mode=args.mode
            )
            _require(result["status"] == "complete", f"{args.mode} acquisition is incomplete")
    except Exception as exc:  # noqa: BLE001 - fail-closed command boundary
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
