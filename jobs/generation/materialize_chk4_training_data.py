"""Materialize audited chk4 Decision-SFT/GRPO datasets from teacher output.

This is a deterministic, offline step and has no chk2 or API dependency.  It
keeps one unique manifest row per meeting, applies direction balancing only to
physical train rows, and emits a separate repeat manifest that maps every
physical row back to its opaque source sample ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation.generate_chk4_sft_targets import (
    MANIFEST_SCHEMA,
    SPLITS,
    STUDENT_SYSTEM_PROMPT,
    Chk4TargetError,
    canonical_json,
    sha256_file,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_ROOT = REPO_ROOT / "output/data/retrain_v2/chk4/deepseek_v4_pro_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/data/retrain_v2/chk4/training_data_v1"
OUTPUT_SCHEMA = "chk4-decision-training-data-v1"
REPEAT_SCHEMA = "chk4-decision-repeat-manifest-v1"
TRAINING_ROW_RE = re.compile(r"^row-[0-9a-f]{24}$")
FORBIDDEN_PROMPT_FRAGMENTS = (
    "current_rate",
    "rate_change",
    "gold label",
    "teacher target",
    "teacher-only canonical decision",
    "actual committee action",
)


class MaterializationError(Chk4TargetError):
    """Teacher outputs cannot be safely materialized for training."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MaterializationError(f"JSON root is not an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MaterializationError(f"required input is missing: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise MaterializationError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            result.append(payload)
    return result


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, "".join(canonical_json(row) + "\n" for row in rows))


def _gold_text(gold: Mapping[str, Any]) -> str:
    if set(gold) != {"direction", "magnitude_bp"}:
        raise MaterializationError("manifest gold has wrong keys")
    direction = gold.get("direction")
    magnitude = gold.get("magnitude_bp")
    if direction not in {"cut", "hold", "hike"}:
        raise MaterializationError(f"invalid manifest direction: {direction!r}")
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    if isinstance(magnitude, bool) or not isinstance(magnitude, int) or magnitude not in allowed:
        raise MaterializationError(f"invalid manifest magnitude: {magnitude!r}")
    return json.dumps(
        {"direction": direction, "magnitude_bp": magnitude},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_prompt(prompt: str, meeting_date: str) -> None:
    lowered = prompt.lower()
    for fragment in FORBIDDEN_PROMPT_FRAGMENTS:
        if fragment in lowered:
            raise MaterializationError(f"prompt leakage fragment: {fragment}")
    if meeting_date and meeting_date in prompt:
        raise MaterializationError("prompt leaks meeting_date")
    if "\"analysis\"" not in prompt:
        raise MaterializationError("prompt does not contain the structured analysis field")


def load_unique_rows(teacher_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    all_sample_ids: set[str] = set()
    all_meetings: set[str] = set()
    for split in SPLITS:
        manifests = _load_jsonl(teacher_root / "manifests" / f"{split}.jsonl")
        sft = _load_jsonl(teacher_root / "sft" / f"{split}.jsonl")
        if len(manifests) != len(sft):
            raise MaterializationError(
                f"{split} manifest/SFT row mismatch: {len(manifests)} != {len(sft)}"
            )
        for index, (manifest, training) in enumerate(zip(manifests, sft, strict=True)):
            if manifest.get("schema_version") != MANIFEST_SCHEMA:
                raise MaterializationError(f"unsupported manifest schema in {split}:{index}")
            if set(training) != {"prompt", "response"}:
                raise MaterializationError(f"SFT row has extra or missing keys in {split}:{index}")
            sample_id = str(manifest.get("sample_id") or "")
            meeting_date = str(manifest.get("meeting_date") or "")
            if not sample_id or sample_id in all_sample_ids:
                raise MaterializationError(f"duplicate/empty sample_id: {sample_id!r}")
            if not meeting_date or meeting_date in all_meetings:
                raise MaterializationError(f"duplicate/empty meeting_date: {meeting_date!r}")
            if manifest.get("split") != split:
                raise MaterializationError(f"manifest split mismatch: {sample_id}")
            population_role = manifest.get("population_role")
            if population_role == "supplement" and split != "train":
                raise MaterializationError("supplement leaked outside train")
            if population_role not in {"core", "supplement"}:
                raise MaterializationError(f"invalid population role: {sample_id}")
            prompt = str(training.get("prompt") or "")
            response = str(training.get("response") or "")
            _validate_prompt(prompt, meeting_date)
            gold = manifest.get("gold")
            if not isinstance(gold, dict):
                raise MaterializationError(f"manifest gold missing: {sample_id}")
            gold_text = _gold_text(gold)
            if response.count("</think>") != 1 or not response.endswith(
                "</think>\n" + gold_text
            ):
                raise MaterializationError(f"SFT response/gold mismatch: {sample_id}")
            reasoning = response.split("</think>", 1)[0].strip()
            if not reasoning:
                raise MaterializationError(f"SFT reasoning is empty: {sample_id}")
            if manifest.get("prompt_sha256") != sha256_text(prompt):
                raise MaterializationError(f"prompt SHA mismatch: {sample_id}")
            if manifest.get("gold_sha256") != sha256_text(gold_text):
                raise MaterializationError(f"gold SHA mismatch: {sample_id}")
            result[split].append(
                {
                    "sample_id": sample_id,
                    "meeting_date": meeting_date,
                    "population": manifest.get("population"),
                    "population_role": population_role,
                    "source_ids": manifest.get("source_ids"),
                    "prompt": prompt,
                    "response": response,
                    "direction": gold["direction"],
                    "magnitude_bp": gold["magnitude_bp"],
                    "prompt_sha256": manifest["prompt_sha256"],
                    "response_sha256": sha256_text(response),
                    "gold_sha256": manifest["gold_sha256"],
                }
            )
            all_sample_ids.add(sample_id)
            all_meetings.add(meeting_date)
    return result


def direction_repeat_factors(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["direction"]) for row in train_rows)
    hold_count = counts.get("hold", 0)
    if hold_count <= 0 or counts.get("cut", 0) <= 0 or counts.get("hike", 0) <= 0:
        raise MaterializationError(f"train is missing an action direction: {dict(counts)}")
    return {
        "hold": 1,
        "cut": min(4, math.ceil(hold_count / counts["cut"])),
        "hike": min(4, math.ceil(hold_count / counts["hike"])),
    }


def _training_row_id(stage: str, sample_id: str, repeat_index: int) -> str:
    digest = sha256_text(f"{OUTPUT_SCHEMA}\0{stage}\0{sample_id}\0{repeat_index}")[:24]
    value = f"row-{digest}"
    assert TRAINING_ROW_RE.fullmatch(value)
    return value


def materialize(
    *, teacher_root: Path, output_root: Path, allow_existing: bool = False
) -> dict[str, Any]:
    summary_path = output_root / "summary.json"
    if output_root.exists() and not allow_existing:
        raise MaterializationError(
            f"output exists; overwrite is forbidden: {output_root}"
        )
    if allow_existing:
        if not summary_path.is_file():
            raise MaterializationError("existing output has no summary")
        summary = _load_json(summary_path)
        if summary.get("teacher_root_sha256") != _tree_sha256(teacher_root):
            raise MaterializationError("existing output teacher binding drift")
        return summary
    teacher_summary = _load_json(teacher_root / "summary.json")
    if teacher_summary.get("status") != "complete":
        raise MaterializationError("teacher acquisition is not complete")
    rows = load_unique_rows(teacher_root)
    factors = direction_repeat_factors(rows["train"])
    output_root.mkdir(parents=True, exist_ok=False)
    unique_counts = {split: len(rows[split]) for split in SPLITS}
    physical_counts: dict[str, dict[str, int]] = {
        "decision_sft": {},
        "decision_grpo": {},
    }
    for split in SPLITS:
        repeat_manifest: list[dict[str, Any]] = []
        sft_rows: list[dict[str, Any]] = []
        grpo_rows: list[dict[str, Any]] = []
        unique_manifest: list[dict[str, Any]] = []
        for source in sorted(rows[split], key=lambda row: str(row["sample_id"])):
            factor = factors[str(source["direction"])] if split == "train" else 1
            unique_manifest.append(
                {
                    key: source[key]
                    for key in (
                        "sample_id",
                        "meeting_date",
                        "population",
                        "population_role",
                        "source_ids",
                        "direction",
                        "magnitude_bp",
                        "prompt_sha256",
                        "response_sha256",
                        "gold_sha256",
                    )
                }
                | {"split": split, "repeat_factor": factor}
            )
            for repeat_index in range(factor):
                sft_row_id = _training_row_id(
                    "decision_sft", str(source["sample_id"]), repeat_index
                )
                grpo_row_id = _training_row_id(
                    "decision_grpo", str(source["sample_id"]), repeat_index
                )
                sft_rows.append(
                    {"prompt": source["prompt"], "response": source["response"]}
                )
                grpo_rows.append(
                    {
                        "sample_id": source["sample_id"],
                        "prompt": source["prompt"],
                        "direction": source["direction"],
                        "magnitude_bp": source["magnitude_bp"],
                    }
                )
                repeat_manifest.extend(
                    (
                        {
                            "schema_version": REPEAT_SCHEMA,
                            "stage": "decision_sft",
                            "training_row_id": sft_row_id,
                            "source_sample_id": source["sample_id"],
                            "split": split,
                            "repeat_index": repeat_index,
                            "repeat_factor": factor,
                        },
                        {
                            "schema_version": REPEAT_SCHEMA,
                            "stage": "decision_grpo",
                            "training_row_id": grpo_row_id,
                            "source_sample_id": source["sample_id"],
                            "split": split,
                            "repeat_index": repeat_index,
                            "repeat_factor": factor,
                        },
                    )
                )
        _write_jsonl(output_root / "decision_sft" / f"{split}.jsonl", sft_rows)
        _write_jsonl(output_root / "decision_grpo" / f"{split}.jsonl", grpo_rows)
        _write_jsonl(output_root / "manifests" / "unique" / f"{split}.jsonl", unique_manifest)
        _write_jsonl(output_root / "manifests" / "repeats" / f"{split}.jsonl", repeat_manifest)
        physical_counts["decision_sft"][split] = len(sft_rows)
        physical_counts["decision_grpo"][split] = len(grpo_rows)
    train_unique_direction = Counter(row["direction"] for row in rows["train"])
    train_physical_direction = Counter()
    for direction, count in train_unique_direction.items():
        train_physical_direction[direction] = count * factors[direction]
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "complete",
        "teacher_root": str(teacher_root),
        "teacher_root_sha256": _tree_sha256(teacher_root),
        "teacher_contract_sha256": teacher_summary.get("contract_sha256"),
        "system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
        "unique_counts": unique_counts,
        "physical_counts": physical_counts,
        "repeat_factors": factors,
        "train_unique_direction_counts": dict(train_unique_direction),
        "train_physical_direction_counts": dict(train_physical_direction),
        "test_is_sealed_evaluation_only": True,
    }
    _write_json(summary_path, summary)
    return summary


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise MaterializationError(f"teacher root is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify the immutable teacher binding instead of rebuilding.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize(
            teacher_root=args.teacher_root.resolve(),
            output_root=args.output_root.resolve(),
            allow_existing=args.verify_existing,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (MaterializationError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False),
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
