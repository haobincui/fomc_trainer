"""Build fail-closed CHK2 recovery cohorts without mutating the source release.

The recovery policy gives one additional attempt to rows with a tractable
failure profile.  It separates fresh generation from verifier-only retries,
and defers long, multi-family generation failures from an identical blind
retry.  Existing quality gates are not weakened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jobs.generation.generate_chk2_target_derived_minutes import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    OUTPUT_SPLITS,
    PreparedRow,
    TargetDerivedDataError,
    prepare_official_targets,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RELEASE = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_v1_20260828"
)
DEFAULT_PLAN_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_plan_v1_20260829"
)
DEFAULT_REGENERATE_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_regenerate_v1_20260829"
)
DEFAULT_REVERIFY_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_reverify_v1_20260829"
)

SOURCE_COMPLETE_STATUS = "machine_screen_complete_human_review_pending"
SCHEMA_VERSION = "chk2-target-derived-recovery-plan-v1"
SOURCE_SPLIT_BY_OUTPUT = {"train": "train", "validation": "eval", "test": "test"}


GENERATION_FAMILY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("destylization", ("analysis_insufficiently_destylized",)),
    (
        "attribution",
        ("reasoning_attribution_set_mismatch", "analysis_attribution_set_mismatch"),
    ),
    (
        "numeric",
        ("reasoning_numeric_multiset_mismatch", "analysis_numeric_multiset_mismatch"),
    ),
    (
        "date",
        ("reasoning_date_set_mismatch", "analysis_date_set_mismatch"),
    ),
    (
        "evidence_coverage",
        (
            "target_sentence_uncovered",
            "analysis_evidence_not_exact",
            "target_span_not_exact",
        ),
    ),
    ("meta_contamination", ("reasoning_transport_meta",)),
    (
        "structure_schema",
        (
            "nonconsecutive_claim_id",
            "content_not_strict_json",
            "atomic_claim_schema",
            "generation_content_keys",
            "serialization_or_tokenization",
            "reasoning_words",
            "fidelity_reasoning_must_be_nonempty_text",
            "atomic_claim_",
            "claim_alignment_",
        ),
    ),
)


class RecoveryPlanError(RuntimeError):
    """The source release cannot be converted into a safe recovery plan."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryPlanError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryPlanError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RecoveryPlanError(f"cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RecoveryPlanError(f"blank JSONL row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryPlanError(
                f"invalid JSONL row {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RecoveryPlanError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(value)
    return rows


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != text:
            raise RecoveryPlanError(f"existing recovery artifact differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _write_json_exact(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_exact(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl_exact(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _write_text_exact(path, text)


def _indexed(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in result:
            raise RecoveryPlanError(f"duplicate/empty sample_id in {label}: {sample_id!r}")
        result[sample_id] = dict(row)
    return result


def generation_reason_families(reasons: Sequence[str]) -> tuple[str, ...]:
    families: list[str] = []
    unknown: list[str] = []
    for reason in reasons:
        matched = [
            family
            for family, prefixes in GENERATION_FAMILY_PREFIXES
            if any(str(reason).startswith(prefix) for prefix in prefixes)
        ]
        if not matched:
            unknown.append(str(reason))
        for family in matched:
            if family not in families:
                families.append(family)
    if unknown:
        raise RecoveryPlanError(f"unmapped generation rejection reasons: {unknown}")
    ordered = [
        family
        for family, _ in GENERATION_FAMILY_PREFIXES
        if family in families
    ]
    return tuple(ordered)


def _semantic_verifier_pass(parsed: object) -> bool:
    if not isinstance(parsed, dict):
        return False
    target_claims = parsed.get("target_claims")
    analysis_claims = parsed.get("analysis_claims")
    issues = parsed.get("issues")
    return (
        isinstance(target_claims, list)
        and bool(target_claims)
        and all(
            isinstance(item, dict) and item.get("verdict") == "entailed"
            for item in target_claims
        )
        and isinstance(analysis_claims, list)
        and bool(analysis_claims)
        and all(
            isinstance(item, dict) and item.get("verdict") == "supported"
            for item in analysis_claims
        )
        and parsed.get("reasoning_compatible") is True
        and parsed.get("bidirectional_entailment") is True
        and isinstance(issues, list)
        and not issues
        and parsed.get("overall_pass") is True
    )


def classify_recovery_action(
    *,
    rejection: Mapping[str, Any],
    prepared: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...], str]:
    stage = str(rejection.get("rejection_stage") or "")
    reasons = tuple(str(item) for item in rejection.get("rejection_reasons") or ())
    word_count = len(str(prepared.get("official_minutes_paragraph") or "").split())
    if stage == "verification_gate":
        parsed = verification.get("parsed_verification") if verification else None
        if parsed is None:
            return (
                "reverify_only",
                ("verification_parse_or_response",),
                "generation already passed; verifier response was absent or unparseable",
            )
        if _semantic_verifier_pass(parsed):
            return (
                "reverify_only",
                ("verification_output_contract",),
                "semantic verdicts were all positive; verifier evidence/schema output failed",
            )
        return (
            "regenerate_and_reverify",
            ("verification_semantic_rejection",),
            "the generated candidate failed semantic equivalence verification",
        )
    if stage != "generation_gate":
        raise RecoveryPlanError(f"unexpected rejection stage: {stage!r}")
    families = generation_reason_families(reasons)
    if not families:
        raise RecoveryPlanError(
            f"generation rejection has no mapped family: {rejection.get('sample_id')}"
        )
    if len(families) == 1 or (len(families) == 2 and word_count <= 200):
        return (
            "regenerate_and_reverify",
            families,
            "one additional fresh response is proportionate for this failure breadth and length",
        )
    return (
        "defer_targeted_repair",
        families,
        "identical blind retry is low-yield for long or three-plus-family failures",
    )


def _load_release_rows(
    source_release: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    set[str],
]:
    prepared_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    pass_ids: set[str] = set()
    for split in OUTPUT_SPLITS:
        prepared_rows.extend(_read_jsonl(source_release / "prepared" / f"{split}.jsonl"))
        rejection_rows.extend(
            _read_jsonl(source_release / "machine_rejections" / f"{split}.jsonl")
        )
        verification_rows.extend(
            _read_jsonl(source_release / "teacher_verifications" / f"{split}.jsonl")
        )
        for row in _read_jsonl(source_release / "manifests" / f"{split}.jsonl"):
            pass_ids.add(str(row.get("sample_id") or ""))
    return (
        _indexed(prepared_rows, label="prepared"),
        _indexed(rejection_rows, label="machine_rejections"),
        _indexed(verification_rows, label="teacher_verifications"),
        pass_ids,
    )


def _selection_records(
    *,
    source_release: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _read_json(source_release / "summary.json")
    if summary.get("status") != SOURCE_COMPLETE_STATUS:
        raise RecoveryPlanError(
            f"source status is not complete: {summary.get('status')!r}"
        )
    if summary.get("phase") != "verify" or summary.get("failure_count") != 0:
        raise RecoveryPlanError("source verification is incomplete or has final failures")
    prepared, rejections, verifications, pass_ids = _load_release_rows(source_release)
    if set(rejections) & pass_ids:
        raise RecoveryPlanError("machine rejection and pass ledgers overlap")
    if set(prepared) != set(rejections) | pass_ids:
        raise RecoveryPlanError("prepared rows do not reconcile to pass + rejection")

    records: list[dict[str, Any]] = []
    for sample_id, rejection in sorted(rejections.items()):
        prepared_row = prepared[sample_id]
        if rejection.get("split") != prepared_row.get("split"):
            raise RecoveryPlanError(f"split mismatch for {sample_id}")
        if rejection.get("official_minutes_sha256") != prepared_row.get(
            "official_minutes_sha256"
        ):
            raise RecoveryPlanError(f"official target hash mismatch for {sample_id}")
        verification = verifications.get(sample_id)
        action, families, rationale = classify_recovery_action(
            rejection=rejection,
            prepared=prepared_row,
            verification=verification,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": prepared_row["split"],
                "meeting_date": prepared_row["meeting_date"],
                "section_name": prepared_row["section_name"],
                "section_category": prepared_row["section_category"],
                "official_minutes_sha256": prepared_row["official_minutes_sha256"],
                "official_word_count": len(
                    str(prepared_row["official_minutes_paragraph"]).split()
                ),
                "original_rejection_stage": rejection["rejection_stage"],
                "original_rejection_reasons": list(
                    rejection.get("rejection_reasons") or []
                ),
                "failure_families": list(families),
                "recovery_action": action,
                "recovery_rationale": rationale,
                "retry_ordinal": 1,
                "maximum_blind_retries": 1,
                "legacy_sample_ids": list(prepared_row.get("legacy_sample_ids") or []),
            }
        )

    action_counts = Counter(row["recovery_action"] for row in records)
    stage_counts = Counter(row["original_rejection_stage"] for row in records)
    split_action_counts: dict[str, dict[str, int]] = {}
    for split in OUTPUT_SPLITS:
        split_rows = [row for row in records if row["split"] == split]
        split_action_counts[split] = dict(
            sorted(Counter(row["recovery_action"] for row in split_rows).items())
        )
    family_counts = Counter(
        family for row in records for family in row["failure_families"]
    )
    plan_summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "recovery_inputs_validated",
        "source_release": {
            "path": str(source_release.relative_to(REPO_ROOT)),
            "summary_sha256": _sha256_file(source_release / "summary.json"),
        },
        "policy": {
            "quality_gates_unchanged": True,
            "preparation_exclusions_not_retried": True,
            "blind_retry_cap": 1,
            "fresh_generation_rule": (
                "verification semantic rejection; or generation failure with one "
                "family; or two families and official paragraph <=200 words"
            ),
            "reverify_only_rule": (
                "generation passed and verifier semantics are all positive but the "
                "verifier output contract failed, or verifier response is missing/unparseable"
            ),
            "defer_rule": (
                "generation failure with >=3 families, or two families and official "
                "paragraph >200 words"
            ),
        },
        "source_machine_rejections": len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "original_stage_counts": dict(sorted(stage_counts.items())),
        "split_action_counts": split_action_counts,
        "nonexclusive_failure_family_counts": dict(sorted(family_counts.items())),
        "selected_for_immediate_recovery": (
            action_counts["regenerate_and_reverify"] + action_counts["reverify_only"]
        ),
        "deferred_for_targeted_repair": action_counts["defer_targeted_repair"],
    }
    expected_actions = {
        "regenerate_and_reverify": 1_531,
        "reverify_only": 53,
        "defer_targeted_repair": 126,
    }
    if dict(action_counts) != expected_actions:
        raise RecoveryPlanError(
            f"recovery cohort drift: observed={dict(action_counts)} expected={expected_actions}"
        )
    return records, plan_summary


def _load_legacy_sources(source_root: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split in ("train", "eval", "test"):
        for row in _read_jsonl(source_root / f"{source_split}_manifest.jsonl"):
            rows.append(row)
    return _indexed(rows, label="legacy source manifests")


def _write_subset_sources(
    *,
    selected: Sequence[Mapping[str, Any]],
    source_root: Path,
    destination: Path,
) -> dict[str, Any]:
    legacy_by_id = _load_legacy_sources(source_root)
    desired_ids = {
        str(legacy_id)
        for row in selected
        for legacy_id in row.get("legacy_sample_ids") or []
    }
    if not desired_ids:
        raise RecoveryPlanError("recovery cohort contains no legacy source ids")
    missing = sorted(desired_ids - set(legacy_by_id))
    if missing:
        raise RecoveryPlanError(f"legacy source ids are missing: {missing[:3]}")

    file_records: dict[str, Any] = {}
    selected_sample_ids = {str(row["sample_id"]) for row in selected}
    for output_split in OUTPUT_SPLITS:
        source_split = SOURCE_SPLIT_BY_OUTPUT[output_split]
        rows = [
            legacy_by_id[legacy_id]
            for legacy_id in sorted(desired_ids)
            if str(legacy_by_id[legacy_id].get("split") or "") == source_split
        ]
        path = destination / f"{source_split}_manifest.jsonl"
        _write_jsonl_exact(path, rows)
        file_records[source_split] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "rows": len(rows),
            "sha256": _sha256_file(path),
        }

    with tempfile.TemporaryDirectory(prefix="chk2-recovery-validate-") as temp_dir:
        try:
            prepared, preparation = prepare_official_targets(
                source_root=destination,
                split_manifest_path=DEFAULT_SPLIT_MANIFEST,
                output_root=Path(temp_dir) / "prepared",
                selected_splits=OUTPUT_SPLITS,
            )
        except TargetDerivedDataError as exc:
            raise RecoveryPlanError(f"subset re-preparation failed: {exc}") from exc
        rebuilt = {
            row.sample_id: row
            for split_rows in prepared.values()
            for row in split_rows
        }
    if set(rebuilt) != selected_sample_ids:
        missing_ids = sorted(selected_sample_ids - set(rebuilt))
        extra_ids = sorted(set(rebuilt) - selected_sample_ids)
        raise RecoveryPlanError(
            f"subset sample ids do not round-trip: missing={missing_ids[:3]} "
            f"extra={extra_ids[:3]}"
        )
    expected_by_id = {str(row["sample_id"]): row for row in selected}
    for sample_id, rebuilt_row in rebuilt.items():
        expected = expected_by_id[sample_id]
        if (
            rebuilt_row.split != expected["split"]
            or rebuilt_row.official_minutes_sha256
            != expected["official_minutes_sha256"]
        ):
            raise RecoveryPlanError(f"subset row binding changed: {sample_id}")
    return {
        "source_files": file_records,
        "selected_sample_ids": len(selected_sample_ids),
        "rebuilt_sample_ids": len(rebuilt),
        "round_trip_exact": True,
        "preparation": preparation,
    }


def _copy_reverify_generation_caches(
    *,
    selected: Sequence[Mapping[str, Any]],
    source_release: Path,
    destination_root: Path,
) -> dict[str, Any]:
    destination = destination_root / "cache" / "generation"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        sample_id = str(row["sample_id"])
        name = f"{sha256_text(sample_id)}.json"
        source_path = source_release / "cache" / "generation" / name
        target_path = destination / name
        if not source_path.is_file():
            raise RecoveryPlanError(f"generation cache missing for {sample_id}")
        source_sha = _sha256_file(source_path)
        if target_path.exists():
            if _sha256_file(target_path) != source_sha:
                raise RecoveryPlanError(f"reverify cache collision: {target_path}")
        else:
            shutil.copy2(source_path, target_path)
        records.append(
            {
                "sample_id": sample_id,
                "source_sha256": source_sha,
                "destination": str(target_path.relative_to(REPO_ROOT)),
            }
        )
    return {
        "cache_count": len(records),
        "copy_mode": "isolated_copy_preserving_source_release",
        "records_sha256": hashlib.sha256(
            _canonical_json(records).encode("utf-8")
        ).hexdigest(),
    }


def build_recovery_plan(
    *,
    source_release: Path,
    source_root: Path,
    plan_root: Path,
    regenerate_root: Path,
    reverify_root: Path,
) -> dict[str, Any]:
    source_release = source_release.resolve()
    source_root = source_root.resolve()
    plan_root = plan_root.resolve()
    regenerate_root = regenerate_root.resolve()
    reverify_root = reverify_root.resolve()
    if len({source_release, plan_root, regenerate_root, reverify_root}) != 4:
        raise RecoveryPlanError("source, plan, regenerate, and reverify roots must differ")

    records, summary = _selection_records(source_release=source_release)
    by_action = {
        action: [row for row in records if row["recovery_action"] == action]
        for action in (
            "regenerate_and_reverify",
            "reverify_only",
            "defer_targeted_repair",
        )
    }

    _write_jsonl_exact(plan_root / "selection" / "all.jsonl", records)
    for action, action_rows in by_action.items():
        _write_jsonl_exact(plan_root / "selection" / f"{action}.jsonl", action_rows)

    regenerate_input = regenerate_root / "inputs" / "minutes_alignment"
    reverify_input = reverify_root / "inputs" / "minutes_alignment"
    regenerate_validation = _write_subset_sources(
        selected=by_action["regenerate_and_reverify"],
        source_root=source_root,
        destination=regenerate_input,
    )
    reverify_validation = _write_subset_sources(
        selected=by_action["reverify_only"],
        source_root=source_root,
        destination=reverify_input,
    )
    reverify_cache_copy = _copy_reverify_generation_caches(
        selected=by_action["reverify_only"],
        source_release=source_release,
        destination_root=reverify_root,
    )

    for root, action, validation in (
        (
            regenerate_root,
            "regenerate_and_reverify",
            regenerate_validation,
        ),
        (reverify_root, "reverify_only", reverify_validation),
    ):
        action_rows = by_action[action]
        _write_jsonl_exact(root / "recovery_selection.jsonl", action_rows)
        _write_json_exact(
            root / "recovery_input_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_release": summary["source_release"],
                "recovery_action": action,
                "selected_rows": len(action_rows),
                "selection_sha256": _sha256_file(root / "recovery_selection.jsonl"),
                "subset_validation": validation,
                "quality_gates_unchanged": True,
            },
        )

    summary.update(
        {
            "artifacts": {
                "selection_all": {
                    "path": str((plan_root / "selection" / "all.jsonl").relative_to(REPO_ROOT)),
                    "sha256": _sha256_file(plan_root / "selection" / "all.jsonl"),
                },
                "regenerate_root": str(regenerate_root.relative_to(REPO_ROOT)),
                "reverify_root": str(reverify_root.relative_to(REPO_ROOT)),
            },
            "subset_validation": {
                "regenerate": regenerate_validation,
                "reverify": reverify_validation,
            },
            "reverify_generation_cache_copy": reverify_cache_copy,
        }
    )
    _write_json_exact(plan_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, default=DEFAULT_SOURCE_RELEASE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--regenerate-root", type=Path, default=DEFAULT_REGENERATE_ROOT)
    parser.add_argument("--reverify-root", type=Path, default=DEFAULT_REVERIFY_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_recovery_plan(
        source_release=args.source_release,
        source_root=args.source_root,
        plan_root=args.plan_root,
        regenerate_root=args.regenerate_root,
        reverify_root=args.reverify_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryPlanError as exc:
        print(f"CHK2 recovery planning failed: {exc}")
        raise SystemExit(2) from exc
