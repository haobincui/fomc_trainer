"""Reacquire chk3 targets whose released reasoning contains task meta-talk.

The input is an immutable, already-clean chk3 release.  Rows whose reasoning
matches the current meta-language contract are regenerated with
``deepseek-v4-pro``; all other rows are revalidated and reused byte-for-byte.
The source release is never modified.  A new immutable release is published
only after the full 2,072-row population passes the current validator.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk3_sft_targets import (
    DEFAULT_TOKENIZER_PATH,
    EXPECTED_SPLIT_COUNTS,
    SPLITS,
    USER_PROMPT_PREFIX,
    Chk3DataError,
    DeepSeekTeacherResponse,
    OpenAIDeepSeekBackend,
    OutputContractError,
    ProviderIdentityGuard,
    TeacherBackend,
    _load_tokenizer,
    _reasoning_meta_categories,
    canonical_json,
    sha256_file,
    sha256_text,
    validate_teacher_target,
)
from jobs.generation.materialize_chk3_training_data import (
    DEFAULT_ACQUISITION_ROOT,
    DEFAULT_CHK1_HANDOFF,
    DEFAULT_CHK3_CONFIG,
    DEFAULT_RELEASE_PARENT,
    AcquisitionRow,
    CleanRow,
    MaterializationError,
    _load_acquisition,
    _load_json,
    _load_jsonl,
    _prepared_row,
    _publish_release,
    _teacher_response,
    _write_json,
    _write_jsonl,
    regenerate_chk3_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "chk3-reasoning-meta-repair-v1"
DEFAULT_PRIOR_RELEASE = (
    REPO_ROOT
    / "dataset/processed/retrain_v2/chk3_minutes_clean_v1_20260805"
)
DEFAULT_RELEASE_ID = "chk3_minutes_clean_v2_20260805"
DEFAULT_WORK_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk3/training_release_reasoning_meta_v2"
)

_OUTPUT_TO_SOURCE_SPLIT = {"train": "train", "validation": "eval", "test": "test"}


@dataclass(frozen=True)
class PriorReleaseRow:
    sample_id: str
    source_split: str
    source_index: int
    analysis: str
    prompt: str
    reasoning: str
    minutes: str
    response: str
    manifest: Mapping[str, Any]
    teacher_response: DeepSeekTeacherResponse


def _parse_analysis(prompt: str, *, sample_id: str) -> str:
    if not prompt.startswith(USER_PROMPT_PREFIX):
        raise MaterializationError(f"prior prompt contract mismatch: {sample_id}")
    try:
        payload = json.loads(prompt[len(USER_PROMPT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise MaterializationError(
            f"prior prompt JSON is invalid: {sample_id}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"analysis"}:
        raise MaterializationError(f"prior prompt schema mismatch: {sample_id}")
    analysis = payload.get("analysis")
    if not isinstance(analysis, str) or not analysis.strip():
        raise MaterializationError(f"prior prompt analysis is empty: {sample_id}")
    return analysis.strip()


def _released_teacher_response(
    *, reasoning: str, minutes: str, teacher: Mapping[str, Any]
) -> DeepSeekTeacherResponse:
    return _teacher_response(
        {
            "reasoning_content": reasoning,
            "content": canonical_json({"answer": minutes}),
            "answer": minutes,
            "teacher": dict(teacher),
        }
    )


def load_prior_release(release_root: Path) -> dict[str, list[PriorReleaseRow]]:
    handoff = _load_json(release_root / "handoff.json", label="prior chk3 handoff")
    if handoff.get("immutable") is not True or handoff.get("quality_status") != "passed":
        raise MaterializationError("prior chk3 release is not immutable and passed")
    manifest_ref = handoff.get("release_manifest")
    if not isinstance(manifest_ref, Mapping):
        raise MaterializationError("prior chk3 handoff lacks release manifest binding")
    release_manifest_path = release_root / str(manifest_ref.get("path") or "")
    if sha256_file(release_manifest_path) != str(manifest_ref.get("sha256") or ""):
        raise MaterializationError("prior chk3 release manifest hash mismatch")

    result: dict[str, list[PriorReleaseRow]] = {}
    observed_ids: set[str] = set()
    for output_split, source_split in _OUTPUT_TO_SOURCE_SPLIT.items():
        training_rows = _load_jsonl(
            release_root / "minutes_alignment" / f"{output_split}.jsonl",
            label=f"prior chk3 {output_split} rows",
        )
        manifests = _load_jsonl(
            release_root / "minutes_alignment/manifests" / f"{output_split}.jsonl",
            label=f"prior chk3 {output_split} manifests",
        )
        expected = EXPECTED_SPLIT_COUNTS[source_split]
        if len(training_rows) != expected or len(manifests) != expected:
            raise MaterializationError(
                f"prior chk3 split count mismatch: {output_split}"
            )
        rows: list[PriorReleaseRow] = []
        for index, (training, manifest) in enumerate(
            zip(training_rows, manifests, strict=True)
        ):
            if set(training) != {"prompt", "response"}:
                raise MaterializationError(
                    f"prior training schema mismatch: {output_split}[{index}]"
                )
            sample_id = str(manifest.get("sample_id") or "")
            if not sample_id or sample_id in observed_ids:
                raise MaterializationError(f"invalid prior sample ID: {sample_id!r}")
            observed_ids.add(sample_id)
            if manifest.get("source_split") != source_split:
                raise MaterializationError(f"prior source split mismatch: {sample_id}")
            source_index = int(manifest.get("source_index", -1))
            if source_index != index:
                raise MaterializationError(f"prior source order mismatch: {sample_id}")
            prompt = str(training["prompt"])
            response = str(training["response"])
            if response.count("\n</think>\n") != 1:
                raise MaterializationError(f"prior response boundary mismatch: {sample_id}")
            reasoning, minutes = response.split("\n</think>\n", 1)
            analysis = _parse_analysis(prompt, sample_id=sample_id)
            expected_hashes = {
                "prompt_sha256": sha256_text(prompt),
                "response_sha256": sha256_text(response),
                "analysis_sha256": sha256_text(analysis),
                "reasoning_sha256": sha256_text(reasoning),
                "minutes_sha256": sha256_text(minutes),
            }
            for field, observed in expected_hashes.items():
                if manifest.get(field) != observed:
                    raise MaterializationError(
                        f"prior {field} mismatch: {sample_id}"
                    )
            teacher = manifest.get("teacher")
            if not isinstance(teacher, Mapping):
                raise MaterializationError(f"prior teacher provenance missing: {sample_id}")
            rows.append(
                PriorReleaseRow(
                    sample_id=sample_id,
                    source_split=source_split,
                    source_index=source_index,
                    analysis=analysis,
                    prompt=prompt,
                    reasoning=reasoning,
                    minutes=minutes,
                    response=response,
                    manifest=manifest,
                    teacher_response=_released_teacher_response(
                        reasoning=reasoning,
                        minutes=minutes,
                        teacher=teacher,
                    ),
                )
            )
        result[source_split] = rows
    if len(observed_ids) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise MaterializationError("prior chk3 population mismatch")
    return result


def _source_index(
    acquisition: Mapping[str, Sequence[AcquisitionRow]],
) -> dict[str, AcquisitionRow]:
    indexed: dict[str, AcquisitionRow] = {}
    for split in SPLITS:
        for row in acquisition[split]:
            if row.sample_id in indexed:
                raise MaterializationError(f"duplicate acquisition row: {row.sample_id}")
            indexed[row.sample_id] = row
    return indexed


def identify_rows_to_reacquire(
    prior: Mapping[str, Sequence[PriorReleaseRow]],
    *,
    source_by_id: Mapping[str, AcquisitionRow],
    tokenizer: Any,
) -> tuple[dict[str, list[PriorReleaseRow]], dict[str, dict[str, Any]]]:
    flagged: dict[str, list[PriorReleaseRow]] = {split: [] for split in SPLITS}
    reasons: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for row in prior[split]:
            source = source_by_id.get(row.sample_id)
            if source is None:
                raise MaterializationError(f"prior sample missing from acquisition: {row.sample_id}")
            prepared = _prepared_row(source, row.analysis)
            if prepared.source_index != row.source_index:
                raise MaterializationError(f"prior/acquisition order mismatch: {row.sample_id}")
            if prepared.user_prompt != row.prompt:
                raise MaterializationError(f"prior/acquisition prompt mismatch: {row.sample_id}")

            categories = sorted(_reasoning_meta_categories(row.reasoning))
            validator_errors: list[str] = []
            if not categories:
                try:
                    target = validate_teacher_target(
                        response=row.teacher_response,
                        analysis=row.analysis,
                        user_prompt=row.prompt,
                        tokenizer=tokenizer,
                    )
                    if target.reasoning != row.reasoning:
                        validator_errors.append("reasoning_requires_resanitization")
                except OutputContractError as exc:
                    validator_errors.extend(exc.codes)
            if categories or validator_errors:
                flagged[split].append(row)
                reasons[row.sample_id] = {
                    "sample_id": row.sample_id,
                    "split": split,
                    "reasoning_meta_categories": categories,
                    "validator_errors": validator_errors,
                    "prior_response_sha256": sha256_text(row.response),
                    "prior_teacher_response_id": row.teacher_response.response_id,
                }
    return flagged, reasons


def run(
    *,
    prior_release: Path,
    acquisition_root: Path,
    chk1_handoff: Path,
    tokenizer_path: Path,
    chk3_config: Path,
    work_root: Path,
    release_parent: Path,
    release_id: str,
    dry_run: bool,
    concurrency: int,
    backend: TeacherBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise MaterializationError("concurrency must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release_id):
        raise MaterializationError("release ID is invalid")

    acquisition = _load_acquisition(acquisition_root)
    source_by_id = _source_index(acquisition)
    prior = load_prior_release(prior_release)
    active_tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    flagged, reasons = identify_rows_to_reacquire(
        prior, source_by_id=source_by_id, tokenizer=active_tokenizer
    )
    flagged_ids = set(reasons)
    category_counts: dict[str, int] = {}
    for reason in reasons.values():
        for category in reason["reasoning_meta_categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1

    base_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared" if dry_run else "running",
        "prior_release": str(prior_release),
        "release_id": release_id,
        "total_rows": sum(EXPECTED_SPLIT_COUNTS.values()),
        "rows_to_reacquire": len(flagged_ids),
        "rows_reused": sum(EXPECTED_SPLIT_COUNTS.values()) - len(flagged_ids),
        "split_counts_to_reacquire": {
            split: len(flagged[split]) for split in SPLITS
        },
        "reasoning_meta_category_counts": dict(sorted(category_counts.items())),
        "api_requests_are_disabled": dry_run,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_root / "preflight_summary.json", base_summary)
    _write_jsonl(
        work_root / "rows_to_reacquire.jsonl",
        [reasons[sample_id] for sample_id in sorted(reasons)],
    )
    if dry_run:
        return base_summary

    provider = backend or OpenAIDeepSeekBackend()
    identity_guard = ProviderIdentityGuard()
    for split in SPLITS:
        for row in prior[split]:
            identity_guard.bind(
                returned_model=row.teacher_response.returned_model,
                system_fingerprint=row.teacher_response.system_fingerprint,
            )

    clean_rows: dict[str, list[CleanRow]] = {split: [] for split in SPLITS}
    pending: list[tuple[PriorReleaseRow, AcquisitionRow]] = []
    for split in SPLITS:
        for row in prior[split]:
            source = source_by_id[row.sample_id]
            prepared = _prepared_row(source, row.analysis)
            if row.sample_id in flagged_ids:
                pending.append((row, source))
                continue
            target = validate_teacher_target(
                response=row.teacher_response,
                analysis=row.analysis,
                user_prompt=row.prompt,
                tokenizer=active_tokenizer,
            )
            if target.reasoning != row.reasoning:
                raise MaterializationError(
                    f"unflagged reasoning changed during validation: {row.sample_id}"
                )
            clean_rows[split].append(
                CleanRow(
                    source=source,
                    prepared=prepared,
                    target=target,
                    teacher_response=row.teacher_response,
                    analysis_mode=str(row.manifest["analysis_mode"]),
                    target_mode=str(row.manifest["target_mode"]),
                )
            )

    generated: dict[str, CleanRow] = {}
    failures: dict[str, str] = {}
    api_calls = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                regenerate_chk3_target,
                _prepared_row(source, row.analysis),
                work_root=work_root,
                tokenizer=active_tokenizer,
                backend=provider,
                identity_guard=identity_guard,
                environment=environment,
            ): (row, source)
            for row, source in pending
        }
        completed = 0
        for future in as_completed(futures):
            row, source = futures[future]
            try:
                target, response, cache_hit = future.result()
                if _reasoning_meta_categories(target.reasoning):
                    raise MaterializationError(
                        f"regenerated reasoning still contains meta-talk: {row.sample_id}"
                    )
                generated[row.sample_id] = CleanRow(
                    source=source,
                    prepared=_prepared_row(source, row.analysis),
                    target=target,
                    teacher_response=response,
                    analysis_mode=str(row.manifest["analysis_mode"]),
                    target_mode="deepseek_meta_regenerated",
                )
                if not cache_hit:
                    api_calls += 1
                completed += 1
                print(
                    f"[chk3-meta-repair] regenerated={completed}/{len(pending)} "
                    f"sample_id={row.sample_id} cache_hit={cache_hit}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - preserve complete failure ledger
                failures[row.sample_id] = f"{type(exc).__name__}:{exc}"
    _write_jsonl(
        work_root / "failures.jsonl",
        [
            {"sample_id": sample_id, "error": failures[sample_id]}
            for sample_id in sorted(failures)
        ],
    )
    if failures:
        summary = {
            **base_summary,
            "status": "incomplete",
            "api_calls": api_calls,
            "accepted_regenerations": len(generated),
            "failures": len(failures),
        }
        _write_json(work_root / "summary.json", summary)
        raise MaterializationError(
            f"reasoning-meta target regeneration incomplete: {len(failures)} failures"
        )

    for split in SPLITS:
        existing = {row.prepared.sample_id: row for row in clean_rows[split]}
        for prior_row in prior[split]:
            replacement = generated.get(prior_row.sample_id)
            if replacement is not None:
                existing[prior_row.sample_id] = replacement
        clean_rows[split] = [existing[row.sample_id] for row in prior[split]]

    handoff = _publish_release(
        clean_rows,
        release_parent=release_parent,
        release_id=release_id,
        acquisition_root=acquisition_root,
        chk1_handoff=chk1_handoff,
        chk3_config=chk3_config,
        tokenizer=active_tokenizer,
        tokenizer_path=tokenizer_path,
        prior_release_root=prior_release,
    )
    summary = {
        **base_summary,
        "status": "complete",
        "api_calls": api_calls,
        "accepted_regenerations": len(generated),
        "failures": 0,
        "release_path": str(release_parent / release_id),
        "dataset_path": handoff["dataset_path"],
        "handoff": str(release_parent / release_id / "handoff.json"),
        "provider_identity": identity_guard.identity,
    }
    _write_json(work_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-release", type=Path, default=DEFAULT_PRIOR_RELEASE)
    parser.add_argument("--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT)
    parser.add_argument("--chk1-handoff", type=Path, default=DEFAULT_CHK1_HANDOFF)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--chk3-config", type=Path, default=DEFAULT_CHK3_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--release-parent", type=Path, default=DEFAULT_RELEASE_PARENT)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            prior_release=args.prior_release.resolve(),
            acquisition_root=args.acquisition_root.resolve(),
            chk1_handoff=args.chk1_handoff.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            chk3_config=args.chk3_config.resolve(),
            work_root=args.work_root.resolve(),
            release_parent=args.release_parent.resolve(),
            release_id=args.release_id,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    except (Chk3DataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
