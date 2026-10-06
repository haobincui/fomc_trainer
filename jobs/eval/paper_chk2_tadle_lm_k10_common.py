"""Frozen preparation contract for the paper-chk2 Tadle-form K=10 rerun.

This module deliberately prepares *atomic* Core8 analyses.  The historical
regression skeleton determines the 84 meeting union; the harmonized 2,048-row
Core8 release supplies exactly eight analyses for every meeting.  Prompts are
then re-rendered with the current paper-chk2 prompt and materialized to exact
token IDs shared by chk0, chk1, and paper_chk2_cp50.

No model is loaded by this module.  Generation and document assembly live in
``eval_paper_chk2_tadle_lm_vllm_k10_dual_dp1`` and are exposed here through
CLI-only lazy imports so the preparation contract remains GPU-free.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as source_contract
from jobs.eval import paper_chk2_core8_loo_common as paper_contract
from jobs.eval import paper_chk2_text_similarity_common as model_contract
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "paper-chk2-cp50-tadle-form-lm-ff-futures-k10-v1"
DEFAULT_OUTPUT_ROOT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_cp50_tadle_form_lm_ff_futures_k10_v1_20260902"
)

PREPARATION_DIRNAME = "preparation"
MANIFEST_FILENAME = "evaluation_manifest.json"
LEDGER_FILENAME = "atomic_prompt_ledger.jsonl"
OVERLAP_FILENAME = "training_overlap_audit.json"

PREPARATION_SCHEMA = "paper-chk2-tadle-lm-k10-preparation-v1"
LEDGER_ROW_SCHEMA = "paper-chk2-tadle-lm-k10-atomic-prompt-row-v1"
OVERLAP_SCHEMA = "paper-chk2-tadle-lm-k10-training-overlap-audit-v1"

TOPICS = tuple(source_contract.CORE_TOPICS)
MODEL_IDS = ("chk0", "chk1", "paper_chk2_cp50")
MODEL_LABELS: Mapping[str, str] = {
    "chk0": "Model chk-0",
    "chk1": "Model chk-1 cp200",
    "paper_chk2_cp50": "Model chk-2 cp50",
}
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)

EXPECTED_MEETINGS = 84
EXPECTED_ESTIMABLE_CURRENT_MEETINGS = 82
EXPECTED_STANDARDIZATION_MEETINGS = 83
EXPECTED_LAG_ONLY_MEETINGS = 2
EXPECTED_PROMPTS = EXPECTED_MEETINGS * len(TOPICS)
EXPECTED_GENERATIONS_PER_MODEL = EXPECTED_PROMPTS * len(REPLICATE_SEEDS)
EXPECTED_GENERATIONS = EXPECTED_GENERATIONS_PER_MODEL * len(MODEL_IDS)
EXPECTED_DOCUMENTS = EXPECTED_MEETINGS * len(REPLICATE_SEEDS) * len(MODEL_IDS)

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 2_560
MAX_MODEL_LEN = 4_096
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 4_096
EXPECTED_MIN_PROMPT_TOKENS = 282
EXPECTED_MAX_PROMPT_TOKENS = 378

POST_RELEASE_MANIFEST_SHA256 = (
    "d7e6d9ea534039f60343dcab317588b45c1de5aa2814d2e6fff0d8ca0f1feb21"
)
LEGACY_SOURCE_RELEASE = ROOT / (
    "output/evaluation/main/tadle_form_official_generated_ff_futures_2004_2015_v1"
)
LEGACY_DOCUMENT_RELEASE = ROOT / (
    "output/evaluation/main/official_minutes_full_document_treasury_beta_1993_2025_v1"
)
LEGACY_SOURCE_MANIFEST = LEGACY_SOURCE_RELEASE / "manifest.json"
LEGACY_SOURCE_MANIFEST_SHA256 = (
    "eeb29722e95903dadedfa9e496b6a8e017ef0b731d8e5a67befc6ffd198de44f"
)
LEGACY_SOURCE_PANEL = LEGACY_SOURCE_RELEASE / "estimation/analysis_panel.jsonl"
LEGACY_SOURCE_PANEL_SHA256 = (
    "72fdcd8aac1063b8d0024ae1f8b7151a064bce7e5b1a45c58e183282a2e25c6b"
)
STATEMENT_DOCUMENTS = LEGACY_SOURCE_RELEASE / "statements/documents.jsonl"
STATEMENT_DOCUMENTS_SHA256 = (
    "972300ce03ba08740ef292e231a78bac20004ed2ae175a5db27bd8af76daec71"
)
OFFICIAL_DOCUMENTS = LEGACY_DOCUMENT_RELEASE / "documents/official_documents.jsonl"
OFFICIAL_DOCUMENTS_SHA256 = (
    "5432c4b04f61da51aae82eeb2e4a7b192564a60c4b6cd2aa39c2a7eb9f4bbb40"
)
POST_CUTOFF = "2011-08-08"


class PaperChk2TadlePreparationError(RuntimeError):
    """A source, model, prompt, tokenizer, or immutable artifact drifted."""


@dataclass(frozen=True, slots=True)
class PreparedArtifacts:
    manifest_path: Path
    manifest: Mapping[str, Any]
    ledger_path: Path
    ledger_rows: tuple[Mapping[str, Any], ...]
    overlap_path: Path
    overlap_audit: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2TadlePreparationError(message)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PaperChk2TadlePreparationError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _read_json(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    _require(not candidate.is_symlink(), f"refusing JSON symlink: {candidate}")
    resolved = candidate.resolve()
    _require(
        resolved.is_file(),
        f"missing/non-regular JSON artifact: {resolved}",
    )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperChk2TadlePreparationError(
            f"cannot read JSON artifact {resolved}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON object required: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    candidate = path.expanduser()
    _require(not candidate.is_symlink(), f"refusing JSONL symlink: {candidate}")
    resolved = candidate.resolve()
    _require(
        resolved.is_file(),
        f"missing/non-regular JSONL artifact: {resolved}",
    )
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            _require(
                raw.endswith("\n"), f"unterminated JSONL row: {resolved}:{line_number}"
            )
            _require(bool(raw.strip()), f"blank JSONL row: {resolved}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PaperChk2TadlePreparationError(
                    f"invalid JSONL row: {resolved}:{line_number}"
                ) from exc
            _require(
                isinstance(row, dict),
                f"JSONL object required: {resolved}:{line_number}",
            )
            rows.append(row)
    return rows


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    candidate = path.expanduser()
    _require(not candidate.is_symlink(), f"refusing bound-file symlink: {candidate}")
    resolved = candidate.resolve()
    _require(resolved.is_file(), f"bound file missing: {resolved}")
    record: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _verify_file(
    path: Path, expected_sha256: str, *, rows: int | None = None
) -> dict[str, Any]:
    record = _record(path, rows=rows)
    _require(record["sha256"] == expected_sha256, f"bound file SHA drift: {path}")
    return record


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink(), f"refusing output symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        if published:
            _fsync_directory(path.parent)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(
                value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new_bytes(
        path,
        "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _preparation_lock(output_root: Path) -> Any:
    """Serialize create/verify so a concurrent prepare can never replace data."""

    lock_path = output_root / ".preparation.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield {"path": str(lock_path.resolve()), "pid": os.getpid()}
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _render_user_prompt(user_template: str, source_analysis: str) -> str:
    marker = '"[SOURCE_ANALYSIS]"'
    _require(user_template.count(marker) == 1, "user-template marker contract drift")
    return user_template.replace(
        marker, json.dumps(source_analysis, ensure_ascii=False)
    )


def _prompt_token_ids(
    tokenizer: Any, system_prompt: str, user_prompt: str
) -> list[int]:
    try:
        values = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise PaperChk2TadlePreparationError(
            f"chat-template rendering failed: {exc}"
        ) from exc
    if hasattr(values, "tolist"):
        values = values.tolist()
    _require(
        isinstance(values, list) and bool(values), "chat template returned no token IDs"
    )
    _require(not isinstance(values[0], list), "unexpected batched prompt token IDs")
    token_ids = [int(value) for value in values]
    _require(all(value >= 0 for value in token_ids), "negative prompt token ID")
    return token_ids


def _load_source_population() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the exact 84 x Core8 atomic population in regression order."""

    # The legacy statistics module imports pandas/statsmodels.  It is needed
    # only while preparing the frozen meeting roster and must never be imported
    # by the lean vLLM worker runtime.
    from jobs.eval import run_tadle_interaction_wgan_dictionary_v1 as legacy_statistics

    source_manifest = _verify_file(
        LEGACY_SOURCE_MANIFEST, LEGACY_SOURCE_MANIFEST_SHA256
    )
    source_panel = _verify_file(
        LEGACY_SOURCE_PANEL, LEGACY_SOURCE_PANEL_SHA256, rows=163_200
    )
    statements = _verify_file(
        STATEMENT_DOCUMENTS, STATEMENT_DOCUMENTS_SHA256, rows=EXPECTED_MEETINGS
    )
    official = _verify_file(OFFICIAL_DOCUMENTS, OFFICIAL_DOCUMENTS_SHA256, rows=256)
    skeleton = legacy_statistics.load_source_skeleton()
    needed_ids = tuple(str(value) for value in skeleton["needed_ids"])
    current_ids = tuple(str(value) for value in skeleton["current_ids"])
    lag_ids = tuple(str(value) for value in skeleton["lag_ids"])
    lag_only = tuple(sorted(set(lag_ids) - set(current_ids)))
    standardization_ids = tuple(sorted(needed_ids)[1:])
    release_ids = standardization_ids
    event_lag_pairs = [
        {"current_meeting_id": current, "lag_meeting_id": lag}
        for current, lag in zip(current_ids, lag_ids, strict=True)
    ]
    _require(
        len(needed_ids) == EXPECTED_MEETINGS, "regression meeting union is not N=84"
    )
    _require(
        len(current_ids)
        == len(set(current_ids))
        == EXPECTED_ESTIMABLE_CURRENT_MEETINGS,
        "estimable current-event inventory is not N=82",
    )
    _require(
        len(lag_only) == EXPECTED_LAG_ONLY_MEETINGS
        and lag_only == ("2004-09-21", "2009-12-16"),
        "lag-only meeting inventory drift",
    )
    _require(
        len(standardization_ids) == EXPECTED_STANDARDIZATION_MEETINGS,
        "standardization meeting inventory is not N=83",
    )

    try:
        harmonized = source_contract.load_harmonized_sources(
            pre_release_sha256=source_contract.PRE_RELEASE_MANIFEST_SHA256,
            post_release_sha256=POST_RELEASE_MANIFEST_SHA256,
        )
    except source_contract.MergedCore8ContractError as exc:
        raise PaperChk2TadlePreparationError(str(exc)) from exc
    selected = [
        dict(row)
        for row in harmonized.rows
        if str(row["meeting_end_date"]) in set(needed_ids)
    ]
    by_meeting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_meeting[str(row["meeting_end_date"])].append(row)
    _require(
        set(by_meeting) == set(needed_ids),
        "Core8/regression meeting join is incomplete",
    )
    ordered: list[dict[str, Any]] = []
    for meeting_rank, meeting_id in enumerate(needed_ids):
        meeting_rows = sorted(
            by_meeting[meeting_id], key=lambda row: int(row["topic_order"])
        )
        _require(
            len(meeting_rows) == len(TOPICS)
            and tuple(str(row["topic"]) for row in meeting_rows) == TOPICS,
            f"meeting is not exact ordered Core8: {meeting_id}",
        )
        for topic_rank, row in enumerate(meeting_rows):
            row["meeting_rank"] = meeting_rank
            row["topic_rank"] = topic_rank
            ordered.append(row)
    _require(len(ordered) == EXPECTED_PROMPTS, "atomic source population is not N=672")
    _require(
        len({str(row["source_sample_id"]) for row in ordered}) == EXPECTED_PROMPTS,
        "atomic source sample IDs are not unique",
    )

    statement_ids = {
        str(row.get("meeting_end_date")) for row in _read_jsonl(STATEMENT_DOCUMENTS)
    }
    official_ids = {
        str(row.get("meeting_end_date")) for row in _read_jsonl(OFFICIAL_DOCUMENTS)
    }
    _require(
        set(needed_ids) <= statement_ids, "84-meeting Statement join is incomplete"
    )
    _require(
        set(needed_ids) <= official_ids,
        "84-meeting official-Minutes join is incomplete",
    )
    return ordered, {
        "legacy_regression_manifest": source_manifest,
        "legacy_analysis_panel": source_panel,
        "statement_documents": statements,
        "official_minutes_documents": official,
        "harmonized_source_releases": {
            key: dict(value) for key, value in harmonized.source_bindings.items()
        },
        "needed_meeting_ids": list(needed_ids),
        "estimable_current_meeting_ids": list(current_ids),
        "lag_meeting_ids": list(lag_ids),
        "lag_only_meeting_ids": list(lag_only),
        "standardization_meeting_ids": list(standardization_ids),
        "release_meeting_ids": list(release_ids),
        "event_lag_pairs": event_lag_pairs,
    }


def _build_overlap_audit(source_meta: Mapping[str, Any]) -> dict[str, Any]:
    paper_rows: list[dict[str, Any]] = []
    sidecar_bindings: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        path = (
            paper_contract.PAPER_RELEASE_ROOT
            / f"minutes_alignment/manifests/{split}.jsonl"
        )
        record = _verify_file(
            path,
            paper_contract.PAPER_SIDECAR_SHA256[split],
            rows=paper_contract.PAPER_SPLIT_ROWS[split],
        )
        rows = _read_jsonl(path)
        _require(
            len(rows) == paper_contract.PAPER_SPLIT_ROWS[split],
            f"paper {split} row count drift",
        )
        _require(
            all(row.get("split") == split for row in rows), f"paper {split} role drift"
        )
        sidecar_bindings[split] = record
        paper_rows.extend(rows)
    by_split = {
        split: {str(row["meeting_date"]) for row in paper_rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    needed = set(str(value) for value in source_meta["needed_meeting_ids"])
    scale = set(str(value) for value in source_meta["standardization_meeting_ids"])
    current = set(str(value) for value in source_meta["estimable_current_meeting_ids"])
    pairs = tuple(source_meta["event_lag_pairs"])
    train_dates = by_split["train"]
    post_pairs = [
        pair for pair in pairs if str(pair["current_meeting_id"]) > POST_CUTOFF
    ]
    no_train_overlap_pairs = [
        pair
        for pair in pairs
        if str(pair["current_meeting_id"]) not in train_dates
        and str(pair["lag_meeting_id"]) not in train_dates
    ]
    post_with_train_overlap = [
        pair
        for pair in post_pairs
        if str(pair["current_meeting_id"]) in train_dates
        or str(pair["lag_meeting_id"]) in train_dates
    ]
    overlap = {
        split: {
            "union_84": sorted(needed & dates),
            "standardization_83": sorted(scale & dates),
            "estimable_current_82": sorted(current & dates),
        }
        for split, dates in by_split.items()
    }
    _require(
        len(overlap["train"]["standardization_83"]) == 47,
        "train/standardization overlap drift",
    )
    _require(
        len(overlap["train"]["estimable_current_82"]) == 46,
        "train/estimable overlap drift",
    )
    _require(
        not overlap["validation"]["union_84"] and not overlap["test"]["union_84"],
        "validation/test meetings unexpectedly overlap the regression sample",
    )
    _require(len(post_pairs) == 30, "post-2011 estimable-event inventory drift")
    _require(
        len(post_with_train_overlap) == len(post_pairs) == 30,
        "not all post-2011 shocks have current-or-lag paper-train overlap",
    )
    _require(
        len(no_train_overlap_pairs) == 34
        and all(
            str(pair["current_meeting_id"]) <= POST_CUTOFF
            for pair in no_train_overlap_pairs
        ),
        "nonoverlap estimable-event inventory drift",
    )
    return {
        "schema_version": OVERLAP_SCHEMA,
        "status": "complete",
        "paper_chk2_training_sidecars": sidecar_bindings,
        "overlap_by_split": overlap,
        "counts": {
            split: {
                population: len(values) for population, values in populations.items()
            }
            for split, populations in overlap.items()
        },
        "event_pair_overlap": {
            "estimable_events": len(pairs),
            "post_2011_estimable_events": len(post_pairs),
            "post_2011_current_or_lag_in_paper_train": len(post_with_train_overlap),
            "current_and_lag_both_absent_from_paper_train": len(no_train_overlap_pairs),
            "all_no_overlap_events_are_pre_2011": True,
        },
        "interpretation": {
            "mixed_scope_historical_diagnostic": True,
            "leakage_safe_evaluation": False,
            "post_period_external_generalization_claim_allowed": False,
            "note": (
                "The N=83 quantity is the sentiment-standardization population. "
                "The estimable current-event inventory is N=82; the N=84 union "
                "contains two lag-only meetings."
            ),
        },
    }


def _model_manifest_bindings() -> dict[str, Any]:
    bindings = model_contract._verify_static_bindings()
    return {
        "chk0": {
            "path": str(model_contract.CHK0_MODEL.resolve()),
            "lineage_full_directory_sha256": model_contract.CHK0_DIGEST,
            "runtime_payload_sha256": model_contract.CHK0_RUNTIME_PAYLOAD_DIGEST,
            "fingerprint": bindings["chk0"],
        },
        "chk1": {
            "path": str(model_contract.CHK1_MODEL.resolve()),
            "sha256": model_contract.CHK1_DIGEST,
            "fingerprint": bindings["chk1"],
        },
        "paper_chk2_cp50": {
            "adapter_path": str(model_contract.CHK2_ADAPTER.resolve()),
            "base_model_path": str(model_contract.CHK1_MODEL.resolve()),
            "adapter_model_sha256": model_contract.CHK2_ADAPTER_MODEL_SHA256,
            "adapter_config_sha256": model_contract.CHK2_ADAPTER_CONFIG_SHA256,
            "fingerprint": bindings["chk2"],
            "selection_receipt": bindings["selection_receipt"],
            "selection_receipt_is_evaluation_authorization": False,
        },
    }


def _build_ledger(
    source_rows: Sequence[Mapping[str, Any]],
    bindings: paper_contract.PaperChk2Bindings,
) -> list[dict[str, Any]]:
    _require(bindings.tokenizer is not None, "loaded paper tokenizer is required")
    try:
        from transformers import AutoTokenizer

        chk0_tokenizer = AutoTokenizer.from_pretrained(
            str(model_contract.CHK0_MODEL),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PaperChk2TadlePreparationError(
            f"cannot load chk0 tokenizer: {exc}"
        ) from exc

    ledger: list[dict[str, Any]] = []
    for absolute_prompt_index, source in enumerate(source_rows):
        analysis = str(source["source_analysis"])
        prompt = _render_user_prompt(bindings.user_template, analysis)
        paper_ids = _prompt_token_ids(
            bindings.tokenizer, bindings.system_prompt, prompt
        )
        chk0_ids = _prompt_token_ids(chk0_tokenizer, bindings.system_prompt, prompt)
        _require(
            chk0_ids == paper_ids,
            f"chk0/chk1 prompt-token drift: {source['source_sample_id']}",
        )
        _require(
            len(paper_ids) <= MAX_PROMPT_TOKENS,
            f"prompt context overflow: {source['source_sample_id']}",
        )
        row: dict[str, Any] = {
            "schema_version": LEDGER_ROW_SCHEMA,
            "absolute_prompt_index": absolute_prompt_index,
            "source_sample_id": source["source_sample_id"],
            "sample_id": source["source_sample_id"],
            "meeting_id": source["meeting_end_date"],
            "meeting_start_date": source["meeting_start_date"],
            "meeting_end_date": source["meeting_end_date"],
            "meeting_rank": source["meeting_rank"],
            "topic": source["topic"],
            "topic_rank": source["topic_rank"],
            "era": source["era"],
            "source_split": source["source_split"],
            "source_analysis": analysis,
            "source_analysis_sha256": sha256_text(analysis),
            "user_prompt": prompt,
            "user_prompt_sha256": sha256_text(prompt),
            "messages_sha256": sha256_text(
                canonical_json(
                    [
                        {"role": "system", "content": bindings.system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
            ),
            "prompt_token_count": len(paper_ids),
            "prompt_token_ids": paper_ids,
            "prompt_token_ids_sha256": sha256_text(canonical_json(paper_ids)),
            "model_input_fields": ["prompt_token_ids"],
            "official_minutes_in_model_input": False,
            "statement_in_model_input": False,
            "market_data_in_model_input": False,
        }
        row["row_sha256"] = sha256_text(canonical_json(row))
        ledger.append(row)
    counts = [int(row["prompt_token_count"]) for row in ledger]
    _require(len(ledger) == EXPECTED_PROMPTS, "prompt ledger is not N=672")
    _require(
        min(counts) == EXPECTED_MIN_PROMPT_TOKENS
        and max(counts) == EXPECTED_MAX_PROMPT_TOKENS,
        f"prompt-token range drift: {min(counts)}--{max(counts)}",
    )
    return ledger


def _artifact_paths(output_root: Path) -> tuple[Path, Path, Path, Path]:
    candidate = output_root.expanduser()
    _require(not candidate.is_symlink(), f"refusing output-root symlink: {candidate}")
    preparation = candidate.resolve() / PREPARATION_DIRNAME
    return (
        preparation,
        preparation / MANIFEST_FILENAME,
        preparation / LEDGER_FILENAME,
        preparation / OVERLAP_FILENAME,
    )


def build_preparation_artifacts(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PreparedArtifacts:
    """Create the immutable preparation directory, or deep-verify it."""

    unresolved = output_root.expanduser()
    _require(not unresolved.is_symlink(), "refusing a symlink output root")
    output_root = unresolved.resolve()
    preparation, manifest_path, ledger_path, overlap_path = _artifact_paths(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _require(not output_root.is_symlink(), "output root became a symlink")
    with _preparation_lock(output_root):
        if preparation.exists() or preparation.is_symlink():
            return validate_preparation_artifacts(
                output_root, verify_static_bindings=True
            )

        source_rows, source_meta = _load_source_population()
        paper_bindings = paper_contract.verify_paper_chk2_bindings(load_tokenizer=True)
        ledger = _build_ledger(source_rows, paper_bindings)
        overlap = _build_overlap_audit(source_meta)
        model_bindings = _model_manifest_bindings()

        temporary = Path(
            tempfile.mkdtemp(prefix=".tadle_k10_preparation.", dir=output_root)
        )
        published = False
        try:
            tmp_ledger = temporary / LEDGER_FILENAME
            tmp_overlap = temporary / OVERLAP_FILENAME
            tmp_manifest = temporary / MANIFEST_FILENAME
            _write_new_jsonl(tmp_ledger, ledger)
            _write_new_json(tmp_overlap, overlap)
            ledger_record = {
                **_record(tmp_ledger, rows=EXPECTED_PROMPTS),
                "path": str(ledger_path.resolve()),
                "row_schema_version": LEDGER_ROW_SCHEMA,
            }
            overlap_record = {
                **_record(tmp_overlap),
                "path": str(overlap_path.resolve()),
                "schema_version": OVERLAP_SCHEMA,
            }
            token_counts = [int(row["prompt_token_count"]) for row in ledger]
            manifest = seal_manifest(
                {
                    "schema_version": PREPARATION_SCHEMA,
                    "evaluation_id": EVALUATION_ID,
                    "created_at_utc": _utc_now(),
                    "status": "prepared",
                    "immutable": True,
                    "scope": {
                        "mixed_scope_historical_document_source_association_diagnostic": True,
                        "leakage_safe_evaluation": False,
                        "held_out_generalization_claim_allowed": False,
                        "loo_generation_rows_reused": False,
                        "causal_market_effect_claim_allowed": False,
                    },
                    "population": {
                        "meeting_union": EXPECTED_MEETINGS,
                        "estimable_current_events": EXPECTED_ESTIMABLE_CURRENT_MEETINGS,
                        "release_meetings": EXPECTED_STANDARDIZATION_MEETINGS,
                        "standardization_meetings": EXPECTED_STANDARDIZATION_MEETINGS,
                        "lag_only_meetings": EXPECTED_LAG_ONLY_MEETINGS,
                        "topics_per_meeting": len(TOPICS),
                        "atomic_prompts": EXPECTED_PROMPTS,
                        "topics": list(TOPICS),
                        "meeting_ids": source_meta["needed_meeting_ids"],
                        "estimable_current_meeting_ids": source_meta[
                            "estimable_current_meeting_ids"
                        ],
                        "lag_only_meeting_ids": source_meta["lag_only_meeting_ids"],
                        "standardization_meeting_ids": source_meta[
                            "standardization_meeting_ids"
                        ],
                        "release_meeting_ids": source_meta["release_meeting_ids"],
                    },
                    "inputs": {
                        "atomic_prompt_ledger": ledger_record,
                        "training_overlap_audit": overlap_record,
                        "legacy_regression": {
                            key: value
                            for key, value in source_meta.items()
                            if key
                            in {
                                "legacy_regression_manifest",
                                "legacy_analysis_panel",
                                "statement_documents",
                                "official_minutes_documents",
                                "harmonized_source_releases",
                            }
                        },
                    },
                    "models": model_bindings,
                    "paper_chk2_prompt": {
                        "prompt_contract": dict(
                            paper_bindings.file_bindings["prompt_contract"]
                        ),
                        "system_prompt_sha256": paper_bindings.system_prompt_sha256,
                        "user_template_sha256": paper_bindings.user_template_sha256,
                        "opening_think_supplied_by_chat_template": True,
                        "response_boundary": "</think>",
                        "same_exact_prompt_token_ids_for_all_models": True,
                    },
                    "generation": {
                        "backend": "vllm-two-independent-single-gpu-dp1-workers",
                        "models": list(MODEL_IDS),
                        "replicate_seeds": list(REPLICATE_SEEDS),
                        "replicates": len(REPLICATE_SEEDS),
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "repetition_penalty": REPETITION_PENALTY,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "max_model_len": MAX_MODEL_LEN,
                        "max_num_seqs": MAX_NUM_SEQS,
                        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                        "dtype": "bfloat16",
                        "quantization": None,
                        "rows_per_model": EXPECTED_GENERATIONS_PER_MODEL,
                        "total_generation_rows": EXPECTED_GENERATIONS,
                        "generated_documents": EXPECTED_DOCUMENTS,
                        "row_seed": "derive_row_seed(replicate_seed,meeting_id+'|'+topic)",
                        "paired_shard": "sha256(meeting_id+'|'+replicate_id)[0:8] mod 2",
                        "transport_retries_after_primary": 2,
                    },
                    "validation": {
                        "source_meeting_topic_join": "84x8_complete",
                        "statement_join": "84_complete",
                        "official_minutes_join": "84_complete",
                        "minimum_prompt_tokens": min(token_counts),
                        "maximum_prompt_tokens": max(token_counts),
                        "all_inputs_within_4096_without_truncation": True,
                        "official_minutes_not_in_model_input": True,
                        "statement_not_in_model_input": True,
                        "market_data_not_in_model_input": True,
                    },
                }
            )
            _write_new_json(tmp_manifest, manifest)
            _fsync_directory(temporary)
            # The lock makes this rename create-once: no concurrent producer
            # can materialize ``preparation`` between the existence check and
            # publication.
            _require(
                not preparation.exists() and not preparation.is_symlink(),
                "preparation appeared during locked publish",
            )
            os.rename(temporary, preparation)
            _fsync_directory(output_root)
            published = True
        finally:
            if not published and temporary.exists():
                for candidate in temporary.iterdir():
                    candidate.unlink()
                temporary.rmdir()
    return validate_preparation_artifacts(output_root, verify_static_bindings=True)


def validate_preparation_artifacts(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    verify_static_bindings: bool = True,
) -> PreparedArtifacts:
    preparation, manifest_path, ledger_path, overlap_path = _artifact_paths(output_root)
    _require(
        preparation.is_dir() and not preparation.is_symlink(),
        "unsafe/missing preparation directory",
    )
    manifest = _read_json(manifest_path)
    try:
        validate_manifest_integrity(manifest)
    except ValueError as exc:
        raise PaperChk2TadlePreparationError(
            f"preparation integrity failed: {exc}"
        ) from exc
    _require(
        manifest.get("schema_version") == PREPARATION_SCHEMA, "preparation schema drift"
    )
    _require(manifest.get("status") == "prepared", "preparation status drift")
    _require(manifest.get("evaluation_id") == EVALUATION_ID, "evaluation ID drift")
    _require(manifest.get("immutable") is True, "preparation is not immutable")
    population = manifest.get("population") or {}
    _require(
        population.get("meeting_union") == EXPECTED_MEETINGS, "meeting union drift"
    )
    _require(
        population.get("release_meetings") == EXPECTED_STANDARDIZATION_MEETINGS,
        "release meeting count drift",
    )
    _require(
        population.get("standardization_meetings") == EXPECTED_STANDARDIZATION_MEETINGS,
        "standardization meeting count drift",
    )
    _require(
        population.get("atomic_prompts") == EXPECTED_PROMPTS,
        "atomic prompt count drift",
    )
    _require(tuple(population.get("topics") or ()) == TOPICS, "topic order drift")
    generation = manifest.get("generation") or {}
    _require(generation.get("models") == list(MODEL_IDS), "model order drift")
    _require(
        generation.get("replicate_seeds") == list(REPLICATE_SEEDS),
        "replicate seed drift",
    )
    _require(
        generation.get("total_generation_rows") == EXPECTED_GENERATIONS,
        "generation count drift",
    )
    ledger_binding = (manifest.get("inputs") or {}).get("atomic_prompt_ledger") or {}
    overlap_binding = (manifest.get("inputs") or {}).get("training_overlap_audit") or {}
    _require(
        ledger_binding.get("path") == str(ledger_path.resolve()), "ledger path drift"
    )
    _require(
        overlap_binding.get("path") == str(overlap_path.resolve()), "overlap path drift"
    )
    _require(
        _record(ledger_path, rows=EXPECTED_PROMPTS)["sha256"]
        == ledger_binding.get("sha256"),
        "ledger SHA drift",
    )
    _require(
        _record(overlap_path)["sha256"] == overlap_binding.get("sha256"),
        "overlap SHA drift",
    )
    ledger = _read_jsonl(ledger_path)
    overlap = _read_json(overlap_path)
    _require(len(ledger) == EXPECTED_PROMPTS, "ledger is not N=672")
    _require(
        len({row.get("sample_id") for row in ledger}) == EXPECTED_PROMPTS,
        "ledger sample IDs are not unique",
    )
    _require(
        Counter(str(row.get("topic")) for row in ledger)
        == Counter({topic: EXPECTED_MEETINGS for topic in TOPICS}),
        "ledger topic allocation drift",
    )
    by_meeting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(ledger):
        _require(
            row.get("schema_version") == LEDGER_ROW_SCHEMA, "ledger row schema drift"
        )
        _require(row.get("absolute_prompt_index") == index, "ledger ordering drift")
        _require(
            row.get("row_sha256")
            == sha256_text(
                canonical_json(
                    {key: value for key, value in row.items() if key != "row_sha256"}
                )
            ),
            "ledger row digest drift",
        )
        token_ids = row.get("prompt_token_ids")
        _require(
            isinstance(token_ids, list) and bool(token_ids),
            "ledger prompt token IDs missing",
        )
        _require(
            row.get("prompt_token_count") == len(token_ids),
            "ledger prompt token count drift",
        )
        _require(
            row.get("prompt_token_ids_sha256")
            == sha256_text(canonical_json(token_ids)),
            "ledger prompt token SHA drift",
        )
        _require(
            len(token_ids) + MAX_NEW_TOKENS <= MAX_MODEL_LEN, "ledger context overflow"
        )
        _require(
            row.get("official_minutes_in_model_input") is False,
            "official Minutes entered model input",
        )
        _require(
            row.get("statement_in_model_input") is False,
            "Statement entered model input",
        )
        by_meeting[str(row["meeting_id"])].append(row)
    _require(len(by_meeting) == EXPECTED_MEETINGS, "ledger meeting count drift")
    for meeting_id, rows in by_meeting.items():
        ordered = sorted(rows, key=lambda row: int(row["topic_rank"]))
        _require(
            tuple(str(row["topic"]) for row in ordered) == TOPICS,
            f"ledger topic order drift: {meeting_id}",
        )
    _require(
        overlap.get("schema_version") == OVERLAP_SCHEMA, "overlap audit schema drift"
    )
    _require(
        overlap.get("counts", {}).get("train", {}).get("standardization_83") == 47,
        "training overlap drift",
    )
    if verify_static_bindings:
        _load_source_population()
        paper_contract.verify_paper_chk2_bindings(load_tokenizer=False)
        _model_manifest_bindings()
    return PreparedArtifacts(
        manifest_path=manifest_path,
        manifest=manifest,
        ledger_path=ledger_path,
        ledger_rows=tuple(ledger),
        overlap_path=overlap_path,
        overlap_audit=overlap,
    )


def load_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> PreparedArtifacts:
    return validate_preparation_artifacts(output_root, verify_static_bindings=False)


def generation_contract() -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "tensor_parallel_size_per_engine": 1,
        "data_parallel_size_per_engine": 1,
        "pipeline_parallel_size_per_engine": 1,
        "independent_engine_count": 2,
        "transport_retries_after_primary": 2,
        "generation_quality_is_diagnostic_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "assemble",
            "validate-preparation",
            "validate-smoke",
            "validate-all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result: Any = build_preparation_artifacts(args.output_root)
        output = {
            "status": "prepared",
            "manifest": _record(result.manifest_path),
            "ledger": _record(result.ledger_path, rows=EXPECTED_PROMPTS),
        }
    elif args.command == "validate-preparation":
        result = validate_preparation_artifacts(
            args.output_root, verify_static_bindings=True
        )
        output = {
            "status": "verified",
            "manifest": _record(result.manifest_path),
            "ledger": _record(result.ledger_path, rows=EXPECTED_PROMPTS),
        }
    else:
        from jobs.eval import eval_paper_chk2_tadle_lm_vllm_k10_dual_dp1 as generation

        if args.command == "assemble":
            output = generation.assemble_generated_documents(
                output_root=args.output_root, resume=args.resume
            )
        else:
            output = generation.validate_all_models(
                manifest_path=args.output_root
                / PREPARATION_DIRNAME
                / MANIFEST_FILENAME,
                output_root=args.output_root,
                smoke=args.command == "validate-smoke",
            )
    print(canonical_json(output), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "EVALUATION_ID",
    "EXPECTED_DOCUMENTS",
    "EXPECTED_GENERATIONS",
    "EXPECTED_GENERATIONS_PER_MODEL",
    "EXPECTED_MEETINGS",
    "EXPECTED_PROMPTS",
    "LEDGER_ROW_SCHEMA",
    "MAX_MODEL_LEN",
    "MAX_NEW_TOKENS",
    "MODEL_IDS",
    "MODEL_LABELS",
    "PaperChk2TadlePreparationError",
    "PreparedArtifacts",
    "REPLICATE_SEEDS",
    "TOPICS",
    "build_preparation_artifacts",
    "canonical_json",
    "generation_contract",
    "load_prepared",
    "sha256_text",
    "validate_preparation_artifacts",
]
