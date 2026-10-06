"""Build, run, and score the CHK3 cp318 Core8 leave-one-out smoke.

The experiment is deliberately separate from the historical canonical LOO.
It measures stage-local indicator sensitivity for the exact-merged CHK3 cp318
model on eight frozen D-1 indicator analyses from each of eight preregistered
1993--2008 meetings.  Each meeting has one full prompt, eight exact deletions,
and eight token-count-matched neutral replacements (136 rows total).

All generation text and token IDs are persisted after every row.  GPU work is
restricted to physical GPU0, while semantic scoring is an offline follow-up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as gpu_guard
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as semantic_shared
from jobs.generation import loo_prompt_builder
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA = "chk3-core8-loo-smoke-config-v1"
SAMPLE_SCHEMA = "chk3-core8-loo-smoke-samples-v1"
INPUT_ROW_SCHEMA = "chk3-core8-loo-smoke-input-row-v1"
RUN_SCHEMA = "chk3-core8-loo-smoke-generation-run-v1"
STATE_SCHEMA = "chk3-core8-loo-smoke-generation-state-v1"
SCORE_ROW_SCHEMA = "chk3-core8-loo-smoke-score-row-v1"
SCORE_SCHEMA = "chk3-core8-loo-smoke-score-manifest-v1"
EVALUATION_ID = "chk3-cp318-core8-loo-smoke-1993-2008-v1"
TASK_CONTRACT_ID = native_eval.TASK_CONTRACT_ID
EXPECTED_MEETINGS = 8
EXPECTED_TOPICS = 8
EXPECTED_ROWS = 136
MODEL_LABEL = "chk3-cp318-exact-merged-core8-loo-smoke"
DEFAULT_CONFIG = ROOT / "configs/main/chk3_cp318_core8_loo_smoke_1993_2008_v1.json"
DEFAULT_SEMANTIC_MANIFEST = ROOT / "configs/main/checkpoint_eval_semantic_models.json"
GPU_LOCK = Path("/tmp/fomc_trainer_chk3_core8_loo_smoke_gpu0.lock")
EVALUATION_SCOPE = "interface_smoke_no_population_inference"
PREPARE_LIMITATIONS = (
    "This is an interface and infrastructure smoke, not population inference.",
    "The primary reference is a deterministic concatenation of eight source-grounded Minutes-style references, not an official Minutes excerpt.",
    "The result measures CHK3 stage-local indicator sensitivity, not causal contribution.",
)
RUN_LIMITATIONS = (
    "This N8 smoke is descriptive and does not authorize population inference.",
    "The generation path uses 4-bit NF4 inference over the exact-merged cp318 artifact.",
)
SCORE_LIMITATIONS = (
    "N8 smoke results are descriptive and must not be reported as formal population inference.",
    "The source-grounded concatenated reference is not an official Minutes excerpt.",
    "Positive target-relative delta does not identify a causal indicator contribution.",
)
FAILURE_ROW_SCHEMA = "chk3-core8-loo-smoke-failure-v1"


class Core8LooSmokeError(RuntimeError):
    """The smoke cannot proceed without violating a sealed contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Core8LooSmokeError(f"value is not canonical finite JSON: {exc}") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return native_probe.common_probe.sha256_file(path.expanduser().resolve())


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8LooSmokeError(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Core8LooSmokeError(f"invalid JSON file {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise Core8LooSmokeError(f"JSON root is not an object: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8LooSmokeError(f"missing regular JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Core8LooSmokeError(
                        f"blank line in JSONL: {resolved}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Core8LooSmokeError(
                        f"non-object JSONL row: {resolved}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Core8LooSmokeError(f"invalid JSONL file {resolved}: {exc}") from exc
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, content: bytes) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise Core8LooSmokeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise Core8LooSmokeError(f"refusing to overwrite artifact: {path}") from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    body = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    _write_new_bytes(path, body.encode("utf-8"))


def _file_binding(
    path: Path, *, rows: int | None = None, sealed: bool = False
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    if sealed:
        result["payload_sha256"] = validate_manifest_integrity(_read_json(resolved))
    return result


def _resolve(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Core8LooSmokeError(f"config has no {label}")
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    resolved = path.expanduser().resolve()
    config = _read_json(resolved)
    if config.get("schema_version") != CONFIG_SCHEMA or config.get(
        "evaluation_id"
    ) != EVALUATION_ID:
        raise Core8LooSmokeError("unsupported smoke config identity")
    if config.get("expected_generation_rows") != EXPECTED_ROWS:
        raise Core8LooSmokeError("smoke config row count drift")
    topics = config.get("topics")
    if not isinstance(topics, list) or len(topics) != EXPECTED_TOPICS or len(
        set(topics)
    ) != EXPECTED_TOPICS:
        raise Core8LooSmokeError("smoke config Core8 inventory drift")
    root = ROOT
    paths = {
        key: _resolve(root, config[key], label=key)
        for key in (
            "source_release_manifest",
            "source_panel",
            "training_config",
            "model",
            "model_anchor_manifest",
            "exact_merge_evidence",
            "semantic_manifest",
        )
    }
    for key in (
        "source_release_manifest",
        "source_panel",
        "training_config",
        "model_anchor_manifest",
        "exact_merge_evidence",
    ):
        expected = config.get(f"{key}_sha256")
        if not isinstance(expected, str) or _sha256_file(paths[key]) != expected:
            raise Core8LooSmokeError(f"config-bound {key} SHA-256 drift")
    return config, paths


def _render_block(topic: str, analysis: str) -> str:
    return (
        f"--- LOO-INDICATOR-BLOCK: {topic} ---\n"
        f"{analysis.strip()}\n"
        f"--- END-LOO-INDICATOR-BLOCK: {topic} ---"
    )


def _render_analysis(blocks: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(_render_block(topic, text) for topic, text in blocks)


def _user_prompt(analysis: str) -> str:
    payload = json.dumps(
        {"analysis": analysis},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "Rewrite the following analysis as formal FOMC Minutes prose:\n\n" + payload


def _response_wrapper(reference: str) -> str:
    return "Reference transport wrapper.\n</think>\n" + reference.strip()


def _chat_prompt_ids(
    tokenizer: Any, *, prompt: str, system_prompt: str, suffix: str | None
) -> list[int]:
    row = {"prompt": prompt}
    try:
        return list(
            native_probe._prompt_ids(
                tokenizer,
                native_probe._messages(
                    row,
                    system_prompt=system_prompt,
                    user_prompt_suffix=suffix,
                ),
            )
        )
    except native_probe.Chk3ProbeError as exc:
        raise Core8LooSmokeError(str(exc)) from exc


def _token_matched_neutral(
    *,
    tokenizer: Any,
    topics: Sequence[str],
    analyses: Mapping[str, str],
    target_topic: str,
    system_prompt: str,
    suffix: str | None,
    target_chat_tokens: int,
) -> tuple[str, str, int]:
    stems = (
        "Analysis withheld.",
        "Information unavailable.",
        "Neutral information withheld.",
        "No additional information is available.",
    )
    padding_units = (" neutral", " withheld", " information", " data")

    def candidate(stem: str, unit: str, repetitions: int) -> tuple[str, str, int]:
        replacement = stem + unit * repetitions
        blocks = [
            (topic, replacement if topic == target_topic else analyses[topic])
            for topic in topics
        ]
        combined = _render_analysis(blocks)
        prompt = _user_prompt(combined)
        count = len(
            _chat_prompt_ids(
                tokenizer,
                prompt=prompt,
                system_prompt=system_prompt,
                suffix=suffix,
            )
        )
        return replacement, combined, count

    maximum = 512
    for stem in stems:
        for unit in padding_units:
            for repetitions in loo_prompt_builder._candidate_k_values(
                lambda value: candidate(stem, unit, value)[2],
                target=target_chat_tokens,
                maximum=maximum,
            ):
                replacement, combined, count = candidate(stem, unit, repetitions)
                if count == target_chat_tokens:
                    return replacement, combined, count
    raise Core8LooSmokeError(
        f"cannot construct token-matched neutral replacement for {target_topic}"
    )


def _select_meetings(
    rows: Sequence[Mapping[str, Any]], bins: Sequence[Sequence[int]]
) -> list[str]:
    meetings: dict[str, int] = {}
    for row in rows:
        meeting_id = row.get("meeting_id")
        date = row.get("meeting_start_date")
        if not isinstance(meeting_id, str) or not isinstance(date, str):
            raise Core8LooSmokeError("Core8 row has no meeting identity")
        year = int(date[:4])
        if meeting_id in meetings and meetings[meeting_id] != year:
            raise Core8LooSmokeError("meeting year drift in Core8")
        meetings[meeting_id] = year
    selected: list[str] = []
    for raw_bin in bins:
        if (
            not isinstance(raw_bin, Sequence)
            or len(raw_bin) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_bin)
        ):
            raise Core8LooSmokeError("invalid meeting-selection bin")
        start, end = int(raw_bin[0]), int(raw_bin[1])
        candidates = [mid for mid, year in meetings.items() if start <= year <= end]
        if not candidates:
            raise Core8LooSmokeError(f"empty meeting-selection bin: {start}-{end}")
        selected.append(
            min(
                candidates,
                key=lambda mid: (_sha256_text(f"{EVALUATION_ID}|{start}-{end}|{mid}"), mid),
            )
        )
    if len(selected) != EXPECTED_MEETINGS or len(set(selected)) != EXPECTED_MEETINGS:
        raise Core8LooSmokeError(
            f"selected meeting inventory is not N{EXPECTED_MEETINGS}"
        )
    return selected


def prepare_inputs(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config, paths = _load_config(config_path)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise Core8LooSmokeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        release = _read_json(paths["source_release_manifest"])
        release_payload = validate_manifest_integrity(release)
        if (
            release.get("schema_version") != "chk3-external-evaluation-release-v1"
            or release.get("status") != "passed"
            or release.get("meeting_count") != 128
            or release.get("core8_rows") != 1024
            or release.get("evaluation_only") is not True
            or release.get("trainable") is not False
            or release.get("checkpoint_selection_allowed") is not False
        ):
            raise Core8LooSmokeError("external release scope/count contract is invalid")
        panel_record = (release.get("files") or {}).get("panels/core8.jsonl")
        if not isinstance(panel_record, Mapping) or (
            panel_record.get("sha256") != _sha256_file(paths["source_panel"])
            or panel_record.get("rows") != 1024
        ):
            raise Core8LooSmokeError("release Core8 panel binding drift")
        panel_rows = _read_jsonl(paths["source_panel"])
        if len(panel_rows) != 1024:
            raise Core8LooSmokeError("Core8 panel is not N1024")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(paths["model"]), local_files_only=True, trust_remote_code=True
        )
        try:
            system_prompt, suffix, training_config_sha = (
                native_probe._load_prompt_contract(paths["training_config"])
            )
        except native_probe.Chk3ProbeError as exc:
            raise Core8LooSmokeError(str(exc)) from exc
        if training_config_sha != config["training_config_sha256"]:
            raise Core8LooSmokeError("training prompt config SHA drift")

        selection = config["meeting_selection"]
        selected_meetings = _select_meetings(panel_rows, selection["bins"])
        topics = [str(value) for value in config["topics"]]
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in panel_rows:
            meeting_id = str(row["meeting_id"])
            topic = str(row["topic"])
            if meeting_id not in selected_meetings:
                continue
            if topic in grouped[meeting_id]:
                raise Core8LooSmokeError(f"duplicate Core8 topic: {meeting_id}:{topic}")
            grouped[meeting_id][topic] = dict(row)
        if set(grouped) != set(selected_meetings) or any(
            set(grouped[mid]) != set(topics) for mid in selected_meetings
        ):
            raise Core8LooSmokeError("selected meetings do not have complete Core8 panels")

        dataset_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        neutral_proofs: list[dict[str, Any]] = []
        for meeting_rank, meeting_id in enumerate(selected_meetings):
            meeting_rows = grouped[meeting_id]
            analyses = {topic: str(meeting_rows[topic]["source_analysis"]) for topic in topics}
            references = {topic: str(meeting_rows[topic]["reference_minutes"]) for topic in topics}
            full_analysis = _render_analysis([(topic, analyses[topic]) for topic in topics])
            full_reference = " ".join(references[topic].strip() for topic in topics)
            full_prompt = _user_prompt(full_analysis)
            full_chat_tokens = len(
                _chat_prompt_ids(
                    tokenizer,
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    suffix=suffix,
                )
            )
            full_analysis_sha = _sha256_text(full_analysis)
            full_reference_sha = _sha256_text(full_reference)
            variants: list[tuple[str, str | None, str, str, dict[str, Any]]] = [
                ("full", None, full_analysis, full_prompt, {})
            ]
            for topic in topics:
                deletion_analysis = _render_analysis(
                    [(other, analyses[other]) for other in topics if other != topic]
                )
                variants.append(
                    (
                        "exact_deletion",
                        topic,
                        deletion_analysis,
                        _user_prompt(deletion_analysis),
                        {},
                    )
                )
                replacement, neutral_analysis, neutral_count = _token_matched_neutral(
                    tokenizer=tokenizer,
                    topics=topics,
                    analyses=analyses,
                    target_topic=topic,
                    system_prompt=system_prompt,
                    suffix=suffix,
                    target_chat_tokens=full_chat_tokens,
                )
                neutral_proof = {
                    "meeting_id": meeting_id,
                    "topic": topic,
                    "replacement_sha256": _sha256_text(replacement),
                    "full_chat_prompt_tokens": full_chat_tokens,
                    "neutral_chat_prompt_tokens": neutral_count,
                    "label_preserved": True,
                    "numeric_surfaces_removed": not bool(
                        native_probe._numeric_values(replacement)
                    ),
                    "date_values_removed": not bool(native_probe._date_values(replacement)),
                }
                if (
                    neutral_count != full_chat_tokens
                    or not neutral_proof["numeric_surfaces_removed"]
                    or not neutral_proof["date_values_removed"]
                ):
                    raise Core8LooSmokeError("neutral replacement proof failed")
                neutral_proofs.append(neutral_proof)
                variants.append(
                    (
                        "token_matched_neutral",
                        topic,
                        neutral_analysis,
                        _user_prompt(neutral_analysis),
                        neutral_proof,
                    )
                )
            if len(variants) != 17:
                raise Core8LooSmokeError("meeting variant count is not 17")
            for variant_rank, (arm, topic, source_analysis, prompt, proof) in enumerate(variants):
                prompt_tokens = len(
                    _chat_prompt_ids(
                        tokenizer,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        suffix=suffix,
                    )
                )
                if prompt_tokens + int(config["decoding"]["max_new_tokens"]) > 4096:
                    raise Core8LooSmokeError(
                        f"context budget exceeds CHK3 training length: {meeting_id}:{arm}:{topic}"
                    )
                topic_slug = topic or "none"
                sample_id = f"core8-loo::{meeting_id}::{arm}::{topic_slug}"
                response = _response_wrapper(full_reference)
                input_row = {
                    "schema_version": INPUT_ROW_SCHEMA,
                    "sample_id": sample_id,
                    "meeting_id": meeting_id,
                    "meeting_start_date": meeting_rows[topics[0]]["meeting_start_date"],
                    "meeting_end_date": meeting_rows[topics[0]]["meeting_end_date"],
                    "meeting_rank": meeting_rank,
                    "variant_rank": variant_rank,
                    "arm": arm,
                    "intervention_topic": topic,
                    "prompt": prompt,
                    "response": response,
                    "source_analysis": source_analysis,
                    "full_source_analysis_sha256": full_analysis_sha,
                    "full_reference_sha256": full_reference_sha,
                    "source_analysis_sha256": _sha256_text(source_analysis),
                    "prompt_sha256": _sha256_text(prompt),
                    "reference_minutes": full_reference,
                    "reference_minutes_sha256": full_reference_sha,
                    "prompt_token_count": prompt_tokens,
                    "full_prompt_token_count": full_chat_tokens,
                    "neutral_proof": proof or None,
                }
                dataset_rows.append(input_row)
                sample_rows.append(
                    {
                        key: input_row[key]
                        for key in (
                            "sample_id",
                            "meeting_id",
                            "meeting_start_date",
                            "meeting_rank",
                            "variant_rank",
                            "arm",
                            "intervention_topic",
                            "prompt_sha256",
                            "source_analysis_sha256",
                            "full_source_analysis_sha256",
                            "reference_minutes_sha256",
                            "prompt_token_count",
                            "full_prompt_token_count",
                        )
                    }
                    | {
                        "length_bucket": "core8_combined",
                        "analysis_reference_exact_identity": source_analysis
                        == full_reference,
                        "normalized_identity": False,
                        "punctuation_insensitive_identity": False,
                    }
                )
        if len(dataset_rows) != EXPECTED_ROWS or len(sample_rows) != EXPECTED_ROWS:
            raise Core8LooSmokeError(
                f"prepared evaluation matrix is not N{EXPECTED_ROWS}"
            )
        if len({row["sample_id"] for row in dataset_rows}) != EXPECTED_ROWS:
            raise Core8LooSmokeError("prepared smoke sample IDs are not unique")

        dataset_path = output_dir / "inputs.jsonl"
        _write_new_jsonl(dataset_path, dataset_rows)
        anchor = _read_json(paths["model_anchor_manifest"])
        anchor_payload = validate_manifest_integrity(anchor)
        if (
            anchor.get("stage_id") != "chk3"
            or anchor.get("model_label") != "chk3-cp318-exact-merged"
            or anchor.get("adapter") is not None
            or anchor.get("model", {}).get("path") != str(paths["model"])
        ):
            raise Core8LooSmokeError("CHK3 cp318 model anchor drift")
        manifest = seal_manifest(
            {
                "schema_version": SAMPLE_SCHEMA,
                "status": "complete",
                "created_at_utc": _utc_now(),
                "evaluation_id": EVALUATION_ID,
                "task_contract_id": TASK_CONTRACT_ID,
                "config": _file_binding(config_path),
                "source_release": _file_binding(
                    paths["source_release_manifest"], sealed=True
                ),
                "source_release_payload_sha256": release_payload,
                "source_panel": _file_binding(paths["source_panel"], rows=1024),
                "model_anchor": _file_binding(
                    paths["model_anchor_manifest"], sealed=True
                ),
                "model_anchor_payload_sha256": anchor_payload,
                "exact_merge_evidence": _file_binding(paths["exact_merge_evidence"], sealed=True),
                "prompt_contract": {
                    "training_config": str(paths["training_config"]),
                    "training_config_sha256": training_config_sha,
                    "system_prompt_sha256": _sha256_text(system_prompt),
                    "user_prompt_suffix_sha256": _sha256_text(suffix) if suffix else None,
                    "wrapper": "native-analysis-json-with-fixed-core8-blocks-v1",
                },
                "tokenizer": native_probe._tokenizer_fingerprint(paths["model"]),
                "generation_contract": dict(config["decoding"]),
                "selection": {
                    "algorithm": selection["algorithm"],
                    "bins": selection["bins"],
                    "meeting_ids": selected_meetings,
                    "meetings": EXPECTED_MEETINGS,
                    "topics": topics,
                    "variants_per_meeting": 17,
                    "rows": EXPECTED_ROWS,
                },
                "dataset": _file_binding(dataset_path, rows=EXPECTED_ROWS),
                "neutral_replacement_proofs": neutral_proofs,
                "samples": sample_rows,
                "limitations": list(PREPARE_LIMITATIONS),
            }
        )
        manifest_path = output_dir / "samples.json"
        _write_new_json(manifest_path, manifest)
        return {
            "status": "prepared",
            "manifest": _file_binding(manifest_path, sealed=True),
            "dataset": _file_binding(dataset_path, rows=EXPECTED_ROWS),
            "selected_meetings": selected_meetings,
            "rows": EXPECTED_ROWS,
            "prompt_tokens": {
                "min": min(row["prompt_token_count"] for row in dataset_rows),
                "max": max(row["prompt_token_count"] for row in dataset_rows),
            },
        }
    except Exception:
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def _load_sample_manifest(
    path: Path, expected_sha256: str, *, load_tokenizer: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]], Any | None]:
    resolved = path.expanduser().resolve()
    if _sha256_file(resolved) != expected_sha256:
        raise Core8LooSmokeError("sample manifest SHA-256 mismatch")
    manifest = _read_json(resolved)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != SAMPLE_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("task_contract_id") != TASK_CONTRACT_ID
    ):
        raise Core8LooSmokeError("sample manifest identity drift")
    samples = manifest.get("samples")
    selection = manifest.get("selection")
    generation_contract = manifest.get("generation_contract")
    if (
        not isinstance(samples, list)
        or len(samples) != EXPECTED_ROWS
        or not isinstance(selection, Mapping)
        or selection.get("rows") != EXPECTED_ROWS
        or selection.get("meetings") != EXPECTED_MEETINGS
        or selection.get("variants_per_meeting") != 17
        or not isinstance(generation_contract, Mapping)
        or generation_contract.get("generation_mode") != "greedy"
        or generation_contract.get("do_sample") is not False
        or generation_contract.get("seed") != 20260814
        or generation_contract.get("max_new_tokens") != 2560
        or generation_contract.get("tail_tokens") != 1024
        or generation_contract.get("physical_gpu_index") != 0
    ):
        raise Core8LooSmokeError("sample manifest matrix drift")
    dataset_binding = manifest.get("dataset")
    if not isinstance(dataset_binding, Mapping):
        raise Core8LooSmokeError("sample manifest has no dataset binding")
    dataset_path = Path(str(dataset_binding.get("path"))).expanduser().resolve()
    if (
        _sha256_file(dataset_path) != dataset_binding.get("sha256")
        or dataset_binding.get("rows") != EXPECTED_ROWS
    ):
        raise Core8LooSmokeError("bound input dataset changed")
    rows = _read_jsonl(dataset_path)
    if len(rows) != EXPECTED_ROWS:
        raise Core8LooSmokeError("bound input dataset row count drift")
    if [row.get("sample_id") for row in rows] != [
        sample.get("sample_id") for sample in samples
    ]:
        raise Core8LooSmokeError("sample/dataset ordering drift")
    for binding_name in (
        "source_release",
        "source_panel",
        "model_anchor",
        "exact_merge_evidence",
    ):
        binding = manifest.get(binding_name)
        if not isinstance(binding, Mapping):
            raise Core8LooSmokeError(f"sample manifest has no {binding_name} binding")
        bound_path = Path(str(binding.get("path"))).expanduser().resolve()
        if _sha256_file(bound_path) != binding.get("sha256"):
            raise Core8LooSmokeError(f"bound {binding_name} changed")
    tokenizer = None
    if load_tokenizer:
        from transformers import AutoTokenizer

        tokenizer_binding = manifest.get("tokenizer")
        if not isinstance(tokenizer_binding, Mapping):
            raise Core8LooSmokeError("sample manifest tokenizer binding missing")
        tokenizer_path = Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
        files = tokenizer_binding.get("files")
        if not isinstance(files, Mapping) or not files:
            raise Core8LooSmokeError("sample tokenizer file inventory missing")
        for name, digest in files.items():
            if _sha256_file(tokenizer_path / str(name)) != digest:
                raise Core8LooSmokeError(f"bound tokenizer changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
        prompt_contract = manifest.get("prompt_contract")
        if not isinstance(prompt_contract, Mapping):
            raise Core8LooSmokeError("sample prompt contract missing")
        try:
            system, suffix, config_sha = native_probe._load_prompt_contract(
                Path(str(prompt_contract["training_config"]))
            )
        except native_probe.Chk3ProbeError as exc:
            raise Core8LooSmokeError(str(exc)) from exc
        if (
            config_sha != prompt_contract.get("training_config_sha256")
            or _sha256_text(system) != prompt_contract.get("system_prompt_sha256")
            or (_sha256_text(suffix) if suffix else None)
            != prompt_contract.get("user_prompt_suffix_sha256")
        ):
            raise Core8LooSmokeError("bound prompt contract changed")
        for sample, row in zip(samples, rows, strict=True):
            if (
                _sha256_text(str(row["prompt"])) != sample.get("prompt_sha256")
                or _sha256_text(str(row["source_analysis"]))
                != sample.get("source_analysis_sha256")
                or _sha256_text(str(row["reference_minutes"]))
                != sample.get("reference_minutes_sha256")
            ):
                raise Core8LooSmokeError("prepared input text/hash drift")
            count = len(
                _chat_prompt_ids(
                    tokenizer,
                    prompt=str(row["prompt"]),
                    system_prompt=system,
                    suffix=suffix,
                )
            )
            if count != sample.get("prompt_token_count"):
                raise Core8LooSmokeError("prepared prompt token-count drift")
            if row.get("arm") == "token_matched_neutral" and count != sample.get(
                "full_prompt_token_count"
            ):
                raise Core8LooSmokeError("neutral prompt is no longer token matched")
    manifest["_validated_payload_sha256"] = payload_sha
    manifest["_validated_path"] = str(resolved)
    return manifest, rows, tokenizer


def _state(
    *,
    status: str,
    completed: int,
    expected: int,
    resume_count: int,
    result_path: Path | None,
    **extra: Any,
) -> dict[str, Any]:
    return seal_manifest(
        {
            "schema_version": STATE_SCHEMA,
            "status": status,
            "updated_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "completed_rows": completed,
            "expected_rows": expected,
            "resume_count": resume_count,
            "results": (
                _file_binding(result_path, rows=completed)
                if result_path is not None and result_path.is_file()
                else None
            ),
            **extra,
        }
    )


def _validate_result_row(
    row: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    input_row: Mapping[str, Any],
    sample_sha256: str,
    tokenizer: Any | None,
) -> None:
    if row.get("loo_smoke_evaluation_id") != EVALUATION_ID:
        raise Core8LooSmokeError("generation row evaluation identity drift")
    for key in (
        "meeting_id",
        "arm",
        "intervention_topic",
        "full_source_analysis_sha256",
        "full_reference_sha256",
        "full_prompt_token_count",
    ):
        if row.get(key) != input_row.get(key):
            raise Core8LooSmokeError(f"generation row LOO identity drift at {key}")
    try:
        native_eval.validate_full_result(
            row,
            stage_id="chk3",
            model_label=MODEL_LABEL,
            sample_manifest_sha256=sample_sha256,
            sample=sample,
            source_analysis=str(input_row["source_analysis"]),
            reference_response=str(input_row["response"]),
            tokenizer=tokenizer,
        )
    except native_eval.NativeThreeModelEvalError as exc:
        raise Core8LooSmokeError(str(exc)) from exc
    expected_seed = derive_row_seed(20260814, str(sample["sample_id"]))
    if row.get("seed") != expected_seed:
        raise Core8LooSmokeError("generation row seed drift")


def _run_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gates = [semantic_shared._recompute_core_hard_gate(row) for row in rows]
    by_arm: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_arm[str(row["arm"])].append(index)

    def summarize(indices: Sequence[int]) -> dict[str, Any]:
        selected_rows = [rows[index] for index in indices]
        selected_gates = [gates[index] for index in indices]
        return {
            "rows": len(indices),
            "structure_delivery_rate": statistics.fmean(
                float(gate["delivery_valid"] and gate["native_structure_valid"])
                for gate in selected_gates
            ),
            "numeric_fidelity_rate": statistics.fmean(
                float(gate["numeric_multiset_preserved"]) for gate in selected_gates
            ),
            "date_fidelity_rate": statistics.fmean(
                float(gate["date_set_preserved"]) for gate in selected_gates
            ),
            "degeneration_free_rate": statistics.fmean(
                float(gate["degeneration_free"]) for gate in selected_gates
            ),
            "eos_rate": statistics.fmean(float(row["hit_eos"]) for row in selected_rows),
            "cap_rows": sum(bool(row["cap_reached"]) for row in selected_rows),
            "periodic_tail_rows": sum(
                bool(row["strict_periodic_tail"]) for row in selected_rows
            ),
            "mean_generated_tokens": statistics.fmean(
                int(row["content_tokens"]) for row in selected_rows
            ),
        }

    return {
        "overall": summarize(list(range(len(rows)))),
        "by_arm": {arm: summarize(indices) for arm, indices in sorted(by_arm.items())},
        "failure_counts": dict(
            sorted(
                Counter(
                    failure
                    for gate in gates
                    for failure in gate["preregistered_core_failures"]
                ).items()
            )
        ),
    }


def run_generation(
    *,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    resume: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
) -> dict[str, Any]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise Core8LooSmokeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible != ["0"]:
        raise Core8LooSmokeError("generation requires CUDA_VISIBLE_DEVICES=0")
    manifest, input_rows, tokenizer = _load_sample_manifest(
        sample_manifest_path, sample_manifest_sha256, load_tokenizer=True
    )
    assert tokenizer is not None
    generation_contract = manifest["generation_contract"]
    max_new_tokens = int(generation_contract["max_new_tokens"])
    tail_tokens = int(generation_contract["tail_tokens"])
    base_seed = int(generation_contract["seed"])
    samples = manifest["samples"]
    output_dir = output_dir.expanduser().resolve()
    if output_dir.is_symlink():
        raise Core8LooSmokeError("generation output directory is a symlink")
    exists = output_dir.exists()
    if exists and not resume:
        raise Core8LooSmokeError(f"generation output already exists: {output_dir}")
    if not exists:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / ".partial").mkdir()
    partial_path = output_dir / ".partial/generations.progress.v1.jsonl"
    final_path = output_dir / "generations.jsonl"
    launch_path = output_dir / "launch.json"
    state_path = output_dir / "state.progress.v1.json"
    run_manifest_path = output_dir / "manifest.json"
    anchor_path = Path(str(manifest["model_anchor"]["path"]))
    anchor = _read_json(anchor_path)
    model_path = Path(str(anchor["model"]["path"])).resolve()
    model_fingerprint = native_probe._fingerprint_model(model_path)
    if model_fingerprint != anchor.get("model"):
        raise Core8LooSmokeError("runtime CHK3 model differs from sealed cp318 anchor")
    launch_payload = seal_manifest(
        {
            "schema_version": RUN_SCHEMA,
            "status": "initializing",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "task_contract_id": TASK_CONTRACT_ID,
            "stage_id": "chk3",
            "model_label": MODEL_LABEL,
            "model": model_fingerprint,
            "model_anchor": dict(manifest["model_anchor"]),
            "sample_manifest": {
                "path": str(sample_manifest_path.expanduser().resolve()),
                "sha256": sample_manifest_sha256,
                "payload_sha256": manifest["_validated_payload_sha256"],
            },
            "generation": {
                "mode": "greedy",
                "do_sample": False,
                "base_seed": base_seed,
                "row_seed": "derive_row_seed(base_seed,sample_id)",
                "max_new_tokens": max_new_tokens,
                "tail_tokens": tail_tokens,
                "load_in_4bit": True,
                "quantization": "nf4-double-quant-bfloat16",
                "attn_implementation": "sdpa",
                "batch_size": 1,
            },
            "runtime": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "cuda_visible_devices": visible,
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                "physical_gpu_index": 0,
            },
            "persistence": {
                "full_text": True,
                "answer": True,
                "generated_token_ids": True,
                "append_flush_fsync_per_row": True,
                "resume": True,
            },
        }
    )
    results: list[dict[str, Any]] = []
    resume_count = 0
    if exists:
        if final_path.exists() or run_manifest_path.exists():
            raise Core8LooSmokeError("cannot resume a finalized generation run")
        prior_launch = _read_json(launch_path)
        validate_manifest_integrity(prior_launch)
        for key in (
            "schema_version",
            "evaluation_id",
            "task_contract_id",
            "stage_id",
            "model_label",
            "model",
            "model_anchor",
            "sample_manifest",
            "generation",
            "runtime",
            "persistence",
        ):
            if prior_launch.get(key) != launch_payload.get(key):
                raise Core8LooSmokeError(f"resume launch contract drift at {key}")
        if not partial_path.is_file() or partial_path.is_symlink():
            raise Core8LooSmokeError("resume partial generation artifact is missing")
        results = _read_jsonl(partial_path)
        if len(results) >= EXPECTED_ROWS:
            raise Core8LooSmokeError("resume prefix is not a strict incomplete prefix")
        prior_state = _read_json(state_path)
        validate_manifest_integrity(prior_state)
        if prior_state.get("completed_rows") != len(results) or prior_state.get(
            "results", {}
        ).get("sha256") != _sha256_file(partial_path):
            raise Core8LooSmokeError("resume state/partial binding drift")
        resume_count = int(prior_state.get("resume_count", 0)) + 1
    else:
        _write_new_json(launch_path, launch_payload)

    prompt_contract = manifest["prompt_contract"]
    try:
        system_prompt, suffix, _ = native_probe._load_prompt_contract(
            Path(str(prompt_contract["training_config"]))
        )
    except native_probe.Chk3ProbeError as exc:
        raise Core8LooSmokeError(str(exc)) from exc
    for index, result in enumerate(results):
        _validate_result_row(
            result,
            sample=samples[index],
            input_row=input_rows[index],
            sample_sha256=sample_manifest_sha256,
            tokenizer=tokenizer,
        )

    def on_wait(status: str, processes: Sequence[Mapping[str, Any]]) -> None:
        print(
            _canonical_json(
                {"status": status, "processes": list(processes), "gpu": 0}
            ),
            flush=True,
        )

    try:
        with gpu_guard.exclusive_gpu0_lease(
            lock_path=GPU_LOCK,
            timeout_seconds=gpu_wait_timeout_seconds,
            poll_seconds=gpu_poll_seconds,
            on_wait=on_wait,
        ) as lease:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise Core8LooSmokeError("exactly one visible CUDA GPU is required")
            physical_identity = gpu_guard._physical_gpu0_identity()
            logical_uuid = gpu_guard._verify_logical_cuda0_is_physical_gpu0(
                torch, physical_identity
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                quantization_config=quantization,
                device_map={"": 0},
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model.eval()
            model.config.use_cache = True
            eos_value = model.generation_config.eos_token_id
            if eos_value is None:
                eos_value = tokenizer.eos_token_id
            eos_ids = native_probe.common_probe._normalize_eos_ids(eos_value)
            if not eos_ids:
                raise Core8LooSmokeError("model/tokenizer has no EOS token")
            for result_index, result in enumerate(results):
                if result.get("eos_token_ids") != sorted(eos_ids):
                    raise Core8LooSmokeError("resume EOS contract drift")
            torch.cuda.reset_peak_memory_stats()
            with partial_path.open("a" if exists else "x", encoding="utf-8") as handle:
                for index in range(len(results), EXPECTED_ROWS):
                    external = [
                        process
                        for process in gpu_guard._external_gpu0_compute_processes()
                        if int(process.get("pid", -1)) != os.getpid()
                    ]
                    if external:
                        raise Core8LooSmokeError(
                            "external GPU0 process appeared after model load"
                        )
                    sample = samples[index]
                    input_row = input_rows[index]
                    prompt_ids = _chat_prompt_ids(
                        tokenizer,
                        prompt=str(input_row["prompt"]),
                        system_prompt=system_prompt,
                        suffix=suffix,
                    )
                    if len(prompt_ids) != sample["prompt_token_count"]:
                        raise Core8LooSmokeError("runtime prompt token-count drift")
                    context_limit = int(
                        getattr(model.config, "max_position_embeddings", 0) or 0
                    )
                    if context_limit and len(prompt_ids) + max_new_tokens > context_limit:
                        raise Core8LooSmokeError("runtime context overflow")
                    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
                    attention_mask = torch.ones_like(input_ids)
                    row_seed = derive_row_seed(base_seed, str(sample["sample_id"]))
                    transformers.set_seed(row_seed)
                    torch.manual_seed(row_seed)
                    torch.cuda.manual_seed_all(row_seed)
                    with torch.inference_mode():
                        sequences = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=sorted(eos_ids),
                            use_cache=True,
                        )
                    generated_ids = sequences[0, input_ids.shape[1] :].tolist()
                    text = native_probe.common_probe.decode_completion_preserving_boundary(
                        tokenizer, generated_ids, eos_ids
                    )
                    try:
                        result = native_eval.build_full_result(
                            text=text,
                            generated_token_ids=generated_ids,
                            eos_token_ids=eos_ids,
                            max_new_tokens=max_new_tokens,
                            tail_tokens=tail_tokens,
                            source_prompt=str(input_row["prompt"]),
                            source_analysis=str(input_row["source_analysis"]),
                            reference_response=str(input_row["response"]),
                            load_in_4bit=True,
                            attn_implementation="sdpa",
                            pad_token_id=int(tokenizer.pad_token_id),
                            stage_id="chk3",
                            model_label=MODEL_LABEL,
                            sample_manifest_sha256=sample_manifest_sha256,
                            sample=sample,
                            seed=row_seed,
                        )
                    except native_eval.NativeThreeModelEvalError as exc:
                        raise Core8LooSmokeError(str(exc)) from exc
                    result.update(
                        {
                            "loo_smoke_evaluation_id": EVALUATION_ID,
                            "meeting_id": input_row["meeting_id"],
                            "arm": input_row["arm"],
                            "intervention_topic": input_row["intervention_topic"],
                            "full_source_analysis_sha256": input_row[
                                "full_source_analysis_sha256"
                            ],
                            "full_reference_sha256": input_row["full_reference_sha256"],
                            "full_prompt_token_count": input_row[
                                "full_prompt_token_count"
                            ],
                            "input_truncated": False,
                        }
                    )
                    _validate_result_row(
                        result,
                        sample=sample,
                        input_row=input_row,
                        sample_sha256=sample_manifest_sha256,
                        tokenizer=tokenizer,
                    )
                    handle.write(_canonical_json(result) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    results.append(result)
                    _atomic_write_json(
                        state_path,
                        _state(
                            status="generating",
                            completed=len(results),
                            expected=EXPECTED_ROWS,
                            resume_count=resume_count,
                            result_path=partial_path,
                            last_sample_id=sample["sample_id"],
                        ),
                    )
                    print(
                        _canonical_json(
                            {
                                "status": "generating",
                                "completed": len(results),
                                "expected": EXPECTED_ROWS,
                                "sample_id": sample["sample_id"],
                            }
                        ),
                        flush=True,
                    )
                    del input_ids, attention_mask, sequences
            os.replace(partial_path, final_path)
            _fsync_directory(output_dir)
            peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            peak_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            runtime = {
                "physical_gpu_index": 0,
                "physical_gpu_identity": physical_identity,
                "logical_cuda0_uuid": logical_uuid,
                "cuda_visible_devices": visible,
                "gpu_lease": lease,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peak_allocated_gib": peak_allocated,
                "peak_reserved_gib": peak_reserved,
                "resume_count": resume_count,
            }
            del model
            torch.cuda.empty_cache()
        summary = _run_summary(results)
        generation_complete_state = _state(
            status="generation_complete",
            completed=EXPECTED_ROWS,
            expected=EXPECTED_ROWS,
            resume_count=resume_count,
            result_path=final_path,
            summary_sha256=_sha256_text(_canonical_json(summary)),
            runtime=runtime,
        )
        _atomic_write_json(state_path, generation_complete_state)
        run_manifest = seal_manifest(
            {
                "schema_version": RUN_SCHEMA,
                "status": "complete",
                "created_at_utc": _utc_now(),
                "evaluation_id": EVALUATION_ID,
                "task_contract_id": TASK_CONTRACT_ID,
                "stage_id": "chk3",
                "model_label": MODEL_LABEL,
                "model": model_fingerprint,
                "model_anchor": dict(manifest["model_anchor"]),
                "sample_manifest": {
                    "path": str(sample_manifest_path.expanduser().resolve()),
                    "sha256": sample_manifest_sha256,
                    "payload_sha256": manifest["_validated_payload_sha256"],
                },
                "generation_contract": launch_payload["generation"],
                "runtime": runtime,
                "artifacts": {
                    "launch": _file_binding(launch_path, sealed=True),
                    "generations": _file_binding(final_path, rows=EXPECTED_ROWS),
                    "generation_complete_state": _file_binding(state_path, sealed=True),
                },
                "summary": summary,
                "limitations": list(RUN_LIMITATIONS),
            }
        )
        _write_new_json(run_manifest_path, run_manifest)
        return run_manifest
    except Exception as exc:
        if not final_path.exists():
            _atomic_write_json(
                state_path,
                _state(
                    status="failed",
                    completed=len(results),
                    expected=EXPECTED_ROWS,
                    resume_count=resume_count,
                    result_path=partial_path if partial_path.is_file() else None,
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            )
        raise


def load_and_validate_run(
    *,
    run_manifest_path: Path,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
) -> dict[str, Any]:
    run_manifest_path = run_manifest_path.expanduser().resolve()
    run = _read_json(run_manifest_path)
    payload_sha = validate_manifest_integrity(run)
    if (
        run.get("schema_version") != RUN_SCHEMA
        or run.get("status") != "complete"
        or run.get("evaluation_id") != EVALUATION_ID
        or run.get("stage_id") != "chk3"
    ):
        raise Core8LooSmokeError("generation run manifest identity drift")
    manifest, inputs, tokenizer = _load_sample_manifest(
        sample_manifest_path, sample_manifest_sha256, load_tokenizer=True
    )
    assert tokenizer is not None
    binding = run.get("sample_manifest")
    if (
        not isinstance(binding, Mapping)
        or Path(str(binding.get("path"))).resolve()
        != sample_manifest_path.expanduser().resolve()
        or binding.get("sha256") != sample_manifest_sha256
        or binding.get("payload_sha256") != manifest["_validated_payload_sha256"]
    ):
        raise Core8LooSmokeError("generation run sample binding drift")
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Core8LooSmokeError("generation run artifact inventory missing")
    generations = artifacts.get("generations")
    state_binding = artifacts.get("generation_complete_state")
    if not isinstance(generations, Mapping) or not isinstance(state_binding, Mapping):
        raise Core8LooSmokeError("generation/state artifact binding missing")
    generation_path = Path(str(generations.get("path"))).resolve()
    state_path = Path(str(state_binding.get("path"))).resolve()
    if generation_path.parent != run_manifest_path.parent or state_path.parent != run_manifest_path.parent:
        raise Core8LooSmokeError("generation artifacts escaped the run directory")
    if (
        _sha256_file(generation_path) != generations.get("sha256")
        or generations.get("rows") != EXPECTED_ROWS
        or _sha256_file(state_path) != state_binding.get("sha256")
        or validate_manifest_integrity(_read_json(state_path))
        != state_binding.get("payload_sha256")
    ):
        raise Core8LooSmokeError("generation artifact hash drift")
    rows = _read_jsonl(generation_path)
    if len(rows) != EXPECTED_ROWS:
        raise Core8LooSmokeError("generation row count is not N136")
    for row, sample, input_row in zip(rows, manifest["samples"], inputs, strict=True):
        _validate_result_row(
            row,
            sample=sample,
            input_row=input_row,
            sample_sha256=sample_manifest_sha256,
            tokenizer=tokenizer,
        )
    summary = _run_summary(rows)
    if run.get("summary") != summary:
        raise Core8LooSmokeError("generation run summary is not reproducible")
    return {
        "manifest": run,
        "manifest_binding": _file_binding(run_manifest_path, sealed=True),
        "sample_manifest": manifest,
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "payload_sha256": payload_sha,
    }


def _metric_mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def score_run(
    *,
    run_manifest_path: Path,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    semantic_manifest_path: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise Core8LooSmokeError(f"score output already exists: {output_dir}")
    validated = load_and_validate_run(
        run_manifest_path=run_manifest_path,
        sample_manifest_path=sample_manifest_path,
        sample_manifest_sha256=sample_manifest_sha256,
    )
    try:
        bert, mpnet, semantic_provenance = semantic_shared.load_formal_semantic_backends(
            semantic_manifest_path=semantic_manifest_path,
            batch_size=batch_size,
            device=device,
        )
        work_items = [("chk3", row) for row in validated["rows"]]
        semantic_values, backend_audit = semantic_shared._score_semantic_pairs(
            work_items=work_items, bert=bert, mpnet=mpnet
        )
    except Exception as exc:
        raise Core8LooSmokeError(f"semantic scoring failed: {exc}") from exc
    if len(semantic_values) != EXPECTED_ROWS:
        raise Core8LooSmokeError("semantic scorer did not return N136")
    score_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for generation, input_row, values in zip(
        validated["rows"], validated["inputs"], semantic_values, strict=True
    ):
        gate = semantic_shared._recompute_core_hard_gate(generation)
        core_valid = bool(gate["preregistered_core_valid"])
        item = {
            "schema_version": SCORE_ROW_SCHEMA,
            "sample_id": generation["sample_id"],
            "meeting_id": generation["meeting_id"],
            "arm": generation["arm"],
            "intervention_topic": generation["intervention_topic"],
            "answer_sha256": generation["answer_sha256"],
            "reference_minutes_sha256": generation["reference_minutes_sha256"],
            "structure_delivery": bool(
                gate["delivery_valid"] and gate["native_structure_valid"]
            ),
            "numeric_fidelity": bool(gate["numeric_multiset_preserved"]),
            "date_fidelity": bool(gate["date_set_preserved"]),
            "degeneration_free": bool(gate["degeneration_free"]),
            "core_valid": core_valid,
            "core_failures": list(gate["preregistered_core_failures"]),
            "mpnet_cosine_raw": float(values["mpnet_cosine"]),
            "bertscore_f1_raw": float(values["bertscore_f1"]),
            "mpnet_cosine_zero_penalized": (
                float(values["mpnet_cosine"]) if core_valid else 0.0
            ),
            "bertscore_f1_zero_penalized": (
                float(values["bertscore_f1"]) if core_valid else 0.0
            ),
            "generated_tokens": generation["content_tokens"],
            "full_token_4gram_repetition": generation[
                "full_token_4gram_repetition"
            ],
            "tail_token_4gram_repetition": generation[
                "tail_token_4gram_repetition"
            ],
        }
        score_rows.append(item)
        if not core_valid:
            failures.append(
                {
                    "schema_version": FAILURE_ROW_SCHEMA,
                    "sample_id": item["sample_id"],
                    "meeting_id": item["meeting_id"],
                    "arm": item["arm"],
                    "intervention_topic": item["intervention_topic"],
                    "failures": item["core_failures"],
                    "completion_sha256": generation["completion_sha256"],
                }
            )
    indexed: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["meeting_id"]), str(row["sample_id"])): row for row in score_rows
    }
    full_by_meeting = {
        str(row["meeting_id"]): row for row in score_rows if row["arm"] == "full"
    }
    if len(full_by_meeting) != EXPECTED_MEETINGS:
        raise Core8LooSmokeError("score matrix has no unique full row per meeting")
    del indexed
    topic_deltas: dict[str, dict[str, Any]] = {}
    topics = validated["sample_manifest"]["selection"]["topics"]
    for topic in topics:
        topic_deltas[topic] = {}
        for arm in ("exact_deletion", "token_matched_neutral"):
            selected = [
                row
                for row in score_rows
                if row["arm"] == arm and row["intervention_topic"] == topic
            ]
            if len(selected) != EXPECTED_MEETINGS:
                raise Core8LooSmokeError(f"score matrix incomplete: {topic}:{arm}")
            mpnet_raw = [
                full_by_meeting[str(row["meeting_id"])]["mpnet_cosine_raw"]
                - row["mpnet_cosine_raw"]
                for row in selected
            ]
            bert_raw = [
                full_by_meeting[str(row["meeting_id"])]["bertscore_f1_raw"]
                - row["bertscore_f1_raw"]
                for row in selected
            ]
            mpnet_penalized = [
                full_by_meeting[str(row["meeting_id"])][
                    "mpnet_cosine_zero_penalized"
                ]
                - row["mpnet_cosine_zero_penalized"]
                for row in selected
            ]
            bert_penalized = [
                full_by_meeting[str(row["meeting_id"])][
                    "bertscore_f1_zero_penalized"
                ]
                - row["bertscore_f1_zero_penalized"]
                for row in selected
            ]
            topic_deltas[topic][arm] = {
                "meetings": EXPECTED_MEETINGS,
                "mpnet_delta_raw_mean": _metric_mean(mpnet_raw),
                "bertscore_f1_delta_raw_mean": _metric_mean(bert_raw),
                "mpnet_delta_zero_penalized_mean": _metric_mean(mpnet_penalized),
                "bertscore_f1_delta_zero_penalized_mean": _metric_mean(
                    bert_penalized
                ),
                "positive_mpnet_raw_meetings": sum(value > 0 for value in mpnet_raw),
                "positive_bert_raw_meetings": sum(value > 0 for value in bert_raw),
                "structure_delivery_rate": _metric_mean(
                    [float(row["structure_delivery"]) for row in selected]
                ),
                "numeric_fidelity_rate": _metric_mean(
                    [float(row["numeric_fidelity"]) for row in selected]
                ),
                "date_fidelity_rate": _metric_mean(
                    [float(row["date_fidelity"]) for row in selected]
                ),
                "degeneration_free_rate": _metric_mean(
                    [float(row["degeneration_free"]) for row in selected]
                ),
            }
    gate_config = _read_json(Path(str(validated["sample_manifest"]["config"]["path"])))
    overall = validated["summary"]["overall"]
    gates = gate_config["quality_gates"]
    operational_failures: list[str] = []
    if overall["cap_rows"] != gates["completion_cap_rows"]:
        operational_failures.append("completion_cap_rows_nonzero")
    if overall["periodic_tail_rows"] != gates["strict_periodic_tail_rows"]:
        operational_failures.append("strict_periodic_tail_rows_nonzero")
    if overall["structure_delivery_rate"] < gates["minimum_structure_delivery_rate"]:
        operational_failures.append("structure_delivery_rate_below_0.99")
    diagnostic_only = gates.get("use") == "diagnostic_only"
    verdict = (
        "descriptive_complete"
        if diagnostic_only
        else ("passed" if not operational_failures else "failed")
    )
    summary = {
        "status": verdict,
        "evaluation_id": EVALUATION_ID,
        "scope": EVALUATION_SCOPE,
        "generation_gate_use": (
            "diagnostic_only" if diagnostic_only else "operational_verdict"
        ),
        "rows": EXPECTED_ROWS,
        "meetings": EXPECTED_MEETINGS,
        "topics": EXPECTED_TOPICS,
        "operational_gate_failures": operational_failures,
        "generation": validated["summary"],
        "semantic": {
            "mpnet_cosine_raw_mean": _metric_mean(
                [row["mpnet_cosine_raw"] for row in score_rows]
            ),
            "bertscore_f1_raw_mean": _metric_mean(
                [row["bertscore_f1_raw"] for row in score_rows]
            ),
            "mpnet_cosine_zero_penalized_mean": _metric_mean(
                [row["mpnet_cosine_zero_penalized"] for row in score_rows]
            ),
            "bertscore_f1_zero_penalized_mean": _metric_mean(
                [row["bertscore_f1_zero_penalized"] for row in score_rows]
            ),
        },
        "topic_deltas": topic_deltas,
        "failure_rows": len(failures),
        "interpretation": (
            "Positive delta means the intervention reduced alignment to the frozen "
            "full Core8 reference. This is descriptive stage-local sensitivity, "
            "not causal attribution."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    row_path = output_dir / "row_scores.jsonl"
    failure_path = output_dir / "failure_rows.jsonl"
    summary_path = output_dir / "summary.json"
    _write_new_jsonl(row_path, score_rows)
    _write_new_jsonl(failure_path, failures)
    _write_new_json(summary_path, summary)
    score_manifest = seal_manifest(
        {
            "schema_version": SCORE_SCHEMA,
            "status": "complete",
            "result_status": verdict,
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "run_manifest": validated["manifest_binding"],
            "sample_manifest": _file_binding(sample_manifest_path, sealed=True),
            "semantic_manifest": _file_binding(semantic_manifest_path, sealed=True),
            "semantic_models": semantic_provenance,
            "semantic_backend_audit": backend_audit,
            "execution": {
                "device": device,
                "batch_size": batch_size,
                "python": sys.executable,
                "network": False,
            },
            "artifacts": {
                "row_scores": _file_binding(row_path, rows=EXPECTED_ROWS),
                "failure_rows": _file_binding(failure_path, rows=len(failures)),
                "summary": _file_binding(summary_path),
            },
            "summary": summary,
            "limitations": list(SCORE_LIMITATIONS),
        }
    )
    manifest_path = output_dir / "manifest.json"
    _write_new_json(manifest_path, score_manifest)
    return score_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--sample-manifest", type=Path, required=True)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--gpu-wait-timeout-seconds", type=int, default=172800)
    run.add_argument("--gpu-poll-seconds", type=int, default=30)
    score = subparsers.add_parser("score")
    score.add_argument("--run-manifest", type=Path, required=True)
    score.add_argument("--sample-manifest", type=Path, required=True)
    score.add_argument("--sample-manifest-sha256", required=True)
    score.add_argument("--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--batch-size", type=int, default=8)
    validate = subparsers.add_parser("validate-run")
    validate.add_argument("--run-manifest", type=Path, required=True)
    validate.add_argument("--sample-manifest", type=Path, required=True)
    validate.add_argument("--sample-manifest-sha256", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_inputs(config_path=args.config, output_dir=args.output_dir)
        elif args.command == "run":
            result = run_generation(
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output_dir=args.output_dir,
                resume=args.resume,
                gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
            )
        elif args.command == "score":
            result = score_run(
                run_manifest_path=args.run_manifest,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                semantic_manifest_path=args.semantic_manifest,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            validated = load_and_validate_run(
                run_manifest_path=args.run_manifest,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
            )
            result = {
                "status": "validated",
                "rows": len(validated["rows"]),
                "manifest": validated["manifest_binding"],
            }
        print(_canonical_json(result))
        return 0
    except Exception as exc:
        print(
            _canonical_json(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
