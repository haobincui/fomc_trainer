"""Run and compare chk0/chk1/chk3 on chk3's native analysis-to-Minutes task.

This evaluator deliberately reuses the immutable sample manifest, prompt
renderer, and task metrics from :mod:`probe_chk3_sft_degeneration`.  Unlike the
historical redacted probe, this evaluator persists the complete decoded
generation, extracted final answer, and generated token IDs so every reported
metric can be recomputed after inference.

One ``run`` invocation loads exactly one model on exactly one caller-pinned
physical GPU.  ``compare`` is CPU-only and fails closed unless all three sealed
run manifests bind the same samples and generation contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROW_SCHEMA_VERSION = "chk3-native-full-generation-row-v1"
RUN_MANIFEST_SCHEMA_VERSION = "chk3-native-generation-run-manifest-v1"
COMPARISON_SCHEMA_VERSION = "chk3-native-three-model-comparison-v1"
SAMPLE_MANIFEST_SCHEMA_VERSION = "chk3-native-test-samples-v1"
TASK_CONTRACT_ID = "chk3-native-analysis-to-minutes-v1"
REQUIRED_STAGES = ("chk0", "chk1", "chk3")
PAIRWISE_CONTRASTS = (
    ("chk0_to_chk1", "chk0", "chk1"),
    ("chk1_to_chk3", "chk1", "chk3"),
    ("chk0_to_chk3", "chk0", "chk3"),
)
SUMMARY_METRICS = (
    "quality_valid_rate",
    "eos_rate",
    "cap_rate",
    "boundary_valid_rate",
    "native_structure_valid_rate",
    "single_paragraph_final_rate",
    "numeric_preservation_rate",
    "signed_numeric_surface_preservation_rate",
    "date_preservation_rate",
    "periodic_tail_rate",
    "mean_full_token_4gram_repetition",
    "mean_tail_token_4gram_repetition",
    "max_full_token_4gram_repetition",
    "max_tail_token_4gram_repetition",
)
CASE_METRICS = (
    "quality_valid",
    "hit_eos",
    "cap_reached",
    "think_boundary_count",
    "exact_boundary_delimiter_count",
    "native_structure_valid",
    "has_nonempty_final_answer",
    "final_answer_single_paragraph",
    "numeric_multiset_preserved",
    "signed_numeric_surface_preserved",
    "missing_signed_numeric_surfaces",
    "unsupported_signed_numeric_surfaces",
    "missing_numeric_occurrences",
    "unsupported_numeric_occurrences",
    "date_set_preserved",
    "missing_date_values",
    "unsupported_date_values",
    "full_token_4gram_repetition",
    "tail_token_4gram_repetition",
    "strict_periodic_tail",
    "raw_generated_tokens",
    "content_tokens",
)


class NativeThreeModelEvalError(RuntimeError):
    """The task-aligned artifact is incomplete, inconsistent, or untrustworthy."""


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
        raise NativeThreeModelEvalError(f"value is not finite JSON: {exc}") from exc


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NativeThreeModelEvalError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    value,
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
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise NativeThreeModelEvalError(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeThreeModelEvalError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeThreeModelEvalError(f"JSON root must be an object: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise NativeThreeModelEvalError(f"missing regular JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise NativeThreeModelEvalError(
                        f"blank JSONL line: {resolved}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise NativeThreeModelEvalError(
                        f"JSONL row is not an object: {resolved}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeThreeModelEvalError(
            f"cannot read JSONL {resolved}: {exc}"
        ) from exc
    return rows


def _file_binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise NativeThreeModelEvalError(f"artifact is not a regular file: {resolved}")
    binding: dict[str, Any] = {
        "path": str(resolved),
        "sha256": native_probe.common_probe.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if payload_sha256 is not None:
        binding["payload_sha256"] = payload_sha256
    return binding


def _validate_stage_id(stage_id: str) -> str:
    if stage_id not in REQUIRED_STAGES:
        raise NativeThreeModelEvalError(
            f"stage_id must be one of {REQUIRED_STAGES}, observed={stage_id!r}"
        )
    return stage_id


def _fingerprint_adapter_for_base(
    adapter_path: Path, *, model_path: Path
) -> Mapping[str, Any]:
    """Fingerprint a PEFT adapter and bind it to the exact requested base."""

    resolved_adapter = adapter_path.expanduser().resolve()
    config_path = resolved_adapter / "adapter_config.json"
    config = _read_json(config_path)
    declared = config.get("base_model_name_or_path")
    if not isinstance(declared, str) or not declared.strip():
        raise NativeThreeModelEvalError("adapter has no base_model_name_or_path")
    declared_path = Path(declared).expanduser()
    if not declared_path.is_absolute():
        declared_path = (Path.cwd() / declared_path).resolve()
    else:
        declared_path = declared_path.resolve()
    if declared_path != model_path:
        raise NativeThreeModelEvalError(
            "adapter base model does not match requested model: "
            f"declared={declared_path}, requested={model_path}"
        )
    try:
        fingerprint = dict(native_probe._fingerprint_adapter(resolved_adapter))
    except native_probe.Chk3ProbeError as exc:
        raise NativeThreeModelEvalError(str(exc)) from exc
    fingerprint["declared_base_model_path"] = str(declared_path)
    fingerprint["adapter_config_sha256"] = (
        native_probe.common_probe.sha256_file(config_path)
    )
    return fingerprint


def _final_answer(text: str) -> str:
    return (
        text.split(native_probe.BOUNDARY, 1)[1].strip()
        if text.count(native_probe.BOUNDARY) == 1
        else ""
    )


def _normalized_identity_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _punctuation_insensitive_identity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


_SIGNED_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\w.])"
)


def _signed_numeric_surfaces(text: str) -> Counter[str]:
    return Counter(match.group(1) for match in _SIGNED_NUMBER_PATTERN.finditer(text))


def _signed_numeric_metrics(source: str, answer: str) -> dict[str, Any]:
    source_values = _signed_numeric_surfaces(source)
    answer_values = _signed_numeric_surfaces(answer)
    missing = source_values - answer_values
    unsupported = answer_values - source_values
    def payload(values: Counter[str]) -> list[list[object]]:
        return [[key, values[key]] for key in sorted(values)]
    return {
        "source_signed_numeric_surfaces_sha256": native_probe.common_probe.sha256_text(
            _canonical_json(payload(source_values))
        ),
        "answer_signed_numeric_surfaces_sha256": native_probe.common_probe.sha256_text(
            _canonical_json(payload(answer_values))
        ),
        "missing_signed_numeric_surfaces": sum(missing.values()),
        "unsupported_signed_numeric_surfaces": sum(unsupported.values()),
        "signed_numeric_surface_preserved": not missing and not unsupported,
    }


def _native_structure_metrics(text: str) -> dict[str, Any]:
    delimiter = "\n</think>\n"
    delimiter_count = text.count(delimiter)
    reasoning = text.split(delimiter, 1)[0].strip() if delimiter_count == 1 else ""
    answer = text.split(delimiter, 1)[1].strip() if delimiter_count == 1 else ""
    failures: list[str] = []
    if delimiter_count != 1:
        failures.append("exact_think_delimiter_count_not_one")
    if "<think>" in text:
        failures.append("opening_think_tag_present")
    if not reasoning:
        failures.append("empty_reasoning")
    if re.search(r"</?think>|<\|[^>]+\|>", answer):
        failures.append("answer_control_tag_present")
    if answer.lstrip().startswith(("{", "[")):
        failures.append("answer_json_like")
    if re.search(r"(?m)^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)", answer):
        failures.append("answer_markdown_heading_or_list")
    if re.search(r"(?i)\bev[-_ ]?\d+\b", answer):
        failures.append("answer_evidence_id_present")
    return {
        "exact_boundary_delimiter_count": delimiter_count,
        "opening_think_tag_absent": "<think>" not in text,
        "has_nonempty_reasoning": bool(reasoning),
        "native_structure_valid": not failures,
        "native_structure_failures": failures,
    }


def build_full_result(
    *,
    text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int] | set[int] | None,
    max_new_tokens: int,
    tail_tokens: int,
    source_prompt: str,
    source_analysis: str,
    reference_response: str,
    load_in_4bit: bool,
    attn_implementation: str,
    pad_token_id: int,
    stage_id: str,
    model_label: str,
    sample_manifest_sha256: str,
    sample: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build one full-text row while reusing the frozen native-probe metrics."""

    _validate_stage_id(stage_id)
    if not all(
        isinstance(value, str)
        for value in (text, source_prompt, source_analysis, reference_response)
    ):
        raise NativeThreeModelEvalError("persisted text fields must be strings")
    try:
        token_ids = [int(value) for value in generated_token_ids]
        eos_ids = sorted(native_probe.common_probe._normalize_eos_ids(eos_token_ids))
    except (TypeError, ValueError, native_probe.common_probe.ProbeError) as exc:
        raise NativeThreeModelEvalError(str(exc)) from exc
    if any(isinstance(value, bool) or value < 0 for value in generated_token_ids):
        raise NativeThreeModelEvalError("generated token IDs must be non-negative integers")
    redacted = native_probe.build_redacted_result(
        text=text,
        generated_token_ids=token_ids,
        eos_token_ids=eos_ids,
        max_new_tokens=max_new_tokens,
        tail_tokens=tail_tokens,
        source_analysis=source_analysis,
        model_label=model_label,
        sample_manifest_sha256=sample_manifest_sha256,
        sample=sample,
        seed=seed,
    )
    answer = _final_answer(text)
    reference_minutes = _final_answer(reference_response)
    if not reference_minutes:
        raise NativeThreeModelEvalError("reference response has no Minutes answer")
    signed_metrics = _signed_numeric_metrics(source_analysis, answer)
    structure_metrics = _native_structure_metrics(text)
    quality_failures = list(redacted["quality_failures"])
    if not signed_metrics["signed_numeric_surface_preserved"]:
        quality_failures.append("signed_numeric_surface_not_preserved")
    quality_failures.extend(structure_metrics["native_structure_failures"])
    return {
        **redacted,
        **signed_metrics,
        **structure_metrics,
        "quality_valid": not quality_failures,
        "quality_failures": sorted(set(quality_failures)),
        "schema_version": ROW_SCHEMA_VERSION,
        "metric_contract_schema_version": native_probe.SCHEMA_VERSION,
        "task_contract_id": TASK_CONTRACT_ID,
        "stage_id": stage_id,
        "do_sample": False,
        "load_in_4bit": load_in_4bit,
        "attn_implementation": attn_implementation,
        "pad_token_id": pad_token_id,
        "source_prompt": source_prompt,
        "source_prompt_sha256": native_probe.common_probe.sha256_text(source_prompt),
        "source_analysis": source_analysis,
        "source_analysis_sha256": native_probe.common_probe.sha256_text(source_analysis),
        "reference_minutes": reference_minutes,
        "reference_minutes_sha256": native_probe.common_probe.sha256_text(
            reference_minutes
        ),
        "analysis_reference_exact_identity": bool(
            sample.get("analysis_reference_exact_identity", False)
        ),
        "normalized_identity": bool(sample.get("normalized_identity", False)),
        "punctuation_insensitive_identity": bool(
            sample.get("punctuation_insensitive_identity", False)
        ),
        "generated_text": text,
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": native_probe.common_probe.sha256_text(
            _canonical_json(token_ids)
        ),
        "eos_token_ids": eos_ids,
        "answer": answer,
        "answer_sha256": native_probe.common_probe.sha256_text(answer),
    }


def validate_full_result(
    row: Mapping[str, Any],
    *,
    stage_id: str,
    model_label: str,
    sample_manifest_sha256: str,
    sample: Mapping[str, Any],
    source_analysis: str,
    reference_response: str,
    tokenizer: Any | None = None,
) -> None:
    """Recompute all metrics from persisted text/token IDs and fail on drift."""

    _validate_stage_id(stage_id)
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise NativeThreeModelEvalError("unsupported full-generation row schema")
    expected_identity = {
        "task_contract_id": TASK_CONTRACT_ID,
        "stage_id": stage_id,
        "model_label": model_label,
        "sample_manifest_sha256": sample_manifest_sha256,
        "sample_id": sample.get("sample_id"),
        "length_bucket": sample.get("length_bucket"),
        "prompt_token_count": sample.get("prompt_token_count"),
        "metric_contract_schema_version": native_probe.SCHEMA_VERSION,
        "analysis_reference_exact_identity": bool(
            sample.get("analysis_reference_exact_identity", False)
        ),
        "normalized_identity": bool(sample.get("normalized_identity", False)),
        "punctuation_insensitive_identity": bool(
            sample.get("punctuation_insensitive_identity", False)
        ),
    }
    for key, expected in expected_identity.items():
        if row.get(key) != expected:
            raise NativeThreeModelEvalError(f"full-generation identity drift at {key}")
    if (
        row.get("do_sample") is not False
        or not isinstance(row.get("load_in_4bit"), bool)
        or row.get("attn_implementation") not in {"sdpa", "flash_attention_2", "eager"}
        or not isinstance(row.get("pad_token_id"), int)
        or isinstance(row.get("pad_token_id"), bool)
    ):
        raise NativeThreeModelEvalError("full-generation runtime contract is invalid")
    text = row.get("generated_text")
    source_prompt = row.get("source_prompt")
    persisted_source_analysis = row.get("source_analysis")
    reference_minutes = row.get("reference_minutes")
    token_ids = row.get("generated_token_ids")
    eos_ids = row.get("eos_token_ids")
    if not all(
        isinstance(value, str)
        for value in (
            text,
            source_prompt,
            persisted_source_analysis,
            reference_minutes,
        )
    ) or not isinstance(token_ids, list):
        raise NativeThreeModelEvalError("full-generation text/token IDs are missing")
    expected_prompt_sha = sample.get("prompt_sha256")
    if (
        native_probe.common_probe.sha256_text(source_prompt) != expected_prompt_sha
        or row.get("source_prompt_sha256") != expected_prompt_sha
    ):
        raise NativeThreeModelEvalError("persisted source prompt drift")
    if native_probe.extract_source_analysis(source_prompt) != source_analysis:
        raise NativeThreeModelEvalError("source prompt/analysis extraction drift")
    if persisted_source_analysis != source_analysis or row.get(
        "source_analysis_sha256"
    ) != native_probe.common_probe.sha256_text(source_analysis):
        raise NativeThreeModelEvalError("persisted source analysis drift")
    expected_reference_minutes = _final_answer(reference_response)
    if reference_minutes != expected_reference_minutes or row.get(
        "reference_minutes_sha256"
    ) != native_probe.common_probe.sha256_text(expected_reference_minutes):
        raise NativeThreeModelEvalError("persisted reference Minutes drift")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in token_ids
    ):
        raise NativeThreeModelEvalError("full-generation token IDs are invalid")
    if not isinstance(eos_ids, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in eos_ids
    ):
        raise NativeThreeModelEvalError("full-generation EOS IDs are invalid")
    eos_positions = [
        index for index, token_id in enumerate(token_ids) if token_id in set(eos_ids)
    ]
    if bool(row.get("hit_eos")):
        if eos_positions != [len(token_ids) - 1]:
            raise NativeThreeModelEvalError("EOS must occur exactly once at the end")
    elif eos_positions:
        raise NativeThreeModelEvalError("non-EOS completion contains an EOS token")
    if tokenizer is not None:
        try:
            decoded = native_probe.common_probe.decode_completion_preserving_boundary(
                tokenizer, token_ids, eos_ids
            )
        except native_probe.common_probe.ProbeError as exc:
            raise NativeThreeModelEvalError(str(exc)) from exc
        if decoded != text:
            raise NativeThreeModelEvalError(
                "persisted generated text does not exactly decode from token IDs"
            )
    max_new_tokens = row.get("max_new_tokens")
    tail_tokens = row.get("tail_tokens")
    if not isinstance(max_new_tokens, int) or not isinstance(tail_tokens, int):
        raise NativeThreeModelEvalError("generation token budget is missing")
    if row.get("completion_sha256") != native_probe.common_probe.sha256_text(text):
        raise NativeThreeModelEvalError("generated text SHA256 mismatch")
    if row.get("generated_token_ids_sha256") != native_probe.common_probe.sha256_text(
        _canonical_json(token_ids)
    ):
        raise NativeThreeModelEvalError("generated token-ID SHA256 mismatch")
    answer = _final_answer(text)
    if row.get("answer") != answer:
        raise NativeThreeModelEvalError("persisted final answer is not exact")
    if row.get("answer_sha256") != native_probe.common_probe.sha256_text(answer):
        raise NativeThreeModelEvalError("final-answer SHA256 mismatch")
    try:
        recomputed = native_probe.analyze_completion(
            text=text,
            generated_token_ids=token_ids,
            eos_token_ids=eos_ids,
            max_new_tokens=max_new_tokens,
            source_analysis=source_analysis,
            tail_tokens=tail_tokens,
        )
    except native_probe.Chk3ProbeError as exc:
        raise NativeThreeModelEvalError(str(exc)) from exc
    for key, expected in recomputed.items():
        if key in {"quality_valid", "quality_failures"}:
            continue
        if row.get(key) != expected:
            raise NativeThreeModelEvalError(f"recomputed metric drift at {key}")
    signed_metrics = _signed_numeric_metrics(source_analysis, answer)
    for key, expected in signed_metrics.items():
        if row.get(key) != expected:
            raise NativeThreeModelEvalError(f"recomputed signed metric drift at {key}")
    structure_metrics = _native_structure_metrics(text)
    for key, expected in structure_metrics.items():
        if row.get(key) != expected:
            raise NativeThreeModelEvalError(f"recomputed structure metric drift at {key}")
    expected_failures = list(recomputed["quality_failures"])
    if not signed_metrics["signed_numeric_surface_preserved"]:
        expected_failures.append("signed_numeric_surface_not_preserved")
    expected_failures.extend(structure_metrics["native_structure_failures"])
    expected_failures = sorted(set(expected_failures))
    if row.get("quality_failures") != expected_failures or row.get(
        "quality_valid"
    ) is not (not expected_failures):
        raise NativeThreeModelEvalError("recomputed signed quality gate drift")


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    sample_manifest_sha256: str,
) -> dict[str, Any]:
    if not rows:
        raise NativeThreeModelEvalError("cannot summarize an empty native run")
    ids = [str(row.get("sample_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise NativeThreeModelEvalError("native run contains duplicate sample IDs")
    for row in rows:
        if row.get("model_label") != model_label:
            raise NativeThreeModelEvalError("native run model-label drift")
        if row.get("sample_manifest_sha256") != sample_manifest_sha256:
            raise NativeThreeModelEvalError("native run sample-manifest drift")
    total = len(rows)
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "complete",
        "model_label": model_label,
        "sample_manifest_sha256": sample_manifest_sha256,
        "cases": total,
        "quality_valid_rate": sum(bool(row["quality_valid"]) for row in rows)
        / total,
        "eos_rate": sum(bool(row["hit_eos"]) for row in rows) / total,
        "cap_rate": sum(bool(row["cap_reached"]) for row in rows) / total,
        "boundary_valid_rate": sum(
            int(row["think_boundary_count"]) == 1 for row in rows
        )
        / total,
        "native_structure_valid_rate": sum(
            bool(row["native_structure_valid"]) for row in rows
        )
        / total,
        "single_paragraph_final_rate": sum(
            bool(row["final_answer_single_paragraph"]) for row in rows
        )
        / total,
        "numeric_preservation_rate": sum(
            bool(row["numeric_multiset_preserved"]) for row in rows
        )
        / total,
        "signed_numeric_surface_preservation_rate": sum(
            bool(row["signed_numeric_surface_preserved"]) for row in rows
        )
        / total,
        "date_preservation_rate": sum(
            bool(row["date_set_preserved"]) for row in rows
        )
        / total,
        "periodic_tail_rate": sum(
            bool(row["strict_periodic_tail"]) for row in rows
        )
        / total,
        "mean_full_token_4gram_repetition": sum(
            float(row["full_token_4gram_repetition"]) for row in rows
        )
        / total,
        "mean_tail_token_4gram_repetition": sum(
            float(row["tail_token_4gram_repetition"]) for row in rows
        )
        / total,
        "max_full_token_4gram_repetition": max(
            float(row["full_token_4gram_repetition"]) for row in rows
        ),
        "max_tail_token_4gram_repetition": max(
            float(row["tail_token_4gram_repetition"]) for row in rows
        ),
    }


def _generation_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise NativeThreeModelEvalError("generation result is empty")
    keys = (
        "generation_mode",
        "do_sample",
        "max_new_tokens",
        "tail_tokens",
        "load_in_4bit",
        "attn_implementation",
        "pad_token_id",
        "eos_token_ids",
    )
    common = {key: rows[0].get(key) for key in keys}
    case_contract = [
        {
            "sample_id": row.get("sample_id"),
            "length_bucket": row.get("length_bucket"),
            "prompt_token_count": row.get("prompt_token_count"),
            "seed": row.get("seed"),
        }
        for row in rows
    ]
    for row in rows[1:]:
        for key, expected in common.items():
            if row.get(key) != expected:
                raise NativeThreeModelEvalError(f"within-run contract drift at {key}")
    return {**common, "cases": case_contract}


def _tokenizer_semantics(tokenizer: Any) -> dict[str, Any]:
    try:
        vocabulary = tokenizer.get_vocab()
    except Exception as exc:
        raise NativeThreeModelEvalError(f"cannot fingerprint tokenizer vocab: {exc}") from exc
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise NativeThreeModelEvalError("tokenizer vocabulary is empty")
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": native_probe.common_probe.sha256_text(
            _canonical_json(vocabulary)
        ),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template_sha256": (
            native_probe.common_probe.sha256_text(chat_template)
            if isinstance(chat_template, str)
            else None
        ),
    }


def _verify_release_file(
    *,
    release: Mapping[str, Any],
    release_root: Path,
    relative_path: str,
    actual_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    files = release.get("files")
    if not isinstance(files, Mapping):
        raise NativeThreeModelEvalError("release manifest has no files binding")
    record = files.get(relative_path)
    if not isinstance(record, Mapping):
        raise NativeThreeModelEvalError(
            f"release manifest does not bind {relative_path}"
        )
    resolved = actual_path.expanduser().resolve()
    if (release_root / relative_path).resolve() != resolved:
        raise NativeThreeModelEvalError(
            f"release-bound path mismatch for {relative_path}"
        )
    observed_sha = native_probe.common_probe.sha256_file(resolved)
    if record.get("sha256") != observed_sha or record.get("rows") != expected_rows:
        raise NativeThreeModelEvalError(
            f"release file hash/row binding mismatch for {relative_path}"
        )
    return {
        "path": str(resolved),
        "release_relative_path": relative_path,
        "sha256": observed_sha,
        "rows": expected_rows,
    }


def build_test_sample_manifest(
    *,
    test_data: Path,
    test_row_manifest: Path,
    release_manifest: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    training_config: Path,
    samples_per_bucket: int = 4,
) -> dict[str, Any]:
    """Build a sealed, deterministic N-row manifest from the held-out test split."""

    if samples_per_bucket != 4:
        raise NativeThreeModelEvalError("formal test manifest requires 4 samples per bucket")
    test_data = test_data.expanduser().resolve()
    test_row_manifest = test_row_manifest.expanduser().resolve()
    release_manifest = release_manifest.expanduser().resolve()
    release = _read_json(release_manifest)
    if release.get("schema_version") != "chk3-minutes-training-release-v1":
        raise NativeThreeModelEvalError("unsupported chk3 release manifest schema")
    if release.get("immutable") is not True or release.get("quality_status") != "passed":
        raise NativeThreeModelEvalError("chk3 release is not immutable and passed")
    split_counts = release.get("split_counts")
    if not isinstance(split_counts, Mapping) or not isinstance(
        split_counts.get("test"), int
    ):
        raise NativeThreeModelEvalError("release manifest has no test split count")
    expected_rows = int(split_counts["test"])
    data_rows = _read_jsonl(test_data)
    row_records = _read_jsonl(test_row_manifest)
    if len(data_rows) != expected_rows or len(row_records) != expected_rows:
        raise NativeThreeModelEvalError("test data/row-manifest count mismatch")
    release_root = release_manifest.parent
    data_binding = _verify_release_file(
        release=release,
        release_root=release_root,
        relative_path="minutes_alignment/test.jsonl",
        actual_path=test_data,
        expected_rows=expected_rows,
    )
    row_manifest_binding = _verify_release_file(
        release=release,
        release_root=release_root,
        relative_path="minutes_alignment/manifests/test.jsonl",
        actual_path=test_row_manifest,
        expected_rows=expected_rows,
    )
    try:
        system_prompt, suffix, config_sha = native_probe._load_prompt_contract(
            training_config
        )
    except native_probe.Chk3ProbeError as exc:
        raise NativeThreeModelEvalError(str(exc)) from exc

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, (row, row_record) in enumerate(
        zip(data_rows, row_records, strict=True), start=1
    ):
        prompt = row.get("prompt")
        response = row.get("response")
        sample_id = row_record.get("sample_id")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise NativeThreeModelEvalError(f"invalid test row at line {line_number}")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise NativeThreeModelEvalError(
                f"invalid/duplicate canonical sample ID at line {line_number}"
            )
        seen_ids.add(sample_id)
        if row_record.get("split") != "test" or row_record.get("source_split") != "test":
            raise NativeThreeModelEvalError(
                f"non-test identity in test row manifest at line {line_number}"
            )
        prompt_sha = native_probe.common_probe.sha256_text(prompt)
        response_sha = native_probe.common_probe.sha256_text(response)
        if (
            row_record.get("prompt_sha256") != prompt_sha
            or row_record.get("response_sha256") != response_sha
        ):
            raise NativeThreeModelEvalError(
                f"test row hash does not match release row manifest at line {line_number}"
            )
        analysis = native_probe.extract_source_analysis(prompt)
        reference_minutes = _final_answer(response)
        if not reference_minutes:
            raise NativeThreeModelEvalError(
                f"test reference has no final Minutes answer at line {line_number}"
            )
        completion_tokens = row_record.get("completion_tokens")
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            raise NativeThreeModelEvalError(
                f"invalid release completion token count at line {line_number}"
            )
        try:
            prompt_ids = native_probe._prompt_ids(
                tokenizer,
                native_probe._messages(
                    row,
                    system_prompt=system_prompt,
                    user_prompt_suffix=suffix,
                ),
            )
        except native_probe.Chk3ProbeError as exc:
            raise NativeThreeModelEvalError(str(exc)) from exc
        candidates.append(
            {
                "sample_id": sample_id,
                "line_number": line_number,
                "release_row_manifest_line_number": line_number,
                "prompt_sha256": prompt_sha,
                "response_sha256": response_sha,
                "analysis_sha256": native_probe.common_probe.sha256_text(analysis),
                "release_analysis_sha256": row_record.get("analysis_sha256"),
                "reference_minutes_sha256": native_probe.common_probe.sha256_text(
                    reference_minutes
                ),
                "completion_tokens": completion_tokens,
                "analysis_reference_exact_identity": analysis == reference_minutes,
                "normalized_identity": _normalized_identity_text(analysis)
                == _normalized_identity_text(reference_minutes),
                "punctuation_insensitive_identity": (
                    _punctuation_insensitive_identity_text(analysis)
                    == _punctuation_insensitive_identity_text(reference_minutes)
                ),
                "source_numeric_multiset_sha256": native_probe._numeric_hash(analysis),
                "source_date_set_sha256": native_probe._date_hash(analysis),
                "source_numeric_occurrences": sum(
                    native_probe._numeric_values(analysis).values()
                ),
                "source_date_values": len(native_probe._date_values(analysis)),
                "prompt_token_count": len(prompt_ids),
            }
        )
    if len(candidates) < samples_per_bucket * 3:
        raise NativeThreeModelEvalError("test split is too small for requested sample")

    ordered = sorted(
        candidates,
        key=lambda row: (int(row["completion_tokens"]), str(row["sample_id"])),
    )
    boundaries = (0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered))
    release_sha = native_probe.common_probe.sha256_file(release_manifest)
    selected: list[dict[str, Any]] = []
    for bucket_index, bucket in enumerate(("short", "medium", "long")):
        population = ordered[boundaries[bucket_index] : boundaries[bucket_index + 1]]
        ranked = sorted(
            population,
            key=lambda row: (
                native_probe.common_probe.sha256_text(
                    f"{TASK_CONTRACT_ID}|{release_sha}|{row['sample_id']}"
                ),
                str(row["sample_id"]),
            ),
        )
        chosen = ranked[:samples_per_bucket]
        for within_bucket_rank, row in enumerate(chosen):
            selected.append(
                {
                    **row,
                    "length_bucket": bucket,
                    "within_bucket_hash_rank": within_bucket_rank,
                    "bucket_population_rows": len(population),
                }
            )

    payload = {
        "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "task_contract_id": TASK_CONTRACT_ID,
        "release": {
            "path": str(release_manifest),
            "sha256": release_sha,
            "release_id": release.get("release_id"),
            "schema_version": release.get("schema_version"),
        },
        "dataset": {**data_binding, "split": "test"},
        "release_row_manifest": row_manifest_binding,
        "prompt_contract": {
            "training_config": str(training_config.expanduser().resolve()),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": native_probe.common_probe.sha256_text(system_prompt),
            "user_prompt_suffix_sha256": (
                native_probe.common_probe.sha256_text(suffix)
                if suffix is not None
                else None
            ),
        },
        "tokenizer": native_probe._tokenizer_fingerprint(tokenizer_path),
        "selection": {
            "algorithm": "held-out-test-completion-length-tertiles-domain-hash-v1",
            "selection_domain": TASK_CONTRACT_ID,
            "ordering": "(completion_tokens,sample_id)",
            "bucket_boundaries": list(boundaries),
            "within_bucket_rank": (
                "sha256(selection_domain|release_manifest_sha256|sample_id)"
            ),
            "samples_per_bucket": samples_per_bucket,
            "rows": len(selected),
            "buckets": ["short", "medium", "long"],
            "checkpoint_selection_split": "validation",
            "evaluation_split": "test",
            "identity_check": {
                "exact_field": "analysis_reference_exact_identity",
                "normalized_field": "normalized_identity",
                "normalization": "NFKC+casefold+whitespace-collapse-v1",
                "punctuation_insensitive_field": "punctuation_insensitive_identity",
                "punctuation_insensitive_normalization": (
                    "NFKC+casefold+remove-Unicode-P-category+whitespace-collapse-v1"
                ),
            },
        },
        "samples": selected,
    }
    return seal_manifest(payload)


def _validate_new_sample_manifest(
    manifest: Mapping[str, Any],
    *,
    path: Path,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    observed_sha = native_probe.common_probe.sha256_file(path)
    if observed_sha != expected_sha256:
        raise NativeThreeModelEvalError(
            f"sample-manifest SHA256 mismatch: expected={expected_sha256}, "
            f"observed={observed_sha}"
        )
    try:
        validate_manifest_integrity(manifest)
    except Exception as exc:
        raise NativeThreeModelEvalError(f"sample manifest integrity failed: {exc}") from exc
    if manifest.get("task_contract_id") != TASK_CONTRACT_ID:
        raise NativeThreeModelEvalError("sample manifest task contract drift")
    selection = manifest.get("selection")
    samples = manifest.get("samples")
    if not isinstance(selection, Mapping) or not isinstance(samples, list) or not samples:
        raise NativeThreeModelEvalError("sample manifest selection is invalid")
    if selection.get("rows") != len(samples):
        raise NativeThreeModelEvalError("sample manifest selected row count drift")
    per_bucket = selection.get("samples_per_bucket")
    if per_bucket != 4 or selection.get("rows") != 12:
        raise NativeThreeModelEvalError("sample manifest bucket size is invalid")
    buckets = [sample.get("length_bucket") for sample in samples if isinstance(sample, Mapping)]
    if len(buckets) != len(samples) or any(
        buckets.count(bucket) != per_bucket for bucket in ("short", "medium", "long")
    ):
        raise NativeThreeModelEvalError("sample manifest length strata are incomplete")
    ids = [sample.get("sample_id") for sample in samples if isinstance(sample, Mapping)]
    if len(ids) != len(samples) or len(set(ids)) != len(ids):
        raise NativeThreeModelEvalError("sample manifest IDs are invalid")
    if any(
        not isinstance(sample.get("analysis_reference_exact_identity"), bool)
        or not isinstance(sample.get("normalized_identity"), bool)
        or not isinstance(sample.get("punctuation_insensitive_identity"), bool)
        for sample in samples
    ):
        raise NativeThreeModelEvalError("sample identity booleans are missing")
    return manifest, observed_sha


def _load_sample_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    resolved = path.expanduser().resolve()
    manifest = _read_json(resolved)
    if manifest.get("schema_version") == SAMPLE_MANIFEST_SCHEMA_VERSION:
        return _validate_new_sample_manifest(
            manifest, path=resolved, expected_sha256=expected_sha256
        )
    # Backward compatibility is retained only so the existing explicit N=3
    # full-config-bound manifest remains runnable as a diagnostic.
    if manifest.get("schema_version") == native_probe.MANIFEST_SCHEMA_VERSION:
        try:
            return native_probe._validate_sample_manifest(resolved, expected_sha256)
        except native_probe.Chk3ProbeError as exc:
            raise NativeThreeModelEvalError(str(exc)) from exc
    raise NativeThreeModelEvalError("unsupported native sample-manifest schema")


def _load_bound_rows(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if manifest.get("schema_version") == native_probe.MANIFEST_SCHEMA_VERSION:
        try:
            return native_probe._load_bound_rows(manifest)
        except native_probe.Chk3ProbeError as exc:
            raise NativeThreeModelEvalError(str(exc)) from exc
    dataset = manifest.get("dataset")
    row_manifest = manifest.get("release_row_manifest")
    release_binding = manifest.get("release")
    if not all(
        isinstance(value, Mapping)
        for value in (dataset, row_manifest, release_binding)
    ):
        raise NativeThreeModelEvalError("sample manifest source bindings are missing")
    dataset_path = Path(str(dataset.get("path"))).expanduser().resolve()
    row_manifest_path = Path(str(row_manifest.get("path"))).expanduser().resolve()
    release_path = Path(str(release_binding.get("path"))).expanduser().resolve()
    for path, binding, label in (
        (dataset_path, dataset, "test dataset"),
        (row_manifest_path, row_manifest, "test row manifest"),
        (release_path, release_binding, "release manifest"),
    ):
        if native_probe.common_probe.sha256_file(path) != binding.get("sha256"):
            raise NativeThreeModelEvalError(f"bound {label} hash changed")
    rows = _read_jsonl(dataset_path)
    row_records = _read_jsonl(row_manifest_path)
    if len(rows) != dataset.get("rows") or len(row_records) != row_manifest.get("rows"):
        raise NativeThreeModelEvalError("bound test artifact row count changed")
    selection_candidates: list[tuple[int, str, int]] = []
    selection_seen_ids: set[str] = set()
    for line_number, row_record in enumerate(row_records, start=1):
        candidate_id = row_record.get("sample_id")
        completion_tokens = row_record.get("completion_tokens")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in selection_seen_ids
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
        ):
            raise NativeThreeModelEvalError("invalid release row selection identity")
        selection_seen_ids.add(candidate_id)
        selection_candidates.append((completion_tokens, candidate_id, line_number))
    selection_candidates.sort(key=lambda item: (item[0], item[1]))
    selection_boundaries = (
        0,
        len(selection_candidates) // 3,
        (2 * len(selection_candidates)) // 3,
        len(selection_candidates),
    )
    release_sha = str(release_binding.get("sha256"))
    expected_selection: list[tuple[str, str, int]] = []
    for bucket_index, bucket in enumerate(("short", "medium", "long")):
        population = selection_candidates[
            selection_boundaries[bucket_index] : selection_boundaries[bucket_index + 1]
        ]
        ranked = sorted(
            population,
            key=lambda item: (
                native_probe.common_probe.sha256_text(
                    f"{TASK_CONTRACT_ID}|{release_sha}|{item[1]}"
                ),
                item[1],
            ),
        )
        expected_selection.extend(
            (sample_id, bucket, line_number)
            for _tokens, sample_id, line_number in ranked[:4]
        )
    observed_selection = [
        (
            str(sample.get("sample_id")),
            str(sample.get("length_bucket")),
            int(sample.get("line_number", -1)),
        )
        for sample in manifest["samples"]
    ]
    if observed_selection != expected_selection:
        raise NativeThreeModelEvalError(
            "sample selection is not the frozen deterministic test N=12"
        )
    bound: dict[str, Mapping[str, Any]] = {}
    for sample in manifest["samples"]:
        line_number = sample.get("line_number")
        release_line = sample.get("release_row_manifest_line_number")
        if (
            not isinstance(line_number, int)
            or line_number <= 0
            or release_line != line_number
            or line_number > len(rows)
        ):
            raise NativeThreeModelEvalError("sample line binding is invalid")
        row = rows[line_number - 1]
        row_record = row_records[line_number - 1]
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise NativeThreeModelEvalError("bound test row schema changed")
        if row_record.get("sample_id") != sample.get("sample_id"):
            raise NativeThreeModelEvalError("bound canonical sample ID changed")
        analysis = native_probe.extract_source_analysis(prompt)
        reference_minutes = _final_answer(response)
        if not reference_minutes:
            raise NativeThreeModelEvalError("bound reference has no Minutes answer")
        checks = {
            "prompt_sha256": native_probe.common_probe.sha256_text(prompt),
            "response_sha256": native_probe.common_probe.sha256_text(response),
            "analysis_sha256": native_probe.common_probe.sha256_text(analysis),
            "source_numeric_multiset_sha256": native_probe._numeric_hash(analysis),
            "source_date_set_sha256": native_probe._date_hash(analysis),
            "reference_minutes_sha256": native_probe.common_probe.sha256_text(
                reference_minutes
            ),
        }
        for key, observed in checks.items():
            if sample.get(key) != observed:
                raise NativeThreeModelEvalError(f"bound sample drift at {key}")
        if row_record.get("prompt_sha256") != checks["prompt_sha256"] or row_record.get(
            "response_sha256"
        ) != checks["response_sha256"]:
            raise NativeThreeModelEvalError("release row-manifest hash drift")
        if sample.get("completion_tokens") != row_record.get("completion_tokens"):
            raise NativeThreeModelEvalError("release completion-token binding drift")
        if sample.get("analysis_reference_exact_identity") is not (
            analysis == reference_minutes
        ) or sample.get("normalized_identity") is not (
            _normalized_identity_text(analysis)
            == _normalized_identity_text(reference_minutes)
        ) or sample.get("punctuation_insensitive_identity") is not (
            _punctuation_insensitive_identity_text(analysis)
            == _punctuation_insensitive_identity_text(reference_minutes)
        ):
            raise NativeThreeModelEvalError("analysis/reference identity flag drift")
        if sample.get("source_numeric_occurrences") != sum(
            native_probe._numeric_values(analysis).values()
        ) or sample.get("source_date_values") != len(native_probe._date_values(analysis)):
            raise NativeThreeModelEvalError("bound sample feature-count drift")
        bound[str(sample["sample_id"])] = row
    return bound


def run_model(
    *,
    stage_id: str,
    model_label: str,
    model_path: Path,
    adapter_path: Path | None,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    physical_gpu_index: int,
    max_new_tokens: int = native_probe.DEFAULT_MAX_NEW_TOKENS,
    tail_tokens: int = native_probe.DEFAULT_TAIL_TOKENS,
    seed: int = native_probe.DEFAULT_SEED,
    load_in_4bit: bool = True,
    attn_implementation: str = "sdpa",
    resume: bool = False,
) -> dict[str, Any]:
    """Generate and seal all native chk3 probe cases for one model."""

    _validate_stage_id(stage_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_label):
        raise NativeThreeModelEvalError("model_label contains unsupported characters")
    if physical_gpu_index != 0:
        raise NativeThreeModelEvalError("formal native evaluation is pinned to physical GPU 0")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if visible != [str(physical_gpu_index)]:
        raise NativeThreeModelEvalError(
            "CUDA_VISIBLE_DEVICES must expose only the requested physical GPU; "
            f"expected={[str(physical_gpu_index)]}, observed={visible}"
        )
    if max_new_tokens < 512 or tail_tokens <= 0:
        raise NativeThreeModelEvalError("invalid generation token budget")
    if output_dir.is_symlink():
        raise NativeThreeModelEvalError(f"run output is a symlink: {output_dir}")
    if output_dir.exists() and not resume:
        raise NativeThreeModelEvalError(f"run output already exists: {output_dir}")
    sample_manifest, observed_sample_sha = _load_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    bound_rows = _load_bound_rows(sample_manifest)
    prompt_contract = sample_manifest.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise NativeThreeModelEvalError("sample manifest has no prompt contract")
    config_path = Path(str(prompt_contract.get("training_config")))
    try:
        system_prompt, suffix, config_sha = native_probe._load_prompt_contract(
            config_path
        )
    except native_probe.Chk3ProbeError as exc:
        raise NativeThreeModelEvalError(str(exc)) from exc
    if config_sha != prompt_contract.get("training_config_sha256"):
        raise NativeThreeModelEvalError("bound chk3 training config changed")
    if native_probe.common_probe.sha256_text(system_prompt) != prompt_contract.get(
        "system_prompt_sha256"
    ):
        raise NativeThreeModelEvalError("bound chk3 system prompt changed")
    suffix_sha = (
        native_probe.common_probe.sha256_text(suffix) if suffix is not None else None
    )
    if suffix_sha != prompt_contract.get("user_prompt_suffix_sha256"):
        raise NativeThreeModelEvalError("bound chk3 prompt suffix changed")

    model_path = model_path.expanduser().resolve()
    try:
        model_fingerprint = native_probe._fingerprint_model(model_path)
    except native_probe.Chk3ProbeError as exc:
        raise NativeThreeModelEvalError(str(exc)) from exc
    adapter_fingerprint = (
        _fingerprint_adapter_for_base(adapter_path, model_path=model_path)
        if adapter_path is not None
        else None
    )
    resuming = output_dir.exists()
    if not resuming:
        output_dir.mkdir(parents=True, exist_ok=False)
    partial_dir = output_dir / ".partial"
    if not resuming:
        partial_dir.mkdir()
    launch_path = output_dir / "launch.json"
    progress_path = partial_dir / "generations.progress.v1.jsonl"
    final_path = output_dir / "generations.jsonl"
    state_path = output_dir / "state.progress.v1.json"
    launch_payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "task_contract_id": TASK_CONTRACT_ID,
        "stage_id": stage_id,
        "model_label": model_label,
        "model": model_fingerprint,
        "adapter": adapter_fingerprint,
        "sample_manifest": {
            "path": str(sample_manifest_path.expanduser().resolve()),
            "sha256": observed_sample_sha,
        },
        "generation": {
            "generation_mode": "greedy",
            "do_sample": False,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "tail_tokens": tail_tokens,
            "load_in_4bit": load_in_4bit,
            "attn_implementation": attn_implementation,
        },
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": visible,
            "physical_gpu_index": physical_gpu_index,
        },
        "persistence": {
            "source_prompt": True,
            "source_analysis": True,
            "reference_minutes": True,
            "generated_text": True,
            "answer": True,
            "generated_token_ids": True,
            "append_flush_fsync_per_row": True,
        },
    }
    resume_count = 0
    results: list[dict[str, Any]] = []
    if resuming:
        if final_path.exists() or (output_dir / "manifest.json").exists():
            raise NativeThreeModelEvalError("cannot resume an already finalized run")
        prior_launch = _read_json(launch_path)
        try:
            validate_manifest_integrity(prior_launch)
        except Exception as exc:
            raise NativeThreeModelEvalError(f"resume launch integrity failed: {exc}") from exc
        for key in (
            "schema_version",
            "task_contract_id",
            "stage_id",
            "model_label",
            "model",
            "adapter",
            "sample_manifest",
            "generation",
            "persistence",
            "runtime",
        ):
            if prior_launch.get(key) != launch_payload.get(key):
                raise NativeThreeModelEvalError(f"resume launch contract drift at {key}")
        if not progress_path.is_file() or progress_path.is_symlink():
            raise NativeThreeModelEvalError("resume partial generations are missing")
        results = _read_jsonl(progress_path)
        if len(results) >= len(sample_manifest["samples"]):
            raise NativeThreeModelEvalError("resume partial result count is invalid")
        if state_path.is_file() and not state_path.is_symlink():
            prior_state = _read_json(state_path)
            resume_count = int(prior_state.get("resume_count", 0)) + 1
        else:
            resume_count = 1
        for index, result in enumerate(results):
            sample = sample_manifest["samples"][index]
            bound = bound_rows[str(sample["sample_id"])]
            validate_full_result(
                result,
                stage_id=stage_id,
                model_label=model_label,
                sample_manifest_sha256=observed_sample_sha,
                sample=sample,
                source_analysis=native_probe.extract_source_analysis(
                    str(bound["prompt"])
                ),
                reference_response=str(bound["response"]),
            )
            if (
                result.get("load_in_4bit") is not load_in_4bit
                or result.get("attn_implementation") != attn_implementation
            ):
                raise NativeThreeModelEvalError("resume generation runtime drift")
    else:
        _write_new_json(launch_path, seal_manifest(launch_payload))
        _atomic_write_json(
            state_path,
            {
                "status": "initializing",
                "stage_id": stage_id,
                "completed_cases": 0,
                "expected_cases": len(sample_manifest["samples"]),
                "resume_count": 0,
                "partial_results": str(progress_path.resolve()),
            },
        )

    model = None
    try:
        import torch
        import transformers
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise NativeThreeModelEvalError(
                "exactly one visible CUDA GPU is required for a model run"
            )
        tokenizer_record = sample_manifest.get("tokenizer")
        if not isinstance(tokenizer_record, Mapping):
            raise NativeThreeModelEvalError("sample manifest has no tokenizer binding")
        tokenizer_path = Path(str(tokenizer_record.get("path"))).expanduser().resolve()
        tokenizer_files = tokenizer_record.get("files")
        if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
            raise NativeThreeModelEvalError("sample tokenizer binding is invalid")
        for name, expected in tokenizer_files.items():
            path = tokenizer_path / str(name)
            if native_probe.common_probe.sha256_file(path) != expected:
                raise NativeThreeModelEvalError(f"bound tokenizer changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer_semantics = _tokenizer_semantics(tokenizer)

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path.expanduser().resolve()),
                is_trainable=False,
                local_files_only=True,
            )
        model.eval()
        model.config.use_cache = True
        if int(getattr(model.config, "vocab_size", 0) or 0) != int(
            tokenizer_semantics["vocabulary_size"]
        ):
            raise NativeThreeModelEvalError(
                "model vocabulary size is incompatible with the frozen tokenizer"
            )
        eos_value = model.generation_config.eos_token_id
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        try:
            eos_ids = native_probe.common_probe._normalize_eos_ids(eos_value)
        except native_probe.common_probe.ProbeError as exc:
            raise NativeThreeModelEvalError(str(exc)) from exc
        if not eos_ids:
            raise NativeThreeModelEvalError("model/tokenizer has no EOS token ID")
        for result_index, result in enumerate(results):
            if result.get("eos_token_ids") != sorted(eos_ids) or result.get(
                "pad_token_id"
            ) != int(tokenizer.pad_token_id):
                raise NativeThreeModelEvalError("resume tokenizer stop contract drift")
            resumed_sample = sample_manifest["samples"][result_index]
            resumed_bound = bound_rows[str(resumed_sample["sample_id"])]
            validate_full_result(
                result,
                stage_id=stage_id,
                model_label=model_label,
                sample_manifest_sha256=observed_sample_sha,
                sample=resumed_sample,
                source_analysis=native_probe.extract_source_analysis(
                    str(resumed_bound["prompt"])
                ),
                reference_response=str(resumed_bound["response"]),
                tokenizer=tokenizer,
            )

        torch.cuda.reset_peak_memory_stats()
        with progress_path.open("a" if resuming else "x", encoding="utf-8") as output_handle:
            for index, sample in enumerate(
                sample_manifest["samples"][len(results) :], start=len(results)
            ):
                sample_id = str(sample["sample_id"])
                row = bound_rows[sample_id]
                try:
                    prompt_ids = native_probe._prompt_ids(
                        tokenizer,
                        native_probe._messages(
                            row,
                            system_prompt=system_prompt,
                            user_prompt_suffix=suffix,
                        ),
                    )
                except native_probe.Chk3ProbeError as exc:
                    raise NativeThreeModelEvalError(str(exc)) from exc
                if len(prompt_ids) != sample.get("prompt_token_count"):
                    raise NativeThreeModelEvalError(
                        f"prompt token-count drift for {sample_id}"
                    )
                context_limit = int(
                    getattr(model.config, "max_position_embeddings", 0) or 0
                )
                if context_limit and len(prompt_ids) + max_new_tokens > context_limit:
                    raise NativeThreeModelEvalError(
                        f"context overflow for {sample_id}: prompt={len(prompt_ids)}, "
                        f"completion={max_new_tokens}, limit={context_limit}"
                    )
                input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device="cuda:0"
                )
                attention_mask = torch.ones_like(input_ids)
                case_seed = seed + index
                transformers.set_seed(case_seed)
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
                try:
                    text = native_probe.common_probe.decode_completion_preserving_boundary(
                        tokenizer, generated_ids, eos_ids
                    )
                except native_probe.common_probe.ProbeError as exc:
                    raise NativeThreeModelEvalError(str(exc)) from exc
                source_analysis = native_probe.extract_source_analysis(
                    str(row["prompt"])
                )
                result = build_full_result(
                    text=text,
                    generated_token_ids=generated_ids,
                    eos_token_ids=eos_ids,
                    max_new_tokens=max_new_tokens,
                    tail_tokens=tail_tokens,
                    source_prompt=str(row["prompt"]),
                    source_analysis=source_analysis,
                    reference_response=str(row["response"]),
                    load_in_4bit=load_in_4bit,
                    attn_implementation=attn_implementation,
                    pad_token_id=int(tokenizer.pad_token_id),
                    stage_id=stage_id,
                    model_label=model_label,
                    sample_manifest_sha256=observed_sample_sha,
                    sample=sample,
                    seed=case_seed,
                )
                validate_full_result(
                    result,
                    stage_id=stage_id,
                    model_label=model_label,
                    sample_manifest_sha256=observed_sample_sha,
                    sample=sample,
                    source_analysis=source_analysis,
                    reference_response=str(row["response"]),
                    tokenizer=tokenizer,
                )
                output_handle.write(_canonical_json(result) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                results.append(result)
                _atomic_write_json(
                    state_path,
                    {
                        "status": "generating",
                        "stage_id": stage_id,
                        "completed_cases": len(results),
                        "expected_cases": len(sample_manifest["samples"]),
                        "resume_count": resume_count,
                        "last_sample_id": sample_id,
                        "partial_results": str(progress_path.resolve()),
                        "partial_results_sha256": native_probe.common_probe.sha256_file(
                            progress_path
                        ),
                    },
                )
                del input_ids, attention_mask, sequences

        os.replace(progress_path, final_path)
        summary = _summary(
            results,
            model_label=model_label,
            sample_manifest_sha256=observed_sample_sha,
        )
        run_payload = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "task_contract_id": TASK_CONTRACT_ID,
            "stage_id": stage_id,
            "model_label": model_label,
            "model": model_fingerprint,
            "adapter": adapter_fingerprint,
            "sample_manifest": {
                "path": str(sample_manifest_path.expanduser().resolve()),
                "sha256": observed_sample_sha,
            },
            "prompt_contract": dict(prompt_contract),
            "tokenizer_semantics": tokenizer_semantics,
            "generation_contract": _generation_contract(results),
            "runtime": {
                "physical_gpu_index": physical_gpu_index,
                "cuda_visible_devices": visible,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                "resume_count": resume_count,
            },
            "persistence": {
                "source_prompt": True,
                "source_analysis": True,
                "reference_minutes": True,
                "generated_text": True,
                "answer": True,
                "generated_token_ids": True,
                "append_flush_fsync_per_row": True,
            },
            "artifacts": {
                "launch": _file_binding(
                    launch_path,
                    payload_sha256=validate_manifest_integrity(_read_json(launch_path)),
                ),
                "generations": {
                    **_file_binding(final_path),
                    "rows": len(results),
                },
            },
            "summary": summary,
            "row_bindings": [
                {
                    "sample_id": row["sample_id"],
                    "completion_sha256": row["completion_sha256"],
                    "answer_sha256": row["answer_sha256"],
                    "reference_minutes_sha256": row[
                        "reference_minutes_sha256"
                    ],
                    "generated_token_ids_sha256": row[
                        "generated_token_ids_sha256"
                    ],
                }
                for row in results
            ],
            "limitations": {
                "cases": len(results),
                "selection": "deterministic short-medium-long length-stratified diagnostic",
                "semantic_minutes_quality_graded": False,
            },
        }
        run_manifest = seal_manifest(run_payload)
        manifest_path = output_dir / "manifest.json"
        _write_new_json(manifest_path, run_manifest)
        _atomic_write_json(
            state_path,
            {
                "status": "complete",
                "stage_id": stage_id,
                "completed_cases": len(results),
                "expected_cases": len(sample_manifest["samples"]),
                "resume_count": resume_count,
                "generations": _file_binding(final_path),
                "manifest": _file_binding(
                    manifest_path,
                    payload_sha256=run_manifest["integrity"]["payload_sha256"],
                ),
            },
        )
        return run_manifest
    except Exception as exc:
        _atomic_write_json(
            state_path,
            {
                "status": "failed",
                "stage_id": stage_id,
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_cases": len(results),
                "expected_cases": len(sample_manifest["samples"]),
                "resume_count": resume_count,
                "partial_results": (
                    _file_binding(progress_path) if progress_path.is_file() else None
                ),
            },
        )
        raise
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def load_and_validate_run(
    manifest_path: Path,
    *,
    expected_stage_id: str,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
) -> dict[str, Any]:
    """Deep-validate one sealed run and all row-level metrics."""

    _validate_stage_id(expected_stage_id)
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    try:
        payload_sha = validate_manifest_integrity(manifest)
    except Exception as exc:
        raise NativeThreeModelEvalError(
            f"run manifest integrity failed for {expected_stage_id}: {exc}"
        ) from exc
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise NativeThreeModelEvalError("unsupported native run manifest schema")
    if manifest.get("status") != "complete":
        raise NativeThreeModelEvalError("native model run is not complete")
    if manifest.get("task_contract_id") != TASK_CONTRACT_ID:
        raise NativeThreeModelEvalError("native model run task contract drift")
    if manifest.get("stage_id") != expected_stage_id:
        raise NativeThreeModelEvalError("native model run stage mismatch")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("physical_gpu_index") != 0
        or runtime.get("cuda_visible_devices") != ["0"]
    ):
        raise NativeThreeModelEvalError("native model run was not executed on GPU0 only")
    model_label = manifest.get("model_label")
    if not isinstance(model_label, str) or not model_label:
        raise NativeThreeModelEvalError("native model run label is invalid")
    sample_binding = manifest.get("sample_manifest")
    if not isinstance(sample_binding, Mapping):
        raise NativeThreeModelEvalError("native model run has no sample binding")
    if (
        Path(str(sample_binding.get("path"))).expanduser().resolve()
        != sample_manifest_path.expanduser().resolve()
        or sample_binding.get("sha256") != sample_manifest_sha256
    ):
        raise NativeThreeModelEvalError("native model run sample binding drift")
    sample_manifest, observed_sample_sha = _load_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    bound_rows = _load_bound_rows(sample_manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise NativeThreeModelEvalError("native model run has no artifacts")
    generation_binding = artifacts.get("generations")
    if not isinstance(generation_binding, Mapping):
        raise NativeThreeModelEvalError("native model run has no generations binding")
    results_path = Path(str(generation_binding.get("path"))).expanduser().resolve()
    if results_path.parent != manifest_path.parent or results_path.name != "generations.jsonl":
        raise NativeThreeModelEvalError("generations path is outside the run directory")
    if native_probe.common_probe.sha256_file(results_path) != generation_binding.get(
        "sha256"
    ):
        raise NativeThreeModelEvalError("generations artifact SHA256 mismatch")
    rows = _read_jsonl(results_path)
    expected_row_count = len(sample_manifest["samples"])
    if len(rows) != generation_binding.get("rows") or len(rows) != expected_row_count:
        raise NativeThreeModelEvalError("generations artifact row count mismatch")
    expected_ids = [str(sample["sample_id"]) for sample in sample_manifest["samples"]]
    if [str(row.get("sample_id")) for row in rows] != expected_ids:
        raise NativeThreeModelEvalError("generations row order/sample IDs drift")
    tokenizer_record = sample_manifest.get("tokenizer")
    if not isinstance(tokenizer_record, Mapping):
        raise NativeThreeModelEvalError("sample manifest has no tokenizer binding")
    tokenizer_path = Path(str(tokenizer_record.get("path"))).expanduser().resolve()
    tokenizer_files = tokenizer_record.get("files")
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise NativeThreeModelEvalError("sample tokenizer binding is invalid")
    for name, expected in tokenizer_files.items():
        if native_probe.common_probe.sha256_file(tokenizer_path / str(name)) != expected:
            raise NativeThreeModelEvalError(f"bound tokenizer changed: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    for row, sample in zip(rows, sample_manifest["samples"], strict=True):
        bound = bound_rows[str(sample["sample_id"])]
        source_analysis = native_probe.extract_source_analysis(str(bound["prompt"]))
        validate_full_result(
            row,
            stage_id=expected_stage_id,
            model_label=model_label,
            sample_manifest_sha256=observed_sample_sha,
            sample=sample,
            source_analysis=source_analysis,
            reference_response=str(bound["response"]),
            tokenizer=tokenizer,
        )
    observed_summary = _summary(
        rows,
        model_label=model_label,
        sample_manifest_sha256=observed_sample_sha,
    )
    if manifest.get("summary") != observed_summary:
        raise NativeThreeModelEvalError("sealed run summary is not reproducible")
    if manifest.get("generation_contract") != _generation_contract(rows):
        raise NativeThreeModelEvalError("sealed run generation contract is not reproducible")
    row_bindings = [
        {
            "sample_id": row["sample_id"],
            "completion_sha256": row["completion_sha256"],
            "answer_sha256": row["answer_sha256"],
            "reference_minutes_sha256": row["reference_minutes_sha256"],
            "generated_token_ids_sha256": row["generated_token_ids_sha256"],
        }
        for row in rows
    ]
    if manifest.get("row_bindings") != row_bindings:
        raise NativeThreeModelEvalError("sealed row bindings are not reproducible")
    return {
        "stage_id": expected_stage_id,
        "model_label": model_label,
        "manifest": manifest,
        "manifest_binding": _file_binding(
            manifest_path, payload_sha256=payload_sha
        ),
        "results": rows,
        "summary": observed_summary,
        "generation_contract": _generation_contract(rows),
    }


def _numeric_delta(after: Any, before: Any) -> float | int:
    if isinstance(after, bool) or isinstance(before, bool):
        return int(bool(after)) - int(bool(before))
    if not isinstance(after, (int, float)) or not isinstance(before, (int, float)):
        raise NativeThreeModelEvalError("contrast metric is not numeric")
    delta = float(after) - float(before)
    if not math.isfinite(delta):
        raise NativeThreeModelEvalError("contrast metric is not finite")
    return round(delta, 8)


def build_comparison(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build a sealed, descriptive three-stage comparison from validated runs."""

    if set(runs) != set(REQUIRED_STAGES):
        raise NativeThreeModelEvalError(
            f"comparison requires exactly these stages: {REQUIRED_STAGES}"
        )
    reference_contract = runs["chk0"].get("generation_contract")
    if not isinstance(reference_contract, Mapping):
        raise NativeThreeModelEvalError("chk0 generation contract is missing")
    for stage_id in REQUIRED_STAGES:
        run = runs[stage_id]
        if run.get("stage_id") != stage_id:
            raise NativeThreeModelEvalError(f"run stage mismatch for {stage_id}")
        if run.get("generation_contract") != reference_contract:
            raise NativeThreeModelEvalError(
                f"generation contract drift between chk0 and {stage_id}"
            )
    reference_rows = runs["chk0"].get("results")
    if not isinstance(reference_rows, list) or not reference_rows:
        raise NativeThreeModelEvalError("chk0 result rows are incomplete")
    expected_ids = [row.get("sample_id") for row in reference_rows]
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for stage_id in REQUIRED_STAGES:
        rows = runs[stage_id].get("results")
        if not isinstance(rows, list) or [row.get("sample_id") for row in rows] != expected_ids:
            raise NativeThreeModelEvalError(f"sample order drift for {stage_id}")
        indexed[stage_id] = {str(row["sample_id"]): row for row in rows}

    aggregate_contrasts: dict[str, Any] = {}
    case_contrasts: dict[str, Any] = {}
    for contrast_id, before_stage, after_stage in PAIRWISE_CONTRASTS:
        before_summary = runs[before_stage]["summary"]
        after_summary = runs[after_stage]["summary"]
        aggregate_contrasts[contrast_id] = {
            "before_stage": before_stage,
            "after_stage": after_stage,
            "deltas": {
                metric: _numeric_delta(
                    after_summary[metric], before_summary[metric]
                )
                for metric in SUMMARY_METRICS
            },
        }
        case_contrasts[contrast_id] = [
            {
                "sample_id": sample_id,
                "length_bucket": indexed[before_stage][sample_id]["length_bucket"],
                "deltas": {
                    metric: _numeric_delta(
                        indexed[after_stage][sample_id][metric],
                        indexed[before_stage][sample_id][metric],
                    )
                    for metric in CASE_METRICS
                },
            }
            for sample_id in expected_ids
        ]

    cohort_summaries: dict[str, Any] = {}
    for cohort_id, identity_value in (
        ("analysis_reference_identity", True),
        ("analysis_reference_non_identity", False),
    ):
        stage_rows = {
            stage_id: [
                row
                for row in runs[stage_id]["results"]
                if bool(row["normalized_identity"]) is identity_value
            ]
            for stage_id in REQUIRED_STAGES
        }
        cohort_summaries[cohort_id] = {
            "normalized_identity": identity_value,
            "cases": len(stage_rows["chk0"]),
            "stage_summaries": {
                stage_id: (
                    _summary(
                        rows,
                        model_label=str(runs[stage_id]["model_label"]),
                        sample_manifest_sha256=str(
                            rows[0]["sample_manifest_sha256"]
                        ),
                    )
                    if rows
                    else None
                )
                for stage_id, rows in stage_rows.items()
            },
        }

    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "task_contract_id": TASK_CONTRACT_ID,
        "stages": list(REQUIRED_STAGES),
        "generation_contract": dict(reference_contract),
        "model_runs": {
            stage_id: {
                "model_label": runs[stage_id]["model_label"],
                "manifest": dict(runs[stage_id]["manifest_binding"]),
                "summary": dict(runs[stage_id]["summary"]),
            }
            for stage_id in REQUIRED_STAGES
        },
        "cases": [
            {
                "sample_id": sample_id,
                "length_bucket": indexed["chk0"][sample_id]["length_bucket"],
                "analysis_reference_exact_identity": indexed["chk0"][sample_id][
                    "analysis_reference_exact_identity"
                ],
                "normalized_identity": indexed["chk0"][sample_id][
                    "normalized_identity"
                ],
                "punctuation_insensitive_identity": indexed["chk0"][sample_id][
                    "punctuation_insensitive_identity"
                ],
                "stages": {
                    stage_id: {
                        "completion_sha256": indexed[stage_id][sample_id][
                            "completion_sha256"
                        ],
                        "answer_sha256": indexed[stage_id][sample_id][
                            "answer_sha256"
                        ],
                        "finish_reason": indexed[stage_id][sample_id]["finish_reason"],
                        "quality_failures": indexed[stage_id][sample_id][
                            "quality_failures"
                        ],
                        "metrics": {
                            metric: indexed[stage_id][sample_id][metric]
                            for metric in CASE_METRICS
                        },
                    }
                    for stage_id in REQUIRED_STAGES
                },
            }
            for sample_id in expected_ids
        ],
        "aggregate_contrasts": aggregate_contrasts,
        "case_contrasts": case_contrasts,
        "cohort_summaries": cohort_summaries,
        "interpretation": {
            "role": "task-aligned diagnostic",
            "semantic_minutes_quality_graded": False,
            "statistical_inference_authorized": False,
            "reason": (
                f"The frozen manifest contains {len(expected_ids)} deterministic "
                "length-stratified cases; it supports structural/fidelity "
                "degeneration checks, not "
                "a population-level semantic-quality claim."
            ),
        },
    }
    return seal_manifest(payload)


def compare_runs(
    *,
    chk0_manifest: Path,
    chk1_manifest: Path,
    chk3_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
    output: Path,
) -> dict[str, Any]:
    runs = {
        "chk0": load_and_validate_run(
            chk0_manifest,
            expected_stage_id="chk0",
            sample_manifest_path=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        ),
        "chk1": load_and_validate_run(
            chk1_manifest,
            expected_stage_id="chk1",
            sample_manifest_path=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        ),
        "chk3": load_and_validate_run(
            chk3_manifest,
            expected_stage_id="chk3",
            sample_manifest_path=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
        ),
    }
    comparison = build_comparison(runs)
    _write_new_json(output, comparison)
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--test-data", required=True, type=Path)
    prepare.add_argument("--test-row-manifest", required=True, type=Path)
    prepare.add_argument("--release-manifest", required=True, type=Path)
    prepare.add_argument("--tokenizer", required=True, type=Path)
    prepare.add_argument("--training-config", required=True, type=Path)
    prepare.add_argument("--samples-per-bucket", type=int, default=4)
    prepare.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--stage-id", choices=REQUIRED_STAGES, required=True)
    run.add_argument("--model-label", required=True)
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--sample-manifest", required=True, type=Path)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--physical-gpu-index", required=True, type=int)
    run.add_argument(
        "--max-new-tokens", type=int, default=native_probe.DEFAULT_MAX_NEW_TOKENS
    )
    run.add_argument("--tail-tokens", type=int, default=native_probe.DEFAULT_TAIL_TOKENS)
    run.add_argument("--seed", type=int, default=native_probe.DEFAULT_SEED)
    run.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    run.add_argument("--resume", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--chk0-manifest", required=True, type=Path)
    compare.add_argument("--chk1-manifest", required=True, type=Path)
    compare.add_argument("--chk3-manifest", required=True, type=Path)
    compare.add_argument("--sample-manifest", required=True, type=Path)
    compare.add_argument("--sample-manifest-sha256", required=True)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.command == "prepare":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(args.tokenizer.expanduser().resolve()),
                local_files_only=True,
                trust_remote_code=True,
            )
            result = build_test_sample_manifest(
                test_data=args.test_data,
                test_row_manifest=args.test_row_manifest,
                release_manifest=args.release_manifest,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                training_config=args.training_config,
                samples_per_bucket=args.samples_per_bucket,
            )
            _write_new_json(args.output, result)
        elif args.command == "run":
            result = run_model(
                stage_id=args.stage_id,
                model_label=args.model_label,
                model_path=args.model,
                adapter_path=args.adapter,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output_dir=args.output_dir,
                physical_gpu_index=args.physical_gpu_index,
                max_new_tokens=args.max_new_tokens,
                tail_tokens=args.tail_tokens,
                seed=args.seed,
                load_in_4bit=args.load_in_4bit,
                attn_implementation=args.attn_implementation,
                resume=args.resume,
            )
        else:
            result = compare_runs(
                chk0_manifest=args.chk0_manifest,
                chk1_manifest=args.chk1_manifest,
                chk3_manifest=args.chk3_manifest,
                sample_manifest=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output=args.output,
            )
        print(
            _canonical_json(
                {
                    "status": result.get("status", "prepared"),
                    "schema_version": result["schema_version"],
                    "payload_sha256": result["integrity"]["payload_sha256"],
                }
            )
        )
        return 0
    except (NativeThreeModelEvalError, OSError, ValueError) as exc:
        print(
            _canonical_json(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
