"""CPU-only independent audit for the chk3 native three-model evaluator.

The generation runner intentionally limits its online gates to deterministic
delivery, fidelity, and degeneration checks.  This companion performs a
second, inference-free pass over the sealed chk0/chk1/chk3 artifacts.  It
deep-validates the original manifests, recomputes text metrics from the full
persisted generations, stratifies identity and length cohorts, and publishes
an immutable audit bundle.

Run this module from the ``fomc_trainer`` environment so token lengths and
token-ID decoding use the same Transformers implementation as generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval.eval_checkpoint_generation import repetition_rate, rouge_l_f1
from jobs.generation import generate_chk3_sft_targets as target_contract
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-native-three-model-cpu-audit-v1"
ROW_SCHEMA_VERSION = "chk3-native-three-model-cpu-audit-row-v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "chk3-native-three-model-cpu-audit-artifacts-v1"
STAGES = ("chk0", "chk1", "chk3")
PAIRWISE = (
    ("chk0_to_chk1", "chk0", "chk1"),
    ("chk1_to_chk3", "chk1", "chk3"),
    ("chk0_to_chk3", "chk0", "chk3"),
)
EXACT_DELIMITER = "\n</think>\n"
FULL_REPETITION_LIMIT = 0.50
TAIL_REPETITION_LIMIT = 0.60

_MARKDOWN_RE = re.compile(
    r"(?m)^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|```)|\*\*|__"
)
_CITATION_RE = re.compile(
    r"(?i)\bev-[0-9a-f]{6,}\b|\[(?:[^\]\n]*\d[^\]\n]*|[^\]\n]*ev-[^\]\n]*)\]"
)
_JSON_META_RE = re.compile(
    r"(?i)^\s*[\[{]|[\"'](?:answer|reasoning|content|schema)[\"']\s*:"
)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|we|us|our|ours)\b", re.I)


class CpuAuditError(RuntimeError):
    """A sealed input or recomputed audit invariant failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CpuAuditError(f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CpuAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CpuAuditError(f"JSON root must be an object: {path}")
    return value


def _file_binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _file_binding_at(
    source_path: Path, logical_path: Path, *, rows: int | None = None
) -> dict[str, Any]:
    """Hash a staging file while binding its immutable post-rename location."""

    result = _file_binding(source_path, rows=rows)
    result["path"] = str(logical_path.resolve())
    return result


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _punctuation_insensitive_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(
            character
            for character in normalized
            if not unicodedata.category(character).startswith("P")
        ).split()
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    if values and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise CpuAuditError("tokenizer returned invalid token IDs")
    return values


def _ngram_repetition(values: Sequence[int], n: int = 4) -> float:
    if len(values) < n:
        return 0.0
    grams = [tuple(values[index : index + n]) for index in range(len(values) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _split_completion(text: str) -> tuple[str, str]:
    if text.count(EXACT_DELIMITER) != 1:
        return "", ""
    reasoning, answer = text.split(EXACT_DELIMITER, 1)
    return reasoning.strip(), answer.strip()


def _answer_markup_failures(answer: str) -> list[str]:
    failures: list[str] = []
    if not answer:
        return ["empty_answer"]
    if _MARKDOWN_RE.search(answer):
        failures.append("markdown_or_list_markup")
    if _CITATION_RE.search(answer):
        failures.append("citation_or_evidence_id")
    if _JSON_META_RE.search(answer):
        failures.append("json_or_schema_meta")
    control_markers = [
        marker for marker in target_contract.CONTROL_MARKERS if marker in answer
    ]
    if control_markers:
        failures.append("model_control_marker")
    if re.search(r"[\r\n\u2028\u2029]", answer):
        failures.append("not_single_paragraph")
    return sorted(set(failures))


def _reasoning_metrics(reasoning: str, source: str, answer: str) -> dict[str, Any]:
    meta = sorted(target_contract._reasoning_meta_categories(reasoning))
    normalized_reasoning = target_contract._normalized_prose(reasoning)
    normalized_source = target_contract._normalized_prose(source)
    normalized_answer = target_contract._normalized_prose(answer)
    source_copy = bool(
        len(normalized_source) >= 80 and normalized_source in normalized_reasoning
    )
    answer_copy = bool(
        len(normalized_answer) >= 80 and normalized_answer in normalized_reasoning
    )
    return {
        "reasoning_meta_categories": meta,
        "reasoning_meta_free": not meta,
        "reasoning_first_person": bool(_FIRST_PERSON_RE.search(reasoning)),
        "reasoning_contains_complete_source": source_copy,
        "reasoning_contains_complete_answer": answer_copy,
    }


def _copy_metrics(answer: str, source: str, reference: str) -> dict[str, Any]:
    return {
        "answer_source_raw_exact": answer == source,
        "answer_source_normalized_exact": _normalized_text(answer)
        == _normalized_text(source),
        "answer_source_punctuation_insensitive_exact": (
            _punctuation_insensitive_text(answer)
            == _punctuation_insensitive_text(source)
        ),
        "answer_reference_raw_exact": answer == reference,
        "answer_reference_normalized_exact": _normalized_text(answer)
        == _normalized_text(reference),
        "answer_reference_punctuation_insensitive_exact": (
            _punctuation_insensitive_text(answer)
            == _punctuation_insensitive_text(reference)
        ),
    }


def analyze_generation_row(
    row: Mapping[str, Any], *, tokenizer: Any, source_artifact: Path, row_number: int
) -> dict[str, Any]:
    """Recompute independent per-row metrics from persisted full text."""

    text = row.get("generated_text")
    source = row.get("source_analysis")
    reference = row.get("reference_minutes")
    answer = row.get("answer")
    token_ids = row.get("generated_token_ids")
    eos_ids = row.get("eos_token_ids")
    if not all(isinstance(value, str) for value in (text, source, reference, answer)):
        raise CpuAuditError(f"missing text field at {source_artifact}:{row_number}")
    if not isinstance(token_ids, list) or not isinstance(eos_ids, list):
        raise CpuAuditError(f"missing token IDs at {source_artifact}:{row_number}")
    reasoning, parsed_answer = _split_completion(text)
    if parsed_answer != answer:
        raise CpuAuditError(f"persisted answer drift at {source_artifact}:{row_number}")

    try:
        decoded = native_eval.native_probe.common_probe.decode_completion_preserving_boundary(
            tokenizer, token_ids, eos_ids
        )
    except Exception as exc:
        raise CpuAuditError(
            f"token decode failed at {source_artifact}:{row_number}: {exc}"
        ) from exc
    token_decode_exact = decoded == text
    eos_positions = [
        index for index, token_id in enumerate(token_ids) if token_id in set(eos_ids)
    ]
    eos_contract_valid = (
        eos_positions == [len(token_ids) - 1]
        if bool(row.get("hit_eos"))
        else not eos_positions
    )

    reasoning_ids = _token_ids(tokenizer, reasoning)
    answer_ids = _token_ids(tokenizer, answer)
    source_ids = _token_ids(tokenizer, source)
    reference_ids = _token_ids(tokenizer, reference)
    markup_failures = _answer_markup_failures(answer)
    reasoning_checks = _reasoning_metrics(reasoning, source, answer)
    copy_checks = _copy_metrics(answer, source, reference)

    source_attributions = target_contract._attribution_categories(source)
    answer_attributions = target_contract._attribution_categories(answer)
    unsupported_attributions = sorted(answer_attributions - source_attributions)
    dropped_attributions = sorted(source_attributions - answer_attributions)

    numeric_preserved = target_contract._numeric_values(source) == target_contract._numeric_values(
        answer
    )
    date_preserved = target_contract._date_values(source) == target_contract._date_values(
        answer
    )
    # The runner's signed-surface diagnostic deliberately compares every raw
    # numeric spelling, not only explicitly signed values.  It is useful for
    # debugging but too strict for the native contract, which permits exact
    # number-form/unit conversions.  Keep it as a diagnostic and do not fold it
    # into independent fidelity validity.
    signed = native_eval._signed_numeric_metrics(source, answer)
    exact_delimiter = text.count(EXACT_DELIMITER) == 1
    delivery_valid = all(
        (
            bool(row.get("hit_eos")),
            not bool(row.get("cap_reached")),
            exact_delimiter,
            bool(reasoning),
            bool(answer),
            bool(row.get("final_answer_single_paragraph")),
            token_decode_exact,
            eos_contract_valid,
        )
    )
    degeneration_free = all(
        (
            not bool(row.get("strict_periodic_tail")),
            float(row.get("full_token_4gram_repetition", 1.0))
            < FULL_REPETITION_LIMIT,
            float(row.get("tail_token_4gram_repetition", 1.0))
            < TAIL_REPETITION_LIMIT,
        )
    )
    fidelity_valid = all(
        (
            numeric_preserved,
            date_preserved,
            not unsupported_attributions,
        )
    )
    answer_format_valid = not markup_failures
    core_native_valid = all(
        (delivery_valid, degeneration_free, fidelity_valid, answer_format_valid)
    )
    reasoning_length_valid = 64 <= len(reasoning_ids) <= 2400
    extended_prompt_contract_valid = all(
        (
            core_native_valid,
            reasoning_length_valid,
            bool(reasoning_checks["reasoning_meta_free"]),
            not reasoning_checks["reasoning_contains_complete_source"],
        )
    )
    rouge_eligible = delivery_valid and bool(answer)
    answer_reference_rouge = rouge_l_f1(answer, reference) if rouge_eligible else None
    answer_source_rouge = rouge_l_f1(answer, source) if rouge_eligible else None

    core_failures: list[str] = []
    if not delivery_valid:
        core_failures.append("delivery_invalid")
    if not degeneration_free:
        core_failures.append("degeneration_detected")
    if not numeric_preserved:
        core_failures.append("numeric_multiset_not_preserved")
    if not date_preserved:
        core_failures.append("date_set_not_preserved")
    if unsupported_attributions:
        core_failures.append("unsupported_attribution_category")
    core_failures.extend(f"answer_format:{value}" for value in markup_failures)
    extended_failures = list(core_failures)
    if not reasoning_length_valid:
        extended_failures.append("reasoning_token_length_outside_64_2400")
    if not reasoning_checks["reasoning_meta_free"]:
        extended_failures.append("reasoning_meta_present")
    if reasoning_checks["reasoning_contains_complete_source"]:
        extended_failures.append("reasoning_contains_complete_source")

    reference_source_rouge = rouge_l_f1(reference, source)
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "stage_id": row["stage_id"],
        "model_label": row["model_label"],
        "sample_id": row["sample_id"],
        "length_bucket": row["length_bucket"],
        "analysis_reference_exact_identity": bool(
            row["analysis_reference_exact_identity"]
        ),
        "normalized_identity": bool(row["normalized_identity"]),
        "punctuation_insensitive_identity": bool(
            row["punctuation_insensitive_identity"]
        ),
        "source_artifact": str(source_artifact.resolve()),
        "source_artifact_row": row_number,
        "source_analysis_sha256": _sha256_text(source),
        "reference_minutes_sha256": _sha256_text(reference),
        "answer_sha256": _sha256_text(answer),
        "completion_sha256": _sha256_text(text),
        "token_decode_exact": token_decode_exact,
        "eos_contract_valid": eos_contract_valid,
        "delivery_valid": delivery_valid,
        "degeneration_free": degeneration_free,
        "fidelity_valid": fidelity_valid,
        "runner_quality_valid": bool(row["quality_valid"]),
        "runner_native_structure_valid": bool(row["native_structure_valid"]),
        "runner_quality_failures": list(row["quality_failures"]),
        "answer_format_valid": answer_format_valid,
        "core_native_valid": core_native_valid,
        "extended_prompt_contract_valid": extended_prompt_contract_valid,
        "rouge_eligible": rouge_eligible,
        "core_failures": sorted(set(core_failures)),
        "audit_failures": sorted(set(extended_failures)),
        "answer_format_failures": markup_failures,
        "reasoning_tokens": len(reasoning_ids),
        "answer_tokens": len(answer_ids),
        "source_tokens": len(source_ids),
        "reference_tokens": len(reference_ids),
        "answer_reference_token_ratio": (
            len(answer_ids) / len(reference_ids) if reference_ids else None
        ),
        "answer_source_token_ratio": (
            len(answer_ids) / len(source_ids) if source_ids else None
        ),
        "full_word_trigram_repetition": round(repetition_rate(text, ngram_size=3), 8),
        "reasoning_word_trigram_repetition": round(
            repetition_rate(reasoning, ngram_size=3), 8
        ),
        "answer_word_trigram_repetition": round(
            repetition_rate(answer, ngram_size=3), 8
        ),
        "reasoning_token_4gram_repetition": round(
            _ngram_repetition(reasoning_ids, 4), 8
        ),
        "answer_token_4gram_repetition": round(_ngram_repetition(answer_ids, 4), 8),
        "runner_full_token_4gram_repetition": row["full_token_4gram_repetition"],
        "runner_tail_token_4gram_repetition": row["tail_token_4gram_repetition"],
        "strict_periodic_tail": bool(row["strict_periodic_tail"]),
        "finish_reason": row["finish_reason"],
        "raw_generated_tokens": row["raw_generated_tokens"],
        "content_tokens": row["content_tokens"],
        "think_boundary_count": row["think_boundary_count"],
        "exact_boundary_delimiter_count": row["exact_boundary_delimiter_count"],
        "numeric_multiset_preserved": numeric_preserved,
        "date_set_preserved": date_preserved,
        "signed_numeric_surface_preserved": bool(
            signed["signed_numeric_surface_preserved"]
        ),
        "attribution_no_unsupported": not unsupported_attributions,
        "source_attribution_categories": sorted(source_attributions),
        "answer_attribution_categories": sorted(answer_attributions),
        "unsupported_attribution_categories": unsupported_attributions,
        "dropped_attribution_categories": dropped_attributions,
        **reasoning_checks,
        **copy_checks,
        "answer_reference_rouge_l_f1": answer_reference_rouge,
        "answer_source_rouge_l_f1": answer_source_rouge,
        "reference_source_rouge_l_f1": reference_source_rouge,
    }


def _median(values: Sequence[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean(values: Sequence[float | int]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"eligible": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "eligible": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0}
    boolean_fields = (
        "delivery_valid",
        "degeneration_free",
        "fidelity_valid",
        "runner_quality_valid",
        "runner_native_structure_valid",
        "answer_format_valid",
        "core_native_valid",
        "extended_prompt_contract_valid",
        "rouge_eligible",
        "numeric_multiset_preserved",
        "date_set_preserved",
        "signed_numeric_surface_preserved",
        "attribution_no_unsupported",
        "reasoning_meta_free",
        "reasoning_first_person",
        "reasoning_contains_complete_source",
        "reasoning_contains_complete_answer",
        "answer_source_raw_exact",
        "answer_source_punctuation_insensitive_exact",
        "answer_reference_raw_exact",
        "answer_reference_punctuation_insensitive_exact",
        "strict_periodic_tail",
    )
    result: dict[str, Any] = {"cases": len(rows)}
    for field in boolean_fields:
        count = sum(bool(row[field]) for row in rows)
        result[f"{field}_count"] = count
        result[f"{field}_rate"] = count / len(rows)
    for field in (
        "reasoning_tokens",
        "answer_tokens",
        "answer_reference_token_ratio",
        "answer_word_trigram_repetition",
        "reasoning_word_trigram_repetition",
        "answer_token_4gram_repetition",
        "reasoning_token_4gram_repetition",
        "runner_full_token_4gram_repetition",
        "runner_tail_token_4gram_repetition",
    ):
        values = [
            row[field]
            for row in rows
            if isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
            and math.isfinite(float(row[field]))
        ]
        result[field] = _numeric_summary(values)
    for field in (
        "answer_reference_rouge_l_f1",
        "answer_source_rouge_l_f1",
        "reference_source_rouge_l_f1",
    ):
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and math.isfinite(float(row[field]))
        ]
        result[field] = _numeric_summary(values)
    failure_counts: Counter[str] = Counter()
    meta_counts: Counter[str] = Counter()
    unsupported_attribution_counts: Counter[str] = Counter()
    dropped_attribution_counts: Counter[str] = Counter()
    for row in rows:
        failure_counts.update(row["audit_failures"])
        meta_counts.update(row["reasoning_meta_categories"])
        unsupported_attribution_counts.update(row["unsupported_attribution_categories"])
        dropped_attribution_counts.update(row["dropped_attribution_categories"])
    result["failure_counts"] = dict(sorted(failure_counts.items()))
    result["reasoning_meta_category_counts"] = dict(sorted(meta_counts.items()))
    result["unsupported_attribution_category_counts"] = dict(
        sorted(unsupported_attribution_counts.items())
    )
    result["dropped_attribution_category_counts"] = dict(
        sorted(dropped_attribution_counts.items())
    )
    return result


def _paired_contrast(
    rows_by_stage: Mapping[str, Mapping[str, Mapping[str, Any]]],
    before: str,
    after: str,
) -> dict[str, Any]:
    ids = list(rows_by_stage[before])
    if set(ids) != set(rows_by_stage[after]):
        raise CpuAuditError(f"paired sample universe drift: {before} vs {after}")
    deltas: list[float] = []
    wins = ties = losses = 0
    core_improved = core_regressed = 0
    cases: list[dict[str, Any]] = []
    for sample_id in ids:
        left = rows_by_stage[before][sample_id]
        right = rows_by_stage[after][sample_id]
        delta: float | None = None
        if left["rouge_eligible"] and right["rouge_eligible"]:
            delta = float(right["answer_reference_rouge_l_f1"]) - float(
                left["answer_reference_rouge_l_f1"]
            )
            deltas.append(delta)
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1
        core_delta = int(bool(right["core_native_valid"])) - int(
            bool(left["core_native_valid"])
        )
        core_improved += core_delta > 0
        core_regressed += core_delta < 0
        cases.append(
            {
                "sample_id": sample_id,
                "length_bucket": left["length_bucket"],
                "identity": left["analysis_reference_exact_identity"],
                "answer_reference_rouge_l_f1_delta": (
                    round(delta, 12) if delta is not None else None
                ),
                "core_native_valid_delta": core_delta,
                "answer_word_trigram_repetition_delta": round(
                    float(right["answer_word_trigram_repetition"])
                    - float(left["answer_word_trigram_repetition"]),
                    12,
                ),
            }
        )
    return {
        "before_stage": before,
        "after_stage": after,
        "cases": len(ids),
        "rouge_l": {
            "eligible_pairs": len(deltas),
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "min_delta": min(deltas) if deltas else None,
            "max_delta": max(deltas) if deltas else None,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "core_native_valid_improved_cases": core_improved,
        "core_native_valid_regressed_cases": core_regressed,
        "case_deltas": cases,
    }


def _validate_comparison(
    comparison_path: Path, runs: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], str]:
    comparison = _read_json(comparison_path)
    try:
        payload_sha = validate_manifest_integrity(comparison)
    except Exception as exc:
        raise CpuAuditError(f"comparison manifest integrity failed: {exc}") from exc
    rebuilt = native_eval.build_comparison(runs)
    observed_payload = {
        key: value
        for key, value in comparison.items()
        if key not in {"created_at_utc", "integrity"}
    }
    rebuilt_payload = {
        key: value
        for key, value in rebuilt.items()
        if key not in {"created_at_utc", "integrity"}
    }
    if observed_payload != rebuilt_payload:
        raise CpuAuditError("comparison payload is not reproducible from the three runs")
    return comparison, payload_sha


def _format_fraction(summary: Mapping[str, Any], field: str) -> str:
    return f"{summary.get(field + '_count', 0)}/{summary.get('cases', 0)}"


def _format_float(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _build_report(summary: Mapping[str, Any]) -> str:
    stages = summary["cohorts"]["all"]["stages"]
    chk1 = stages["chk1"]
    chk3 = stages["chk3"]
    chk3_regressed = chk3["core_native_valid_count"] < chk1["core_native_valid_count"]
    delivery_regressed = chk3["delivery_valid_count"] < chk1["delivery_valid_count"]
    runner_quality_regressed = (
        chk3["runner_quality_valid_count"] < chk1["runner_quality_valid_count"]
    )
    if chk3_regressed or delivery_regressed or runner_quality_regressed:
        verdict = (
            "chk3 checkpoint-250 在原生 analysis → Minutes N12 上相对 chk1 出现了"
            "可复现的交付/退化回归，不能判定为无退化。"
        )
    elif chk3["core_native_valid_count"] < chk3["cases"]:
        verdict = "chk3 checkpoint-250 未全量通过原生合同，仍存在失败样本。"
    else:
        verdict = "chk3 checkpoint-250 在本次 N12 诊断上通过核心原生合同。"

    lines = [
        "# chk3 原生 analysis → Minutes 三模型 N12 独立 CPU 审计",
        "",
        "## 结论",
        "",
        verdict,
        "",
        (
            "本审计深验了三路 sealed run manifest 与 comparison，重新从完整生成文本、"
            "原始 token IDs、source analysis 和 teacher reference 计算指标。N12 是确定性"
            "长度分层诊断，不授权总体显著性或因果结论。"
        ),
        "",
        "## 核心交付与保真",
        "",
        "| 模型 | Delivery | Runner quality | 独立核心合同 | 扩展 prompt 合同 | 数字 | 日期 | Attribution 无新增 | 周期尾 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "chk0": "chk0",
        "chk1": "chk1 cp200",
        "chk3": "chk3 cp250",
    }
    for stage in STAGES:
        item = stages[stage]
        lines.append(
            "| {label} | {delivery} | {runner_quality} | {core} | {extended} | {numeric} | {date} | "
            "{attribute} | {periodic} |".format(
                label=labels[stage],
                delivery=_format_fraction(item, "delivery_valid"),
                runner_quality=_format_fraction(item, "runner_quality_valid"),
                core=_format_fraction(item, "core_native_valid"),
                extended=_format_fraction(item, "extended_prompt_contract_valid"),
                numeric=_format_fraction(item, "numeric_multiset_preserved"),
                date=_format_fraction(item, "date_set_preserved"),
                attribute=_format_fraction(item, "attribution_no_unsupported"),
                periodic=_format_fraction(item, "strict_periodic_tail"),
            )
        )

    lines.extend(
        [
            "",
            "扩展 prompt 合同还要求 reasoning 为 64–2,400 tokens、无禁止元话语，且不完整复制 source；它比在线核心门禁更严格。",
            "",
            "## Final answer 与 reference",
            "",
            "| 模型 | ROUGE-L all | ROUGE-L non-identity N11 | Answer tokens p50/max | Answer word-3gram rep | Source exact copy | Reference exact copy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    nonidentity = summary["cohorts"]["non_identity_n11"]["stages"]
    for stage in STAGES:
        all_item = stages[stage]
        non_item = nonidentity[stage]
        answer_tokens = all_item["answer_tokens"]
        answer_rep = all_item["answer_word_trigram_repetition"]
        lines.append(
            "| {label} | {all_rouge} | {non_rouge} | {median}/{maximum} | {rep} | {source_copy} | {ref_copy} |".format(
                label=labels[stage],
                all_rouge=_format_float(
                    all_item["answer_reference_rouge_l_f1"]["mean"]
                ),
                non_rouge=_format_float(
                    non_item["answer_reference_rouge_l_f1"]["mean"]
                ),
                median=_format_float(answer_tokens["median"], 1),
                maximum=_format_float(answer_tokens["max"], 0),
                rep=_format_float(answer_rep["mean"]),
                source_copy=_format_fraction(all_item, "answer_source_raw_exact"),
                ref_copy=_format_fraction(all_item, "answer_reference_raw_exact"),
            )
        )

    lines.extend(["", "## 最关键的相邻阶段回归证据", ""])
    critical = summary["critical_regressions"]
    if not critical["delivery"] and not critical["fidelity"]:
        lines.append("- chk1 → chk3 未发现 delivery 或 deterministic fidelity 回归。")
    for item in critical["delivery"]:
        lines.append(
            f"- Delivery：`{item['sample_id']}`（{item['length_bucket']}）在 chk1 "
            f"以 `{item['chk1_raw_generated_tokens']}` tokens 正常 EOS；chk3 生成 "
            f"`{item['chk3_raw_generated_tokens']}` tokens 后 `{item['chk3_finish_reason']}`，"
            f"boundary `{item['chk3_think_boundary_count']}`，strict periodic tail "
            f"`{str(item['chk3_strict_periodic_tail']).lower()}`，full/tail token-4gram "
            f"`{item['chk3_full_repetition']:.6f}/{item['chk3_tail_repetition']:.6f}`。"
        )
    for item in critical["fidelity"]:
        lines.append(
            f"- Fidelity：`{item['sample_id']}`（{item['length_bucket']}）chk1 deterministic "
            f"fidelity 通过、chk3 失败：{', '.join(item['chk3_core_failures'])}。"
        )

    lines.extend(["", "## Paired contrasts", ""])
    for contrast_id, contrast in summary["paired_contrasts"].items():
        rouge = contrast["rouge_l"]
        lines.append(
            f"- `{contrast_id}`：ROUGE-L eligible `{rouge['eligible_pairs']}`，"
            f"mean Δ `{_format_float(rouge['mean_delta'])}`，"
            f"win/tie/loss `{rouge['wins']}/{rouge['ties']}/{rouge['losses']}`；"
            f"核心合同 improved/regressed `{contrast['core_native_valid_improved_cases']}/"
            f"{contrast['core_native_valid_regressed_cases']}`。"
        )

    lines.extend(["", "## 失败样本", ""])
    failures = summary["core_failure_inventory"]
    if not failures:
        lines.append("- 无。")
    else:
        for item in failures:
            lines.append(
                f"- `{item['stage_id']}` / `{item['sample_id']}` / `{item['length_bucket']}`："
                + ", ".join(item["audit_failures"])
            )

    baseline = summary["sample_baseline"]
    lines.extend(
        [
            "",
            "## 样本与解释边界",
            "",
            f"- N12 中 raw exact identity 为 `{baseline['raw_exact_identity_cases']}/12`；主结果同时给出 non-identity N11。",
            f"- Source 共 `{baseline['source_numeric_occurrences']}` 个 canonical numeric occurrences、"
            f"`{baseline['source_date_values']}` 个 date values；coarse attribution 只覆盖 "
            f"`{baseline['source_attribution_covered_cases']}/12`。",
            f"- Teacher reference vs source ROUGE-L 均值为 `{baseline['reference_source_rouge_l_f1_mean']:.6f}`；reference 是合成 Minutes-style target，不是官方 Minutes。",
            "- Attribution 只是类别级正则 proxy；没有本地 source-only semantic judge，不能据此证明完整 factual、causal 或 direction fidelity。",
            "- N12 很小且按 teacher completion length 分层，只用于退化诊断；不得做总体显著性推断或 canonical DAG 晋升。",
            "",
            "## 工件与哈希",
            "",
            f"- Run root：`{summary['run_root']}`",
            f"- Sample manifest SHA-256：`{summary['inputs']['sample_manifest']['sha256']}`",
            f"- Comparison SHA-256：`{summary['inputs']['comparison']['sha256']}`",
            f"- Audit rows SHA-256：`{summary['artifacts']['audit_rows']['sha256']}`",
            f"- Audit summary payload SHA-256：`{summary['integrity']['payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(*, run_root: Path, output_dir: Path) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not run_root.is_dir() or run_root.is_symlink():
        raise CpuAuditError(f"invalid run root: {run_root}")
    if output_dir.exists() or output_dir.is_symlink():
        raise CpuAuditError(f"refusing to overwrite audit output: {output_dir}")
    if output_dir.parent != run_root:
        raise CpuAuditError("audit output must be a direct child of run root")

    stage_manifest_paths = {
        stage: run_root / stage / "manifest.json" for stage in STAGES
    }
    first_manifest = _read_json(stage_manifest_paths["chk0"])
    sample_binding = first_manifest.get("sample_manifest")
    if not isinstance(sample_binding, Mapping):
        raise CpuAuditError("chk0 manifest has no sample binding")
    sample_manifest_path = Path(str(sample_binding.get("path"))).resolve()
    sample_manifest_sha = str(sample_binding.get("sha256"))
    if _sha256_file(sample_manifest_path) != sample_manifest_sha:
        raise CpuAuditError("sample manifest SHA256 mismatch")

    runs: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        runs[stage] = native_eval.load_and_validate_run(
            stage_manifest_paths[stage],
            expected_stage_id=stage,
            sample_manifest_path=sample_manifest_path,
            sample_manifest_sha256=sample_manifest_sha,
        )
    comparison_path = run_root / "comparison.json"
    comparison, comparison_payload_sha = _validate_comparison(comparison_path, runs)

    sample_manifest = _read_json(sample_manifest_path)
    tokenizer_record = sample_manifest.get("tokenizer")
    if not isinstance(tokenizer_record, Mapping):
        raise CpuAuditError("sample manifest has no tokenizer record")
    tokenizer_path = Path(str(tokenizer_record.get("path"))).resolve()
    from transformers import AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )

    audit_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        source_artifact = run_root / stage / "generations.jsonl"
        for row_number, row in enumerate(runs[stage]["results"], start=1):
            audit_rows.append(
                analyze_generation_row(
                    row,
                    tokenizer=tokenizer,
                    source_artifact=source_artifact,
                    row_number=row_number,
                )
            )

    indexed: dict[str, dict[str, dict[str, Any]]] = {
        stage: {
            str(row["sample_id"]): row
            for row in audit_rows
            if row["stage_id"] == stage
        }
        for stage in STAGES
    }
    ids = list(indexed["chk0"])
    if any(list(indexed[stage]) != ids for stage in STAGES):
        raise CpuAuditError("audit sample order drift across stages")

    cohort_filters = {
        "all": lambda row: True,
        "non_identity_n11": lambda row: not row["analysis_reference_exact_identity"],
        "identity_n1": lambda row: row["analysis_reference_exact_identity"],
        "length_short": lambda row: row["length_bucket"] == "short",
        "length_medium": lambda row: row["length_bucket"] == "medium",
        "length_long": lambda row: row["length_bucket"] == "long",
    }
    cohorts: dict[str, Any] = {}
    for cohort_id, predicate in cohort_filters.items():
        stage_summaries = {
            stage: summarize_rows(
                [row for row in audit_rows if row["stage_id"] == stage and predicate(row)]
            )
            for stage in STAGES
        }
        cases = stage_summaries["chk0"]["cases"]
        if any(stage_summaries[stage]["cases"] != cases for stage in STAGES):
            raise CpuAuditError(f"cohort denominator drift: {cohort_id}")
        cohorts[cohort_id] = {"cases_per_stage": cases, "stages": stage_summaries}

    baseline_rows = [row for row in audit_rows if row["stage_id"] == "chk0"]
    source_numeric_occurrences = 0
    source_date_values = 0
    source_attribution_covered_cases = 0
    for row in runs["chk0"]["results"]:
        source = str(row["source_analysis"])
        source_numeric_occurrences += sum(target_contract._numeric_values(source).values())
        source_date_values += len(target_contract._date_values(source))
        source_attribution_covered_cases += bool(
            target_contract._attribution_categories(source)
        )

    paired = {
        contrast_id: _paired_contrast(indexed, before, after)
        for contrast_id, before, after in PAIRWISE
    }
    extended_failure_inventory = [
        {
            "stage_id": row["stage_id"],
            "sample_id": row["sample_id"],
            "length_bucket": row["length_bucket"],
            "core_native_valid": row["core_native_valid"],
            "extended_prompt_contract_valid": row["extended_prompt_contract_valid"],
            "audit_failures": row["audit_failures"],
        }
        for row in audit_rows
        if not row["core_native_valid"] or not row["extended_prompt_contract_valid"]
    ]
    core_failure_inventory = [
        item for item in extended_failure_inventory if not item["core_native_valid"]
    ]
    for item in core_failure_inventory:
        row = indexed[item["stage_id"]][item["sample_id"]]
        item["core_failures"] = list(row["core_failures"])
        item["audit_failures"] = list(row["core_failures"])

    delivery_regressions: list[dict[str, Any]] = []
    fidelity_regressions: list[dict[str, Any]] = []
    for sample_id in ids:
        chk1_row = indexed["chk1"][sample_id]
        chk3_row = indexed["chk3"][sample_id]
        if chk1_row["delivery_valid"] and not chk3_row["delivery_valid"]:
            delivery_regressions.append(
                {
                    "sample_id": sample_id,
                    "length_bucket": chk3_row["length_bucket"],
                    "chk1_raw_generated_tokens": chk1_row["raw_generated_tokens"],
                    "chk1_finish_reason": chk1_row["finish_reason"],
                    "chk3_raw_generated_tokens": chk3_row["raw_generated_tokens"],
                    "chk3_finish_reason": chk3_row["finish_reason"],
                    "chk3_think_boundary_count": chk3_row["think_boundary_count"],
                    "chk3_strict_periodic_tail": chk3_row["strict_periodic_tail"],
                    "chk3_full_repetition": chk3_row[
                        "runner_full_token_4gram_repetition"
                    ],
                    "chk3_tail_repetition": chk3_row[
                        "runner_tail_token_4gram_repetition"
                    ],
                }
            )
        if chk1_row["fidelity_valid"] and not chk3_row["fidelity_valid"]:
            fidelity_regressions.append(
                {
                    "sample_id": sample_id,
                    "length_bucket": chk3_row["length_bucket"],
                    "chk3_core_failures": list(chk3_row["core_failures"]),
                }
            )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=str(run_root))
    )
    try:
        rows_path = staging / "audit_rows.jsonl"
        with rows_path.open("x", encoding="utf-8") as handle:
            for row in audit_rows:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        run_bindings = {
            stage: {
                "manifest": _file_binding(stage_manifest_paths[stage]),
                "manifest_payload_sha256": runs[stage]["manifest"]["integrity"][
                    "payload_sha256"
                ],
                "generations": _file_binding(
                    run_root / stage / "generations.jsonl", rows=12
                ),
            }
            for stage in STAGES
        }
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "question": "chk3 native analysis-to-Minutes chk0/chk1/chk3 N12 comparison",
            "run_root": str(run_root),
            "deep_validation": {
                "three_run_manifests": "passed",
                "comparison_rebuild": "passed",
                "token_id_decode_exact": all(
                    row["token_decode_exact"] for row in audit_rows
                ),
                "same_sample_order": True,
                "cpu_only": True,
            },
            "runtime": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "transformers": transformers.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "tokenizer_path": str(tokenizer_path),
            },
            "inputs": {
                "sample_manifest": _file_binding(sample_manifest_path),
                "comparison": {
                    **_file_binding(comparison_path),
                    "payload_sha256": comparison_payload_sha,
                },
                "runs": run_bindings,
            },
            "sample_baseline": {
                "cases": 12,
                "raw_exact_identity_cases": sum(
                    row["analysis_reference_exact_identity"] for row in baseline_rows
                ),
                "normalized_identity_cases": sum(
                    row["normalized_identity"] for row in baseline_rows
                ),
                "punctuation_insensitive_identity_cases": sum(
                    row["punctuation_insensitive_identity"] for row in baseline_rows
                ),
                "source_numeric_occurrences": source_numeric_occurrences,
                "source_date_values": source_date_values,
                "source_attribution_covered_cases": source_attribution_covered_cases,
                "reference_source_rouge_l_f1_mean": float(
                    statistics.fmean(
                        row["reference_source_rouge_l_f1"] for row in baseline_rows
                    )
                ),
                "reference_is_synthetic_teacher_target": True,
            },
            "cohorts": cohorts,
            "paired_contrasts": paired,
            "critical_regressions": {
                "delivery": delivery_regressions,
                "fidelity": fidelity_regressions,
            },
            "core_failure_inventory": core_failure_inventory,
            "extended_failure_inventory": extended_failure_inventory,
            "limitations": {
                "statistical_inference_authorized": False,
                "semantic_source_only_judge_run": False,
                "attribution_metric": "coarse deterministic category proxy",
                "signed_numeric_coverage": "surface-only; selected N12 has no explicit +/- source occurrence",
                "reference": "synthetic DeepSeek Minutes-style target, not official Minutes",
            },
            "artifacts": {
                "audit_rows": _file_binding_at(
                    rows_path, output_dir / "audit_rows.jsonl", rows=len(audit_rows)
                )
            },
        }
        sealed_summary = seal_manifest(payload)
        summary_path = staging / "audit_summary.json"
        summary_path.write_text(
            json.dumps(sealed_summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        with summary_path.open("rb") as handle:
            os.fsync(handle.fileno())

        report_path = staging / "native_analysis_to_minutes_report.md"
        report_path.write_text(_build_report(sealed_summary), encoding="utf-8")
        with report_path.open("rb") as handle:
            os.fsync(handle.fileno())

        artifact_payload = {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "artifacts": {
                "audit_rows": _file_binding_at(
                    rows_path, output_dir / "audit_rows.jsonl", rows=len(audit_rows)
                ),
                "audit_summary": {
                    **_file_binding_at(
                        summary_path, output_dir / "audit_summary.json"
                    ),
                    "payload_sha256": sealed_summary["integrity"]["payload_sha256"],
                },
                "report": _file_binding_at(
                    report_path, output_dir / "native_analysis_to_minutes_report.md"
                ),
            },
        }
        artifact_manifest = seal_manifest(artifact_payload)
        artifact_path = staging / "artifact_manifest.json"
        artifact_path.write_text(
            json.dumps(
                artifact_manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with artifact_path.open("rb") as handle:
            os.fsync(handle.fileno())

        if output_dir.exists() or output_dir.is_symlink():
            raise CpuAuditError(f"audit output appeared concurrently: {output_dir}")
        os.rename(staging, output_dir)
        return {
            "output_dir": str(output_dir),
            "summary": sealed_summary,
            "artifact_manifest": artifact_manifest,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.run_root / "cpu_audit"
    result = run_audit(run_root=args.run_root, output_dir=output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": result["output_dir"],
                "summary_payload_sha256": result["summary"]["integrity"][
                    "payload_sha256"
                ],
                "artifact_manifest_payload_sha256": result["artifact_manifest"][
                    "integrity"
                ]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
