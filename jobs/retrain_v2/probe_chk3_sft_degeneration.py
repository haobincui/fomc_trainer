"""Redacted, task-aligned quality probe for a chk3 SFT smoke adapter.

The probe compares a chk1 base model with the same model plus one chk3 LoRA
adapter on three immutable validation rows: the shortest, median-length, and
longest prompts under the bound tokenizer and chk3 system prompt.

``prepare`` freezes the three row locations and hashes without copying prompt,
analysis, target, or generated text into the manifest.  ``run`` performs greedy
generation on one visible GPU (4-bit NF4 by default) and persists only hashes,
counts, and quality flags.  ``compare`` applies absolute chk3 output gates and
bounded repetition-delta gates to hash-identical cases.

This is a diagnostic utility.  It never edits the dataset, model, or adapter.
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
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk3_sft_targets import (
    USER_PROMPT_PREFIX,
    _date_values,
    _numeric_values,
)
from jobs.retrain_v2 import probe_chk1_sft_degeneration as common_probe


SCHEMA_VERSION = "chk3-sft-degeneration-probe-v1"
MANIFEST_SCHEMA_VERSION = "chk3-sft-degeneration-samples-v1"
COMPARISON_SCHEMA_VERSION = "chk3-sft-degeneration-comparison-v1"
BOUNDARY = "</think>"
LENGTH_BUCKETS = (("short", 0.0), ("medium", 0.5), ("long", 1.0))
DEFAULT_MAX_NEW_TOKENS = 3072
DEFAULT_TAIL_TOKENS = 1024
DEFAULT_SEED = 20260810
FULL_REPETITION_LIMIT = 0.50
TAIL_REPETITION_LIMIT = 0.60


class Chk3ProbeError(RuntimeError):
    """The chk3 probe could not produce a trustworthy comparison."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Chk3ProbeError(f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Chk3ProbeError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise Chk3ProbeError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise Chk3ProbeError(f"missing regular JSONL file: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Chk3ProbeError(
                        f"blank JSONL line: {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise Chk3ProbeError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Chk3ProbeError(f"cannot read JSONL file {path}: {exc}") from exc
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise Chk3ProbeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def _load_prompt_contract(config_path: Path) -> tuple[str, str | None, str]:
    try:
        return common_probe._load_training_prompt_contract(config_path)
    except common_probe.ProbeError as exc:
        raise Chk3ProbeError(str(exc)) from exc


def _prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        return common_probe._prompt_ids(tokenizer, messages)
    except common_probe.ProbeError as exc:
        raise Chk3ProbeError(str(exc)) from exc


def _messages(
    row: Mapping[str, Any],
    *,
    system_prompt: str,
    user_prompt_suffix: str | None,
) -> list[dict[str, str]]:
    try:
        return common_probe._messages(
            row,
            system_prompt=system_prompt,
            user_prompt_suffix=user_prompt_suffix,
        )
    except common_probe.ProbeError as exc:
        raise Chk3ProbeError(str(exc)) from exc


def extract_source_analysis(prompt: str) -> str:
    """Extract only the chk3 source analysis from its strict user-prompt JSON."""

    if not isinstance(prompt, str) or not prompt.startswith(USER_PROMPT_PREFIX):
        raise Chk3ProbeError("chk3 prompt does not use the frozen rewrite prefix")
    try:
        payload = json.loads(prompt[len(USER_PROMPT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise Chk3ProbeError("chk3 prompt payload is not valid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"analysis"}:
        raise Chk3ProbeError("chk3 prompt payload must contain only analysis")
    analysis = payload.get("analysis")
    if not isinstance(analysis, str) or not analysis.strip():
        raise Chk3ProbeError("chk3 source analysis is empty")
    return analysis.strip()


def _counter_payload(values: Counter[str]) -> list[list[object]]:
    return [[value, values[value]] for value in sorted(values)]


def _numeric_hash(text: str) -> str:
    return common_probe.sha256_text(
        common_probe.canonical_json(_counter_payload(_numeric_values(text)))
    )


def _date_hash(text: str) -> str:
    return common_probe.sha256_text(
        common_probe.canonical_json(sorted(_date_values(text)))
    )


def _tokenizer_fingerprint(tokenizer_path: Path) -> Mapping[str, Any]:
    tokenizer_path = tokenizer_path.expanduser().resolve()
    if not tokenizer_path.is_dir() or tokenizer_path.is_symlink():
        raise Chk3ProbeError(f"invalid tokenizer directory: {tokenizer_path}")
    files: dict[str, str] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
    ):
        path = tokenizer_path / name
        if path.is_file() and not path.is_symlink():
            files[name] = common_probe.sha256_file(path)
    if not files:
        raise Chk3ProbeError(
            f"tokenizer directory has no hashable files: {tokenizer_path}"
        )
    return {"path": str(tokenizer_path), "files": files}


def _select_length_buckets(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(candidates) < 3:
        raise Chk3ProbeError("chk3 validation probe requires at least three rows")
    ordered = sorted(
        candidates,
        key=lambda row: (int(row["prompt_token_count"]), str(row["sample_id"])),
    )
    selected: list[Mapping[str, Any]] = []
    used: set[str] = set()
    for bucket, quantile in LENGTH_BUCKETS:
        target = round((len(ordered) - 1) * quantile)
        available = [
            (abs(index - target), index, row)
            for index, row in enumerate(ordered)
            if str(row["sample_id"]) not in used
        ]
        if not available:
            raise Chk3ProbeError("length-bucket selection is not unique")
        _, _, row = min(available, key=lambda item: (item[0], item[1]))
        selected.append(
            {**row, "length_bucket": bucket, "source_quantile": quantile}
        )
        used.add(str(row["sample_id"]))
    return selected


def build_sample_manifest(
    *,
    validation_data: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    training_config: Path,
) -> Mapping[str, Any]:
    """Build a deterministic, text-free short/medium/long sample manifest."""

    validation_data = validation_data.expanduser().resolve()
    rows = _read_jsonl(validation_data)
    system_prompt, suffix, config_sha = _load_prompt_contract(training_config)
    candidates: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not prompt:
            raise Chk3ProbeError(f"invalid prompt at validation:{line_number}")
        if not isinstance(response, str) or not response:
            raise Chk3ProbeError(f"invalid response at validation:{line_number}")
        analysis = extract_source_analysis(prompt)
        prompt_sha = common_probe.sha256_text(prompt)
        sample_id = f"chk3-validation-{prompt_sha[:24]}"
        if sample_id in seen_ids:
            raise Chk3ProbeError("validation contains duplicate prompt identities")
        seen_ids.add(sample_id)
        prompt_token_count = len(
            _prompt_ids(
                tokenizer,
                _messages(
                    row,
                    system_prompt=system_prompt,
                    user_prompt_suffix=suffix,
                ),
            )
        )
        candidates.append(
            {
                "sample_id": sample_id,
                "line_number": line_number,
                "prompt_sha256": prompt_sha,
                "response_sha256": common_probe.sha256_text(response),
                "analysis_sha256": common_probe.sha256_text(analysis),
                "source_numeric_multiset_sha256": _numeric_hash(analysis),
                "source_date_set_sha256": _date_hash(analysis),
                "source_numeric_occurrences": sum(_numeric_values(analysis).values()),
                "source_date_values": len(_date_values(analysis)),
                "prompt_token_count": prompt_token_count,
            }
        )
    selected = _select_length_buckets(candidates)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "validation_data": {
            "path": str(validation_data),
            "sha256": common_probe.sha256_file(validation_data),
            "rows": len(rows),
        },
        "prompt_contract": {
            "training_config": str(training_config.expanduser().resolve()),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": common_probe.sha256_text(system_prompt),
            "user_prompt_suffix_sha256": (
                common_probe.sha256_text(suffix) if suffix is not None else None
            ),
        },
        "tokenizer": _tokenizer_fingerprint(tokenizer_path),
        "selection": {
            "algorithm": "validation-token-length-short-medium-long-v1",
            "rows": 3,
            "buckets": [item[0] for item in LENGTH_BUCKETS],
        },
        "samples": selected,
    }


def _validate_sample_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise Chk3ProbeError("unsupported chk3 sample-manifest schema")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise Chk3ProbeError("sample-manifest SHA256 is invalid")
    observed = common_probe.sha256_file(path)
    if observed != expected_sha256:
        raise Chk3ProbeError(
            f"sample-manifest SHA256 mismatch: expected={expected_sha256}, "
            f"observed={observed}"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 3:
        raise Chk3ProbeError("sample manifest must contain exactly three rows")
    ids = [row.get("sample_id") for row in samples if isinstance(row, Mapping)]
    buckets = [
        row.get("length_bucket") for row in samples if isinstance(row, Mapping)
    ]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise Chk3ProbeError("sample manifest IDs are invalid")
    if set(buckets) != {"short", "medium", "long"}:
        raise Chk3ProbeError("sample manifest length buckets are invalid")
    return manifest, observed


def _load_bound_rows(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    dataset = manifest.get("validation_data")
    if not isinstance(dataset, Mapping):
        raise Chk3ProbeError("sample manifest has no validation_data binding")
    path = Path(str(dataset.get("path"))).expanduser().resolve()
    if common_probe.sha256_file(path) != dataset.get("sha256"):
        raise Chk3ProbeError("bound validation dataset hash changed")
    rows = _read_jsonl(path)
    if len(rows) != dataset.get("rows"):
        raise Chk3ProbeError("bound validation dataset row count changed")
    bound: dict[str, Mapping[str, Any]] = {}
    for sample in manifest["samples"]:
        line_number = sample.get("line_number")
        if not isinstance(line_number, int) or line_number <= 0:
            raise Chk3ProbeError("sample manifest contains an invalid line number")
        try:
            row = rows[line_number - 1]
        except IndexError as exc:
            raise Chk3ProbeError("sample location is outside validation data") from exc
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise Chk3ProbeError("bound chk3 row schema changed")
        analysis = extract_source_analysis(prompt)
        checks = {
            "prompt_sha256": common_probe.sha256_text(prompt),
            "response_sha256": common_probe.sha256_text(response),
            "analysis_sha256": common_probe.sha256_text(analysis),
            "source_numeric_multiset_sha256": _numeric_hash(analysis),
            "source_date_set_sha256": _date_hash(analysis),
        }
        for key, observed in checks.items():
            if sample.get(key) != observed:
                raise Chk3ProbeError(f"bound sample drift at {key}")
        if sample.get("source_numeric_occurrences") != sum(
            _numeric_values(analysis).values()
        ):
            raise Chk3ProbeError("bound sample numeric-count drift")
        if sample.get("source_date_values") != len(_date_values(analysis)):
            raise Chk3ProbeError("bound sample date-count drift")
        bound[str(sample["sample_id"])] = row
    return bound


def _feature_metrics(source_analysis: str, final_answer: str) -> Mapping[str, Any]:
    source_numbers = _numeric_values(source_analysis)
    answer_numbers = _numeric_values(final_answer)
    missing_numbers = source_numbers - answer_numbers
    unsupported_numbers = answer_numbers - source_numbers
    source_dates = _date_values(source_analysis)
    answer_dates = _date_values(final_answer)
    missing_dates = source_dates - answer_dates
    unsupported_dates = answer_dates - source_dates
    return {
        "source_numeric_occurrences": sum(source_numbers.values()),
        "final_numeric_occurrences": sum(answer_numbers.values()),
        "missing_numeric_occurrences": sum(missing_numbers.values()),
        "unsupported_numeric_occurrences": sum(unsupported_numbers.values()),
        "numeric_multiset_preserved": not missing_numbers
        and not unsupported_numbers,
        "source_numeric_multiset_sha256": _numeric_hash(source_analysis),
        "final_numeric_multiset_sha256": _numeric_hash(final_answer),
        "source_date_values": len(source_dates),
        "final_date_values": len(answer_dates),
        "missing_date_values": len(missing_dates),
        "unsupported_date_values": len(unsupported_dates),
        "date_set_preserved": not missing_dates and not unsupported_dates,
        "source_date_set_sha256": _date_hash(source_analysis),
        "final_date_set_sha256": _date_hash(final_answer),
    }


def analyze_completion(
    *,
    text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int] | set[int] | None,
    max_new_tokens: int,
    source_analysis: str,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> Mapping[str, Any]:
    """Return redaction-safe structural, degeneration, and fidelity metrics."""

    try:
        base = common_probe.analyze_completion(
            text=text,
            generated_token_ids=generated_token_ids,
            eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens,
            tail_tokens=tail_tokens,
        )
    except common_probe.ProbeError as exc:
        raise Chk3ProbeError(str(exc)) from exc
    boundary_count = int(base["think_boundary_count"])
    final_answer = text.split(BOUNDARY, 1)[1].strip() if boundary_count == 1 else ""
    one_paragraph = bool(final_answer) and re.search(
        r"[\r\n\u2028\u2029]", final_answer
    ) is None
    features = _feature_metrics(source_analysis, final_answer)
    failures = list(base.get("catastrophic_reasons") or [])
    if not bool(base["hit_eos"]):
        failures.append("missing_terminal_eos")
    if boundary_count != 1:
        failures.append("think_boundary_count_not_one")
    if not final_answer:
        failures.append("empty_final_answer")
    if final_answer and not one_paragraph:
        failures.append("final_answer_not_single_paragraph")
    if not features["numeric_multiset_preserved"]:
        failures.append("source_numeric_multiset_not_preserved")
    if not features["date_set_preserved"]:
        failures.append("source_date_set_not_preserved")
    failures = sorted(set(failures))
    return {
        **base,
        **features,
        "has_nonempty_final_answer": bool(final_answer),
        "final_answer_single_paragraph": one_paragraph,
        "finish_reason": (
            "eos"
            if base["hit_eos"]
            else "length"
            if base["cap_reached"]
            else "other"
        ),
        "quality_valid": not failures,
        "quality_failures": failures,
    }


def build_redacted_result(
    *,
    text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int] | set[int] | None,
    max_new_tokens: int,
    tail_tokens: int,
    source_analysis: str,
    model_label: str,
    sample_manifest_sha256: str,
    sample: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any]:
    metrics = analyze_completion(
        text=text,
        generated_token_ids=generated_token_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
        source_analysis=source_analysis,
        tail_tokens=tail_tokens,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_label": model_label,
        "sample_manifest_sha256": sample_manifest_sha256,
        "sample_id": sample["sample_id"],
        "length_bucket": sample["length_bucket"],
        "seed": seed,
        "generation_mode": "greedy",
        "max_new_tokens": max_new_tokens,
        "tail_tokens": tail_tokens,
        "prompt_token_count": sample["prompt_token_count"],
        "completion_sha256": common_probe.sha256_text(text),
        **metrics,
    }


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    sample_manifest_sha256: str,
) -> Mapping[str, Any]:
    if len(results) != 3:
        raise Chk3ProbeError("chk3 probe summary requires exactly three rows")
    ids = [str(row.get("sample_id")) for row in results]
    if len(set(ids)) != 3:
        raise Chk3ProbeError("chk3 probe contains duplicate samples")
    if {row.get("length_bucket") for row in results} != {
        "short",
        "medium",
        "long",
    }:
        raise Chk3ProbeError("chk3 probe length buckets are incomplete")
    for row in results:
        if row.get("model_label") != model_label:
            raise Chk3ProbeError("probe row model label mismatch")
        if row.get("sample_manifest_sha256") != sample_manifest_sha256:
            raise Chk3ProbeError("probe row sample-manifest mismatch")
    total = len(results)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "model_label": model_label,
        "sample_manifest_sha256": sample_manifest_sha256,
        "cases": total,
        "quality_valid_rate": sum(bool(row["quality_valid"]) for row in results)
        / total,
        "eos_rate": sum(bool(row["hit_eos"]) for row in results) / total,
        "cap_rate": sum(bool(row["cap_reached"]) for row in results) / total,
        "boundary_valid_rate": sum(
            int(row["think_boundary_count"]) == 1 for row in results
        )
        / total,
        "single_paragraph_final_rate": sum(
            bool(row["final_answer_single_paragraph"]) for row in results
        )
        / total,
        "numeric_preservation_rate": sum(
            bool(row["numeric_multiset_preserved"]) for row in results
        )
        / total,
        "date_preservation_rate": sum(
            bool(row["date_set_preserved"]) for row in results
        )
        / total,
        "periodic_tail_rate": sum(
            bool(row["strict_periodic_tail"]) for row in results
        )
        / total,
        "mean_full_token_4gram_repetition": sum(
            float(row["full_token_4gram_repetition"]) for row in results
        )
        / total,
        "mean_tail_token_4gram_repetition": sum(
            float(row["tail_token_4gram_repetition"]) for row in results
        )
        / total,
        "max_full_token_4gram_repetition": max(
            float(row["full_token_4gram_repetition"]) for row in results
        ),
        "max_tail_token_4gram_repetition": max(
            float(row["tail_token_4gram_repetition"]) for row in results
        ),
    }


def _fingerprint_model(path: Path) -> Mapping[str, Any]:
    path = path.expanduser().resolve()
    weight_names = sorted(
        {
            weight.name
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for weight in path.glob(pattern)
            if weight.is_file() and not weight.is_symlink()
        }
    )
    if not weight_names:
        raise Chk3ProbeError(f"base model has no local weight files: {path}")
    try:
        return common_probe._fingerprint_directory(
            path,
            (
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                *weight_names,
            ),
        )
    except common_probe.ProbeError as exc:
        raise Chk3ProbeError(str(exc)) from exc


def _fingerprint_adapter(path: Path) -> Mapping[str, Any]:
    try:
        return common_probe._fingerprint_directory(
            path,
            (
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
            ),
        )
    except common_probe.ProbeError as exc:
        raise Chk3ProbeError(str(exc)) from exc


def run_generation_probe(
    *,
    base_model: Path,
    adapter: Path | None,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    model_label: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
    seed: int = DEFAULT_SEED,
    load_in_4bit: bool = True,
    attn_implementation: str = "sdpa",
) -> Mapping[str, Any]:
    """Generate three redacted chk3 probe records on one visible GPU."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_label):
        raise Chk3ProbeError("model_label contains unsupported characters")
    if max_new_tokens < 512 or tail_tokens <= 0:
        raise Chk3ProbeError("invalid generation token budget")
    if output_dir.exists() or output_dir.is_symlink():
        raise Chk3ProbeError(f"probe output already exists: {output_dir}")
    manifest, observed_manifest_sha = _validate_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    bound_rows = _load_bound_rows(manifest)
    prompt_contract = manifest.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise Chk3ProbeError("sample manifest has no prompt contract")
    config_path = Path(str(prompt_contract.get("training_config")))
    system_prompt, suffix, config_sha = _load_prompt_contract(config_path)
    if config_sha != prompt_contract.get("training_config_sha256"):
        raise Chk3ProbeError("bound training config changed")
    if common_probe.sha256_text(system_prompt) != prompt_contract.get(
        "system_prompt_sha256"
    ):
        raise Chk3ProbeError("bound chk3 system prompt changed")
    suffix_sha = common_probe.sha256_text(suffix) if suffix is not None else None
    if suffix_sha != prompt_contract.get("user_prompt_suffix_sha256"):
        raise Chk3ProbeError("bound chk3 user prompt suffix changed")

    base_fingerprint = _fingerprint_model(base_model)
    adapter_fingerprint = _fingerprint_adapter(adapter) if adapter else None
    output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "model_label": model_label,
        "base_model": base_fingerprint,
        "adapter": adapter_fingerprint,
        "sample_manifest": {
            "path": str(sample_manifest_path.expanduser().resolve()),
            "sha256": observed_manifest_sha,
        },
        "generation": {
            "do_sample": False,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "tail_tokens": tail_tokens,
            "attn_implementation": attn_implementation,
            "load_in_4bit": load_in_4bit,
        },
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "redaction": {
            "prompt_text_persisted": False,
            "source_analysis_text_persisted": False,
            "completion_text_persisted": False,
            "target_text_persisted": False,
        },
    }
    _write_json(output_dir / "launch.json", launch)

    result_path = output_dir / "results.jsonl"
    results: list[Mapping[str, Any]] = []
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

        if not torch.cuda.is_available():
            raise Chk3ProbeError("CUDA is required for chk3 generation probe")
        if torch.cuda.device_count() != 1:
            raise Chk3ProbeError(
                "expose exactly one GPU with CUDA_VISIBLE_DEVICES"
            )
        tokenizer_record = manifest.get("tokenizer")
        if not isinstance(tokenizer_record, Mapping):
            raise Chk3ProbeError("sample manifest has no tokenizer binding")
        tokenizer_path = Path(str(tokenizer_record.get("path")))
        files = tokenizer_record.get("files")
        if not isinstance(files, Mapping) or not files:
            raise Chk3ProbeError("sample manifest tokenizer files are invalid")
        for name, expected in files.items():
            path = tokenizer_path / str(name)
            if common_probe.sha256_file(path) != expected:
                raise Chk3ProbeError(f"bound tokenizer file changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

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
            str(base_model.expanduser().resolve()),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if adapter is not None:
            model = PeftModel.from_pretrained(
                model,
                str(adapter.expanduser().resolve()),
                is_trainable=False,
                local_files_only=True,
            )
        model.eval()
        model.config.use_cache = True
        eos_value = model.generation_config.eos_token_id
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        try:
            eos_ids = common_probe._normalize_eos_ids(eos_value)
        except common_probe.ProbeError as exc:
            raise Chk3ProbeError(str(exc)) from exc
        if not eos_ids:
            raise Chk3ProbeError("model/tokenizer has no EOS token ID")

        torch.cuda.reset_peak_memory_stats()
        with result_path.open("x", encoding="utf-8") as output_handle:
            for index, sample in enumerate(manifest["samples"]):
                sample_id = str(sample["sample_id"])
                row = bound_rows[sample_id]
                prompt_ids = _prompt_ids(
                    tokenizer,
                    _messages(
                        row,
                        system_prompt=system_prompt,
                        user_prompt_suffix=suffix,
                    ),
                )
                if len(prompt_ids) != sample.get("prompt_token_count"):
                    raise Chk3ProbeError(f"prompt token-count drift for {sample_id}")
                context_limit = int(
                    getattr(model.config, "max_position_embeddings", 0) or 0
                )
                if context_limit and len(prompt_ids) + max_new_tokens > context_limit:
                    raise Chk3ProbeError(
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
                    text = common_probe.decode_completion_preserving_boundary(
                        tokenizer, generated_ids, eos_ids
                    )
                except common_probe.ProbeError as exc:
                    raise Chk3ProbeError(str(exc)) from exc
                source_analysis = extract_source_analysis(str(row["prompt"]))
                result = build_redacted_result(
                    text=text,
                    generated_token_ids=generated_ids,
                    eos_token_ids=eos_ids,
                    max_new_tokens=max_new_tokens,
                    tail_tokens=tail_tokens,
                    source_analysis=source_analysis,
                    model_label=model_label,
                    sample_manifest_sha256=observed_manifest_sha,
                    sample=sample,
                    seed=case_seed,
                )
                output_handle.write(common_probe.canonical_json(result) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                results.append(result)
                del input_ids, attention_mask, sequences

        summary = summarize_results(
            results,
            model_label=model_label,
            sample_manifest_sha256=observed_manifest_sha,
        )
        summary = {
            **summary,
            "created_at_utc": _utc_now(),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3, 3
            ),
            "runtime_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "results": {
                "path": str(result_path.resolve()),
                "sha256": common_probe.sha256_file(result_path),
                "rows": len(results),
            },
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        _write_json(
            output_dir / "failure.json",
            {
                **launch,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_cases": len(results),
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


def compare_results(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    per_case_repetition_delta_limit: float = 0.20,
    mean_repetition_delta_limit: float = 0.10,
) -> Mapping[str, Any]:
    """Apply absolute chk3 quality gates and relative repetition gates."""

    for limit in (per_case_repetition_delta_limit, mean_repetition_delta_limit):
        if not math.isfinite(limit) or limit < 0:
            raise Chk3ProbeError("comparison limits must be finite and non-negative")

    def index(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id in indexed:
                raise Chk3ProbeError(f"duplicate comparison sample: {sample_id}")
            indexed[sample_id] = row
        return indexed

    baseline_index = index(baseline)
    candidate_index = index(candidate)
    if len(baseline_index) != 3 or baseline_index.keys() != candidate_index.keys():
        raise Chk3ProbeError("baseline and candidate must share exactly three cases")
    manifests = {
        str(row.get("sample_manifest_sha256"))
        for row in [*baseline, *candidate]
    }
    if (
        len(manifests) != 1
        or re.fullmatch(r"[0-9a-f]{64}", next(iter(manifests))) is None
    ):
        raise Chk3ProbeError("results are not bound to one sample manifest")

    case_deltas: list[Mapping[str, Any]] = []
    case_failures: list[Mapping[str, Any]] = []
    for sample_id in sorted(baseline_index):
        before = baseline_index[sample_id]
        after = candidate_index[sample_id]
        for key in (
            "length_bucket",
            "seed",
            "generation_mode",
            "max_new_tokens",
            "tail_tokens",
            "prompt_token_count",
        ):
            if before.get(key) != after.get(key):
                raise Chk3ProbeError(
                    f"baseline/candidate generation contract drift at {sample_id}:{key}"
                )
        full_delta = float(after["full_token_4gram_repetition"]) - float(
            before["full_token_4gram_repetition"]
        )
        tail_delta = float(after["tail_token_4gram_repetition"]) - float(
            before["tail_token_4gram_repetition"]
        )
        reasons = list(after.get("quality_failures") or [])
        if float(after["full_token_4gram_repetition"]) >= FULL_REPETITION_LIMIT:
            reasons.append("full_4gram_repetition_ge_0.50")
        if float(after["tail_token_4gram_repetition"]) >= TAIL_REPETITION_LIMIT:
            reasons.append("tail_4gram_repetition_ge_0.60")
        if full_delta > per_case_repetition_delta_limit:
            reasons.append("material_full_repetition_increase")
        if tail_delta > per_case_repetition_delta_limit:
            reasons.append("material_tail_repetition_increase")
        delta = {
            "sample_id": sample_id,
            "length_bucket": after.get("length_bucket"),
            "full_repetition_delta": round(full_delta, 8),
            "tail_repetition_delta": round(tail_delta, 8),
            "baseline_quality_valid": bool(before.get("quality_valid")),
            "candidate_quality_valid": bool(after.get("quality_valid")),
        }
        case_deltas.append(delta)
        if reasons:
            case_failures.append({**delta, "reasons": sorted(set(reasons))})

    baseline_summary = summarize_results(
        baseline,
        model_label=str(baseline[0].get("model_label")),
        sample_manifest_sha256=next(iter(manifests)),
    )
    candidate_summary = summarize_results(
        candidate,
        model_label=str(candidate[0].get("model_label")),
        sample_manifest_sha256=next(iter(manifests)),
    )
    full_mean_delta = float(
        candidate_summary["mean_full_token_4gram_repetition"]
    ) - float(baseline_summary["mean_full_token_4gram_repetition"])
    tail_mean_delta = float(
        candidate_summary["mean_tail_token_4gram_repetition"]
    ) - float(baseline_summary["mean_tail_token_4gram_repetition"])
    aggregate_failures: list[str] = []
    if full_mean_delta > mean_repetition_delta_limit:
        aggregate_failures.append("material_mean_full_repetition_increase")
    if tail_mean_delta > mean_repetition_delta_limit:
        aggregate_failures.append("material_mean_tail_repetition_increase")
    if float(candidate_summary["quality_valid_rate"]) != 1.0:
        aggregate_failures.append("candidate_quality_valid_rate_below_one")
    passed = not case_failures and not aggregate_failures
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "sample_manifest_sha256": next(iter(manifests)),
        "thresholds": {
            "full_4gram_repetition": FULL_REPETITION_LIMIT,
            "tail_4gram_repetition": TAIL_REPETITION_LIMIT,
            "per_case_repetition_delta_limit": per_case_repetition_delta_limit,
            "mean_repetition_delta_limit": mean_repetition_delta_limit,
            "candidate_quality_valid_rate": 1.0,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "aggregate_deltas": {
            "mean_full_repetition": round(full_mean_delta, 8),
            "mean_tail_repetition": round(tail_mean_delta, 8),
        },
        "case_deltas": case_deltas,
        "case_failures": case_failures,
        "aggregate_failures": aggregate_failures,
    }


def _load_results(path: Path) -> list[Mapping[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise Chk3ProbeError(f"probe result file is empty: {path}")
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--validation-data", required=True, type=Path)
    prepare.add_argument("--tokenizer", required=True, type=Path)
    prepare.add_argument("--training-config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--base-model", required=True, type=Path)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--sample-manifest", required=True, type=Path)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--model-label", required=True)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--tail-tokens", type=int, default=DEFAULT_TAIL_TOKENS)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-results", required=True, type=Path)
    compare.add_argument("--candidate-results", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument(
        "--per-case-repetition-delta-limit", type=float, default=0.20
    )
    compare.add_argument("--mean-repetition-delta-limit", type=float, default=0.10)
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
            manifest = build_sample_manifest(
                validation_data=args.validation_data,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                training_config=args.training_config,
            )
            _write_new_json(args.output, manifest)
            print(
                common_probe.canonical_json(
                    {
                        "status": "prepared",
                        "path": str(args.output.resolve()),
                        "sha256": common_probe.sha256_file(args.output),
                        "samples": 3,
                    }
                )
            )
            return 0
        if args.command == "run":
            summary = run_generation_probe(
                base_model=args.base_model,
                adapter=args.adapter,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output_dir=args.output_dir,
                model_label=args.model_label,
                max_new_tokens=args.max_new_tokens,
                tail_tokens=args.tail_tokens,
                seed=args.seed,
                load_in_4bit=args.load_in_4bit,
                attn_implementation=args.attn_implementation,
            )
            print(common_probe.canonical_json(summary))
            return 0
        comparison = compare_results(
            _load_results(args.baseline_results),
            _load_results(args.candidate_results),
            per_case_repetition_delta_limit=args.per_case_repetition_delta_limit,
            mean_repetition_delta_limit=args.mean_repetition_delta_limit,
        )
        comparison = {
            **comparison,
            "inputs": {
                "baseline_results": {
                    "path": str(args.baseline_results.expanduser().resolve()),
                    "sha256": common_probe.sha256_file(args.baseline_results),
                },
                "candidate_results": {
                    "path": str(args.candidate_results.expanduser().resolve()),
                    "sha256": common_probe.sha256_file(args.candidate_results),
                },
            },
        }
        _write_new_json(args.output, comparison)
        print(common_probe.canonical_json(comparison))
        return 0 if comparison["status"] == "passed" else 2
    except (Chk3ProbeError, OSError, ValueError) as exc:
        print(
            common_probe.canonical_json(
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
