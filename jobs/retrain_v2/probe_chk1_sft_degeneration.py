"""Task-aligned generation probe for chk1 SFT degeneration.

This utility deliberately uses Transformers + PEFT instead of vLLM.  It has
three separately auditable phases:

``prepare``
    Select six immutable eval/test prompts (short/median/long for both changed
    and unchanged clean-release rows) and bind them to dataset/tokenizer/config
    hashes without copying prompt or target text into the manifest.
``run``
    Generate from chk0 alone or chk0 plus one LoRA adapter, then record output
    contract and token-level repetition measurements.
``compare``
    Apply a narrow degeneration gate.  The candidate fails on any non-finite
    result, length-cap completion, strict periodic tail, catastrophic 4-gram
    repetition, or a material increase relative to the hash-identical baseline.

The probe is diagnostic only: it never edits a dataset, adapter, or model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

from open_r1.trainer.prompt_contract import compose_user_prompt


SCHEMA_VERSION = "chk1-sft-degeneration-probe-v1"
MANIFEST_SCHEMA_VERSION = "chk1-sft-degeneration-samples-v1"
COMPARISON_SCHEMA_VERSION = "chk1-sft-degeneration-comparison-v1"
BOUNDARY = "</think>"
DEFAULT_MAX_NEW_TOKENS = 3072
DEFAULT_TAIL_TOKENS = 1024
DEFAULT_SYSTEM_PROMPT = (
    "Analyze only the supplied point-in-time FOMC evidence. Do not infer facts "
    "from later releases or from the target meeting's Minutes. Use the model's "
    "native reasoning boundary, then provide a concise, evidence-grounded "
    "final analysis."
)
SPLITS = ("eval", "test")
QUANTILES = (("short", 0.0), ("median", 0.5), ("long", 1.0))


class ProbeError(RuntimeError):
    """The probe could not produce a trustworthy, comparable result."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProbeError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"missing regular JSONL file: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ProbeError(f"blank JSONL line: {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ProbeError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read JSONL file {path}: {exc}") from exc
    return rows


def _resolve_child(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProbeError("manifest path must be a non-empty string")
    raw = Path(relative)
    if raw.is_absolute():
        raise ProbeError(f"manifest child path must be relative: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ProbeError(f"manifest child escapes candidate directory: {relative}")
    if resolved.is_symlink() or not resolved.is_file():
        raise ProbeError(f"manifest child is not a regular file: {resolved}")
    return resolved


def _require_hash(path: Path, expected: object, *, label: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ProbeError(f"{label} has an invalid SHA256")
    observed = sha256_file(path)
    if observed != expected:
        raise ProbeError(
            f"{label} SHA256 mismatch: expected={expected}, observed={observed}"
        )
    return observed


def _prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=False,
        )
    except Exception as exc:
        raise ProbeError(f"chat-template tokenization failed: {exc}") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], Sequence) and not isinstance(
        value[0], (str, bytes)
    ):
        if len(value) != 1:
            raise ProbeError("chat template returned an unexpected batch")
        value = value[0]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProbeError("chat template returned invalid token IDs")
    try:
        result = [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise ProbeError("chat template returned non-integer token IDs") from exc
    if not result or any(token_id < 0 for token_id in result):
        raise ProbeError("chat template returned empty or negative token IDs")
    return result


def _messages(
    row: Mapping[str, Any],
    *,
    system_prompt: str,
    user_prompt_suffix: str | None,
) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ProbeError("dataset row has no non-empty prompt")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": compose_user_prompt(prompt, user_prompt_suffix),
        },
    ]


def _load_training_prompt_contract(config_path: Path) -> tuple[str, str | None, str]:
    if not config_path.is_file() or config_path.is_symlink():
        raise ProbeError(f"missing regular training config: {config_path}")
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProbeError(f"cannot load training YAML {config_path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ProbeError("training config root must be an object")
    system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    suffix = config.get("user_prompt_suffix")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ProbeError("training config system_prompt must be a non-empty string")
    if suffix is not None and not isinstance(suffix, str):
        raise ProbeError("training config user_prompt_suffix must be a string or null")
    return system_prompt, suffix, sha256_file(config_path)


def _load_candidate_rows(
    candidate_dir: Path,
) -> tuple[
    Mapping[str, Any],
    Path,
    dict[str, list[Mapping[str, Any]]],
    dict[str, list[Mapping[str, Any]]],
]:
    candidate_dir = candidate_dir.expanduser().resolve()
    if not candidate_dir.is_dir() or candidate_dir.is_symlink():
        raise ProbeError(f"candidate directory is missing or is a symlink: {candidate_dir}")
    manifest_path = candidate_dir / "candidate_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "chk1-clean-sft-candidate-v2":
        raise ProbeError("unsupported clean candidate manifest schema")
    if manifest.get("immutable_candidate") is not True:
        raise ProbeError("candidate is not marked immutable")
    if manifest.get("quality_status") not in {
        "pending_semantic_audit",
        "failed_semantic_audit",
    }:
        raise ProbeError(
            "degeneration probe only accepts the explicit clean-candidate states"
        )

    split_records = manifest.get("split_files")
    if not isinstance(split_records, Mapping):
        raise ProbeError("candidate manifest has no split_files object")
    split_rows: dict[str, list[Mapping[str, Any]]] = {}
    for split in SPLITS:
        record = split_records.get(split)
        if not isinstance(record, Mapping):
            raise ProbeError(f"candidate manifest has no {split} split record")
        path = _resolve_child(candidate_dir, record.get("path"))
        _require_hash(path, record.get("sha256"), label=f"candidate {split}")
        rows = _read_jsonl(path)
        if record.get("rows") != len(rows):
            raise ProbeError(f"candidate {split} row-count mismatch")
        split_rows[split] = rows

    repair_record = manifest.get("repair_manifest")
    if not isinstance(repair_record, Mapping):
        raise ProbeError("candidate manifest has no repair_manifest record")
    repair_path = _resolve_child(candidate_dir, repair_record.get("path"))
    _require_hash(
        repair_path,
        repair_record.get("sha256"),
        label="candidate repair manifest",
    )
    repair_rows = _read_jsonl(repair_path)
    if repair_record.get("rows") != len(repair_rows):
        raise ProbeError("repair-manifest row-count mismatch")
    repairs_by_split = {
        split: [row for row in repair_rows if row.get("split") == split]
        for split in SPLITS
    }
    for split in SPLITS:
        if len(repairs_by_split[split]) != len(split_rows[split]):
            raise ProbeError(f"candidate/repair row-count mismatch for {split}")
        for line_number, (row, repair) in enumerate(
            zip(split_rows[split], repairs_by_split[split], strict=True), start=1
        ):
            for field, repair_field in (
                ("prompt", "prompt_sha256"),
                ("provided_data", "provided_data_sha256"),
                ("response", "new_response_sha256"),
            ):
                value = row.get(field)
                if not isinstance(value, str):
                    raise ProbeError(
                        f"candidate field {field!r} is invalid at {split}:{line_number}"
                    )
                if sha256_text(value) != repair.get(repair_field):
                    raise ProbeError(
                        f"candidate/repair hash mismatch at {split}:{line_number}:{field}"
                    )
    return manifest, manifest_path, split_rows, repairs_by_split


def _choose_quantiles(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if len(candidates) < len(QUANTILES):
        raise ProbeError("need at least three rows for each changed/unchanged group")
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["prompt_token_count"]),
            str(item["sample_id"]),
        ),
    )
    chosen: list[Mapping[str, Any]] = []
    used: set[str] = set()
    for label, quantile in QUANTILES:
        target = round((len(ordered) - 1) * quantile)
        available = [
            (abs(index - target), index, item)
            for index, item in enumerate(ordered)
            if str(item["sample_id"]) not in used
        ]
        _, _, selected = min(available, key=lambda value: (value[0], value[1]))
        selected = dict(selected)
        selected["length_bucket"] = label
        selected["source_quantile"] = quantile
        chosen.append(selected)
        used.add(str(selected["sample_id"]))

    # Preserve actual min/max.  If all three happen to come from one split,
    # replace only the median with the nearest row from the missing split.
    selected_splits = {str(item["split"]) for item in chosen}
    if len(selected_splits) == 1:
        missing_split = next(split for split in SPLITS if split not in selected_splits)
        median_tokens = int(chosen[1]["prompt_token_count"])
        alternatives = [
            item
            for item in ordered
            if item["split"] == missing_split
            and str(item["sample_id"]) not in {str(chosen[0]["sample_id"]), str(chosen[2]["sample_id"])}
        ]
        if not alternatives:
            raise ProbeError("cannot split-balance selected eval/test prompts")
        replacement = min(
            alternatives,
            key=lambda item: (
                abs(int(item["prompt_token_count"]) - median_tokens),
                str(item["sample_id"]),
            ),
        )
        chosen[1] = {
            **replacement,
            "length_bucket": "median",
            "source_quantile": 0.5,
        }
    return chosen


def build_sample_manifest(
    *,
    candidate_dir: Path,
    candidate_manifest_sha256: str,
    tokenizer: Any,
    tokenizer_path: Path,
    training_config: Path,
    sample_seed: int = 20260809,
    greedy_longest: int = 2,
) -> Mapping[str, Any]:
    """Return a text-free, hash-bound sample manifest."""

    manifest, manifest_path, split_rows, repairs_by_split = _load_candidate_rows(
        candidate_dir
    )
    observed_manifest_sha = _require_hash(
        manifest_path,
        candidate_manifest_sha256,
        label="candidate manifest",
    )
    system_prompt, suffix, config_sha = _load_training_prompt_contract(training_config)
    if greedy_longest < 0 or greedy_longest > 6:
        raise ProbeError("greedy_longest must be in 0..6")

    all_candidates: dict[bool, list[Mapping[str, Any]]] = {False: [], True: []}
    for split in SPLITS:
        for line_number, (row, repair) in enumerate(
            zip(split_rows[split], repairs_by_split[split], strict=True), start=1
        ):
            sample_id = repair.get("sample_id")
            changed = repair.get("changed")
            if not isinstance(sample_id, str) or not sample_id:
                raise ProbeError(f"invalid sample_id at {split}:{line_number}")
            if not isinstance(changed, bool):
                raise ProbeError(f"invalid changed flag at {split}:{line_number}")
            prompt_ids = _prompt_ids(
                tokenizer,
                _messages(
                    row,
                    system_prompt=system_prompt,
                    user_prompt_suffix=suffix,
                ),
            )
            all_candidates[changed].append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "line_number": line_number,
                    "changed": changed,
                    "prompt_sha256": repair["prompt_sha256"],
                    "provided_data_sha256": repair["provided_data_sha256"],
                    "response_sha256": repair["new_response_sha256"],
                    "prompt_token_count": len(prompt_ids),
                }
            )

    selected: list[dict[str, Any]] = []
    for changed in (False, True):
        for item in _choose_quantiles(all_candidates[changed]):
            selected.append(dict(item))
    if len(selected) != 6 or len({item["sample_id"] for item in selected}) != 6:
        raise ProbeError("sample selection did not produce six unique rows")

    longest_ids = {
        item["sample_id"]
        for item in sorted(
            selected,
            key=lambda item: (
                -int(item["prompt_token_count"]),
                str(item["sample_id"]),
            ),
        )[:greedy_longest]
    }
    for index, item in enumerate(selected):
        item["sampling_seed"] = sample_seed + index
        item["run_greedy"] = item["sample_id"] in longest_ids

    tokenizer_path = tokenizer_path.expanduser().resolve()
    tokenizer_files: dict[str, str] = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
    ):
        path = tokenizer_path / name
        if path.is_file() and not path.is_symlink():
            tokenizer_files[name] = sha256_file(path)
    if not tokenizer_files:
        raise ProbeError(f"tokenizer directory has no hashable files: {tokenizer_path}")

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "candidate": {
            "path": str(Path(candidate_dir).expanduser().resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": observed_manifest_sha,
            "candidate_id": manifest.get("candidate_id"),
            "quality_status": manifest.get("quality_status"),
            "split_files": {split: manifest["split_files"][split] for split in SPLITS},
            "repair_manifest": manifest["repair_manifest"],
        },
        "prompt_contract": {
            "training_config": str(training_config.expanduser().resolve()),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_suffix_sha256": (
                sha256_text(suffix) if suffix is not None else None
            ),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "files": tokenizer_files,
        },
        "selection": {
            "algorithm": "changed-x-short-median-long-v1",
            "source_splits": list(SPLITS),
            "rows": 6,
            "changed_rows": 3,
            "unchanged_rows": 3,
            "greedy_longest": greedy_longest,
            "sample_seed": sample_seed,
        },
        "samples": selected,
    }


def _ngram_repetition(token_ids: Sequence[int], *, n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def has_strict_periodic_tail(token_ids: Sequence[int]) -> bool:
    """Detect a literal repeated suffix of at least 48 tokens and 3 periods."""

    size = len(token_ids)
    for period in range(1, min(128, size // 3) + 1):
        if period * 3 < 48:
            continue
        unit = list(token_ids[size - period :])
        repeats = 1
        cursor = size - 2 * period
        while cursor >= 0 and list(token_ids[cursor : cursor + period]) == unit:
            repeats += 1
            cursor -= period
        if repeats >= 3 and repeats * period >= 48:
            return True
    return False


def _normalize_eos_ids(
    eos_token_ids: int | Iterable[int] | None,
) -> set[int]:
    """Return EOS IDs as a set and remain idempotent for internal callers.

    The generation path normalizes ``GenerationConfig.eos_token_id`` once and
    then passes that set to both decoding and metric analysis.  ``set`` is an
    iterable but is intentionally not a ``collections.abc.Sequence``; limiting
    this helper to ``Sequence`` therefore made the second normalization fail
    before the first completion could be recorded.  Integral also covers NumPy
    integer scalars sometimes returned by tokenizer integrations.
    """

    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, Integral) and not isinstance(eos_token_ids, bool):
        result = {int(eos_token_ids)}
    elif isinstance(eos_token_ids, (str, bytes, Mapping)) or not isinstance(
        eos_token_ids, Iterable
    ):
        raise ProbeError("EOS token IDs must be an integer or iterable of integers")
    else:
        try:
            values = list(eos_token_ids)
        except TypeError as exc:
            raise ProbeError(
                "EOS token IDs must be an integer or iterable of integers"
            ) from exc
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise ProbeError("EOS token IDs contain a non-integer")
        result = {int(value) for value in values}
    if any(value < 0 for value in result):
        raise ProbeError("EOS token IDs contain a negative integer")
    return result


def decode_completion_preserving_boundary(
    tokenizer: Any,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Iterable[int] | None,
) -> str:
    """Decode a completion without losing DeepSeek's native think boundary.

    DeepSeek represents ``</think>`` as token 128014.  Whether an individual
    Transformers release considers that token "special" is tokenizer-version
    dependent, so ``skip_special_tokens=True`` is unsafe for this gate.  Remove
    only the terminal EOS ID ourselves and decode all remaining tokens with
    special-token skipping disabled.  This preserves the boundary while also
    ensuring that the persisted completion never contains the final EOS text.
    """

    try:
        token_ids = [int(value) for value in generated_token_ids]
    except (TypeError, ValueError) as exc:
        raise ProbeError("generated token IDs contain a non-integer") from exc
    eos_ids = _normalize_eos_ids(eos_token_ids)
    content_ids = token_ids[:-1] if token_ids and token_ids[-1] in eos_ids else token_ids
    try:
        value = tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise ProbeError(f"completion decoding failed: {exc}") from exc
    if not isinstance(value, str):
        raise ProbeError("tokenizer decode did not return text")
    eos_token = getattr(tokenizer, "eos_token", None)
    if isinstance(eos_token, str) and eos_token and value.endswith(eos_token):
        raise ProbeError("decoded completion unexpectedly retains terminal EOS")
    return value


def analyze_completion(
    *,
    text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Iterable[int] | None,
    max_new_tokens: int,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> Mapping[str, Any]:
    if max_new_tokens <= 0 or tail_tokens <= 0:
        raise ProbeError("token limits must be positive")
    try:
        raw_ids = [int(value) for value in generated_token_ids]
    except (TypeError, ValueError) as exc:
        raise ProbeError("generated token IDs contain a non-integer") from exc
    finite = bool(raw_ids) and bool(text) and all(value >= 0 for value in raw_ids)
    eos_ids = _normalize_eos_ids(eos_token_ids)
    hit_eos = bool(raw_ids and raw_ids[-1] in eos_ids)
    content_ids = raw_ids[:-1] if hit_eos else raw_ids
    cap_reached = len(raw_ids) >= max_new_tokens and not hit_eos
    boundary_count = text.count(BOUNDARY)
    answer = text.split(BOUNDARY, 1)[1].strip() if boundary_count == 1 else ""
    full_repetition = _ngram_repetition(content_ids, n=4)
    tail_repetition = _ngram_repetition(content_ids[-tail_tokens:], n=4)
    periodic = has_strict_periodic_tail(content_ids)
    catastrophic_reasons: list[str] = []
    if not finite:
        catastrophic_reasons.append("non_finite_or_empty_output")
    if cap_reached:
        catastrophic_reasons.append("completion_length_cap")
    if periodic:
        catastrophic_reasons.append("strict_periodic_tail")
    if full_repetition >= 0.50:
        catastrophic_reasons.append("full_4gram_repetition_ge_0.50")
    if tail_repetition >= 0.60:
        catastrophic_reasons.append("tail_4gram_repetition_ge_0.60")
    return {
        "status": "ok" if finite else "invalid",
        "raw_generated_tokens": len(raw_ids),
        "content_tokens": len(content_ids),
        "hit_eos": hit_eos,
        "cap_reached": cap_reached,
        "think_boundary_count": boundary_count,
        "has_nonempty_answer": bool(answer),
        "contract_valid": boundary_count == 1 and bool(answer),
        "full_token_4gram_repetition": round(full_repetition, 8),
        "tail_token_count": min(len(content_ids), tail_tokens),
        "tail_token_4gram_repetition": round(tail_repetition, 8),
        "strict_periodic_tail": periodic,
        "catastrophic_reasons": catastrophic_reasons,
    }


def _fingerprint_directory(path: Path, names: Sequence[str]) -> Mapping[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise ProbeError(f"model/adapter directory is missing or is a symlink: {path}")
    files: dict[str, str] = {}
    for name in names:
        item = path / name
        if item.is_file() and not item.is_symlink():
            files[name] = sha256_file(item)
    if not files:
        raise ProbeError(f"no provenance files found under {path}")
    return {"path": str(path), "files": files}


def _validate_sample_manifest(
    sample_manifest_path: Path,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    sample_manifest = _read_json(sample_manifest_path)
    if sample_manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ProbeError("unsupported sample-manifest schema")
    observed = _require_hash(
        sample_manifest_path,
        expected_sha256,
        label="sample manifest",
    )
    samples = sample_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 6:
        raise ProbeError("sample manifest must contain exactly six samples")
    ids = [item.get("sample_id") for item in samples if isinstance(item, Mapping)]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ProbeError("sample manifest IDs are missing or duplicated")
    return sample_manifest, observed


def _load_bound_rows(
    sample_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    candidate = sample_manifest.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ProbeError("sample manifest has no candidate object")
    candidate_dir = Path(str(candidate.get("path")))
    _, manifest_path, split_rows, repairs_by_split = _load_candidate_rows(candidate_dir)
    _require_hash(
        manifest_path,
        candidate.get("manifest_sha256"),
        label="bound candidate manifest",
    )
    bound: dict[str, Mapping[str, Any]] = {}
    for sample in sample_manifest["samples"]:
        split = sample.get("split")
        line_number = sample.get("line_number")
        if split not in SPLITS or not isinstance(line_number, int) or line_number <= 0:
            raise ProbeError("sample manifest contains an invalid row location")
        try:
            row = split_rows[split][line_number - 1]
            repair = repairs_by_split[split][line_number - 1]
        except IndexError as exc:
            raise ProbeError("sample row location is outside the bound split") from exc
        sample_id = sample.get("sample_id")
        if repair.get("sample_id") != sample_id:
            raise ProbeError(f"sample ID drift at {split}:{line_number}")
        for field, key in (
            ("prompt", "prompt_sha256"),
            ("provided_data", "provided_data_sha256"),
            ("response", "response_sha256"),
        ):
            value = row.get(field)
            if not isinstance(value, str) or sha256_text(value) != sample.get(key):
                raise ProbeError(f"sample field drift at {split}:{line_number}:{field}")
        bound[str(sample_id)] = row
    return bound


def _generation_cases(sample_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases: list[Mapping[str, Any]] = []
    for sample in sample_manifest["samples"]:
        cases.append(
            {
                "sample_id": sample["sample_id"],
                "mode": "sampled",
                "seed": sample["sampling_seed"],
                "temperature": 0.6,
                "top_p": 0.95,
            }
        )
        if sample.get("run_greedy") is True:
            cases.append(
                {
                    "sample_id": sample["sample_id"],
                    "mode": "greedy",
                    "seed": sample["sampling_seed"],
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            )
    return cases


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    load_in_4bit: bool = True,
    attn_implementation: str = "sdpa",
) -> Mapping[str, Any]:
    """Load one model and generate all hash-bound diagnostic cases."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_label):
        raise ProbeError("model_label contains unsupported characters")
    if max_new_tokens < 512:
        raise ProbeError("max_new_tokens below 512 cannot test long-reasoning degeneration")
    if output_dir.exists():
        raise ProbeError(f"probe output already exists: {output_dir}")

    sample_manifest, observed_sample_sha = _validate_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    bound_rows = _load_bound_rows(sample_manifest)
    prompt_contract = sample_manifest["prompt_contract"]
    training_config = Path(str(prompt_contract["training_config"]))
    system_prompt, suffix, config_sha = _load_training_prompt_contract(training_config)
    if config_sha != prompt_contract.get("training_config_sha256"):
        raise ProbeError("training prompt config changed after sample selection")
    if sha256_text(system_prompt) != prompt_contract.get("system_prompt_sha256"):
        raise ProbeError("training system prompt changed after sample selection")
    suffix_sha = sha256_text(suffix) if suffix is not None else None
    if suffix_sha != prompt_contract.get("user_prompt_suffix_sha256"):
        raise ProbeError("training user prompt suffix changed after sample selection")

    base_model = base_model.expanduser().resolve()
    weight_names = sorted(
        {
            path.name
            for pattern in ("*.safetensors", "pytorch_model*.bin")
            for path in base_model.glob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )
    if not weight_names:
        raise ProbeError(f"base model has no local weight files: {base_model}")
    base_fingerprint = _fingerprint_directory(
        base_model,
        (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            *weight_names,
        ),
    )
    adapter_fingerprint = (
        _fingerprint_directory(
            adapter,
            (
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
            ),
        )
        if adapter is not None
        else None
    )
    # Create the immutable run directory only after all static paths and hashes
    # pass; a bad receipt/hash must not strand an empty directory that blocks a
    # corrected retry.
    output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": utc_now(),
        "model_label": model_label,
        "base_model": base_fingerprint,
        "adapter": adapter_fingerprint,
        "sample_manifest": {
            "path": str(sample_manifest_path.expanduser().resolve()),
            "sha256": observed_sample_sha,
        },
        "generation": {
            "sampled": {"temperature": 0.6, "top_p": 0.95},
            "greedy": {"temperature": 0.0, "top_p": 1.0},
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
    }
    _write_json(output_dir / "launch.json", launch)

    result_path = output_dir / "results.jsonl"
    results: list[Mapping[str, Any]] = []
    started = datetime.now(timezone.utc)
    model = None
    try:
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise ProbeError("CUDA is required for the generation probe")
        if torch.cuda.device_count() != 1:
            raise ProbeError(
                "Expose exactly one GPU per generation-probe process with "
                "CUDA_VISIBLE_DEVICES"
            )
        tokenizer_path = Path(str(sample_manifest["tokenizer"]["path"]))
        for name, expected in sample_manifest["tokenizer"]["files"].items():
            _require_hash(tokenizer_path / name, expected, label=f"tokenizer {name}")
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
        eos_ids = _normalize_eos_ids(eos_value)
        if not eos_ids:
            raise ProbeError("model/tokenizer has no EOS token ID")

        torch.cuda.reset_peak_memory_stats()
        with result_path.open("x", encoding="utf-8") as output_handle:
            for case in _generation_cases(sample_manifest):
                sample_id = str(case["sample_id"])
                sample = next(
                    item
                    for item in sample_manifest["samples"]
                    if item["sample_id"] == sample_id
                )
                row = bound_rows[sample_id]
                messages = _messages(
                    row,
                    system_prompt=system_prompt,
                    user_prompt_suffix=suffix,
                )
                prompt_ids = _prompt_ids(tokenizer, messages)
                if len(prompt_ids) != sample["prompt_token_count"]:
                    raise ProbeError(f"prompt token-count drift for {sample_id}")
                context_limit = int(
                    getattr(model.config, "max_position_embeddings", 0) or 0
                )
                if context_limit and len(prompt_ids) + max_new_tokens > context_limit:
                    raise ProbeError(
                        f"context overflow for {sample_id}: prompt={len(prompt_ids)}, "
                        f"completion={max_new_tokens}, limit={context_limit}"
                    )
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
                attention_mask = torch.ones_like(input_ids)
                seed = int(case["seed"])
                transformers.set_seed(seed)
                generation_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": sorted(eos_ids),
                    "use_cache": True,
                }
                if case["mode"] == "sampled":
                    generation_kwargs.update(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                    )
                else:
                    generation_kwargs.update(do_sample=False)
                with torch.inference_mode():
                    sequences = model.generate(**generation_kwargs)
                generated_ids = sequences[0, input_ids.shape[1] :].tolist()
                text = decode_completion_preserving_boundary(
                    tokenizer, generated_ids, eos_ids
                )
                metrics = analyze_completion(
                    text=text,
                    generated_token_ids=generated_ids,
                    eos_token_ids=eos_ids,
                    max_new_tokens=max_new_tokens,
                    tail_tokens=tail_tokens,
                )
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "model_label": model_label,
                    "sample_manifest_sha256": observed_sample_sha,
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "changed": sample["changed"],
                    "length_bucket": sample["length_bucket"],
                    "mode": case["mode"],
                    "seed": seed,
                    "prompt_token_count": len(prompt_ids),
                    "completion_sha256": sha256_text(text),
                    "completion": text,
                    **metrics,
                }
                output_handle.write(canonical_json(result) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                results.append(result)
                del input_ids, attention_mask, sequences

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        summary = summarize_probe_results(
            results,
            model_label=model_label,
            sample_manifest_sha256=observed_sample_sha,
        )
        summary = {
            **summary,
            "created_at_utc": utc_now(),
            "elapsed_seconds": round(elapsed, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "runtime_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "results": {
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
                "rows": len(results),
            },
        }
        _write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        failure = {
            **launch,
            "status": "failed",
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_cases": len(results),
        }
        _write_json(output_dir / "failure.json", failure)
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


def summarize_probe_results(
    results: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    sample_manifest_sha256: str,
) -> Mapping[str, Any]:
    if not results:
        raise ProbeError("cannot summarize an empty probe")
    expected_cases = {
        (str(row.get("sample_id")), str(row.get("mode")), int(row.get("seed", -1)))
        for row in results
    }
    if len(expected_cases) != len(results):
        raise ProbeError("probe contains duplicate generation cases")
    for row in results:
        if row.get("model_label") != model_label:
            raise ProbeError("probe row model-label mismatch")
        if row.get("sample_manifest_sha256") != sample_manifest_sha256:
            raise ProbeError("probe row sample-manifest mismatch")
    total = len(results)
    catastrophic = sum(bool(row.get("catastrophic_reasons")) for row in results)
    return {
        "schema_version": SCHEMA_VERSION,
        # Metric collection is complete even if it found a catastrophic row.
        # Only ``compare`` is a pass/fail gate; this lets a degenerate baseline
        # remain available as the control instead of aborting the workflow.
        "status": "complete",
        "model_label": model_label,
        "sample_manifest_sha256": sample_manifest_sha256,
        "cases": total,
        "sampled_cases": sum(row.get("mode") == "sampled" for row in results),
        "greedy_cases": sum(row.get("mode") == "greedy" for row in results),
        "finite_rate": sum(row.get("status") == "ok" for row in results) / total,
        "eos_rate": sum(bool(row.get("hit_eos")) for row in results) / total,
        "cap_rate": sum(bool(row.get("cap_reached")) for row in results) / total,
        "contract_valid_rate": sum(bool(row.get("contract_valid")) for row in results)
        / total,
        "periodic_tail_rate": sum(bool(row.get("strict_periodic_tail")) for row in results)
        / total,
        "catastrophic_cases": catastrophic,
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


def compare_probe_results(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    per_case_repetition_delta_limit: float = 0.20,
    mean_repetition_delta_limit: float = 0.10,
    contract_rate_drop_limit: float = 0.25,
) -> Mapping[str, Any]:
    """Compare identical generation cases and enforce the degeneration gate."""

    if not baseline or not candidate:
        raise ProbeError("baseline and candidate results must both be non-empty")
    for limit in (
        per_case_repetition_delta_limit,
        mean_repetition_delta_limit,
        contract_rate_drop_limit,
    ):
        if not math.isfinite(limit) or limit < 0:
            raise ProbeError("comparison limits must be finite and non-negative")

    def index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
        indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
        for row in rows:
            key = (
                str(row.get("sample_id")),
                str(row.get("mode")),
                int(row.get("seed", -1)),
            )
            if key in indexed:
                raise ProbeError(f"duplicate comparison case: {key}")
            indexed[key] = row
        return indexed

    baseline_index = index(baseline)
    candidate_index = index(candidate)
    if baseline_index.keys() != candidate_index.keys():
        raise ProbeError("baseline and candidate generation cases do not match")
    manifests = {
        str(row.get("sample_manifest_sha256"))
        for row in [*baseline, *candidate]
    }
    if (
        len(manifests) != 1
        or "None" in manifests
        or re.fullmatch(r"[0-9a-f]{64}", next(iter(manifests))) is None
    ):
        raise ProbeError("baseline and candidate are not bound to one sample manifest")

    failures: list[Mapping[str, Any]] = []
    deltas: list[Mapping[str, Any]] = []
    for key in sorted(baseline_index):
        before = baseline_index[key]
        after = candidate_index[key]
        reasons = list(after.get("catastrophic_reasons") or [])
        if after.get("status") != "ok" and "non_finite_or_empty_output" not in reasons:
            reasons.append("candidate_status_not_ok")
        if bool(after.get("cap_reached")):
            reasons.append("completion_length_cap")
        if bool(after.get("strict_periodic_tail")):
            reasons.append("strict_periodic_tail")
        if float(after["full_token_4gram_repetition"]) >= 0.50:
            reasons.append("full_4gram_repetition_ge_0.50")
        if float(after["tail_token_4gram_repetition"]) >= 0.60:
            reasons.append("tail_4gram_repetition_ge_0.60")
        full_delta = float(after["full_token_4gram_repetition"]) - float(
            before["full_token_4gram_repetition"]
        )
        tail_delta = float(after["tail_token_4gram_repetition"]) - float(
            before["tail_token_4gram_repetition"]
        )
        if full_delta > per_case_repetition_delta_limit:
            reasons.append("material_full_repetition_increase")
        if tail_delta > per_case_repetition_delta_limit:
            reasons.append("material_tail_repetition_increase")
        delta = {
            "sample_id": key[0],
            "mode": key[1],
            "seed": key[2],
            "full_repetition_delta": round(full_delta, 8),
            "tail_repetition_delta": round(tail_delta, 8),
            "baseline_contract_valid": bool(before.get("contract_valid")),
            "candidate_contract_valid": bool(after.get("contract_valid")),
        }
        deltas.append(delta)
        if reasons:
            failures.append({**delta, "reasons": sorted(set(reasons))})

    baseline_summary = summarize_probe_results(
        baseline,
        model_label=str(baseline[0].get("model_label")),
        sample_manifest_sha256=str(baseline[0].get("sample_manifest_sha256")),
    )
    candidate_summary = summarize_probe_results(
        candidate,
        model_label=str(candidate[0].get("model_label")),
        sample_manifest_sha256=str(candidate[0].get("sample_manifest_sha256")),
    )
    aggregate_failures: list[str] = []
    full_mean_delta = float(
        candidate_summary["mean_full_token_4gram_repetition"]
    ) - float(baseline_summary["mean_full_token_4gram_repetition"])
    tail_mean_delta = float(
        candidate_summary["mean_tail_token_4gram_repetition"]
    ) - float(baseline_summary["mean_tail_token_4gram_repetition"])
    if full_mean_delta > mean_repetition_delta_limit:
        aggregate_failures.append("material_mean_full_repetition_increase")
    if tail_mean_delta > mean_repetition_delta_limit:
        aggregate_failures.append("material_mean_tail_repetition_increase")
    contract_drop = float(baseline_summary["contract_valid_rate"]) - float(
        candidate_summary["contract_valid_rate"]
    )
    if contract_drop > contract_rate_drop_limit:
        aggregate_failures.append("material_output_contract_rate_drop")
    if float(candidate_summary["finite_rate"]) != 1.0:
        aggregate_failures.append("candidate_non_finite_rate")
    if float(candidate_summary["cap_rate"]) != 0.0:
        aggregate_failures.append("candidate_completion_cap_rate")

    passed = not failures and not aggregate_failures
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "sample_manifest_sha256": next(iter(manifests)),
        "thresholds": {
            "catastrophic_full_4gram_repetition": 0.50,
            "catastrophic_tail_4gram_repetition": 0.60,
            "per_case_repetition_delta_limit": per_case_repetition_delta_limit,
            "mean_repetition_delta_limit": mean_repetition_delta_limit,
            "contract_rate_drop_limit": contract_rate_drop_limit,
            "completion_cap_rate": 0.0,
            "finite_rate": 1.0,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "aggregate_deltas": {
            "mean_full_repetition": round(full_mean_delta, 8),
            "mean_tail_repetition": round(tail_mean_delta, 8),
            "contract_valid_rate_drop": round(contract_drop, 8),
        },
        "case_failures": failures,
        "aggregate_failures": aggregate_failures,
        "case_deltas": deltas,
    }


def _load_result_file(path: Path) -> list[Mapping[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ProbeError(f"probe result file is empty: {path}")
    return rows


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ProbeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create the fixed sample manifest")
    prepare.add_argument("--candidate-dir", required=True, type=Path)
    prepare.add_argument("--candidate-manifest-sha256", required=True)
    prepare.add_argument("--tokenizer", required=True, type=Path)
    prepare.add_argument("--training-config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--sample-seed", type=int, default=20260809)
    prepare.add_argument("--greedy-longest", type=int, default=2)

    run = subparsers.add_parser("run", help="generate one baseline or adapter probe")
    run.add_argument("--base-model", required=True, type=Path)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--sample-manifest", required=True, type=Path)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--model-label", required=True)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--tail-tokens", type=int, default=DEFAULT_TAIL_TOKENS)
    run.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )

    compare = subparsers.add_parser("compare", help="compare baseline and adapter probes")
    compare.add_argument("--baseline-results", required=True, type=Path)
    compare.add_argument("--candidate-results", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--per-case-repetition-delta-limit", type=float, default=0.20)
    compare.add_argument("--mean-repetition-delta-limit", type=float, default=0.10)
    compare.add_argument("--contract-rate-drop-limit", type=float, default=0.25)
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
            artifact = build_sample_manifest(
                candidate_dir=args.candidate_dir,
                candidate_manifest_sha256=args.candidate_manifest_sha256,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                training_config=args.training_config,
                sample_seed=args.sample_seed,
                greedy_longest=args.greedy_longest,
            )
            _write_new_json(args.output, artifact)
            print(
                canonical_json(
                    {
                        "status": "prepared",
                        "path": str(args.output.resolve()),
                        "sha256": sha256_file(args.output),
                        "samples": len(artifact["samples"]),
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
                load_in_4bit=args.load_in_4bit,
                attn_implementation=args.attn_implementation,
            )
            print(canonical_json(summary))
            return 0 if summary["status"] == "complete" else 2
        comparison = compare_probe_results(
            _load_result_file(args.baseline_results),
            _load_result_file(args.candidate_results),
            per_case_repetition_delta_limit=args.per_case_repetition_delta_limit,
            mean_repetition_delta_limit=args.mean_repetition_delta_limit,
            contract_rate_drop_limit=args.contract_rate_drop_limit,
        )
        comparison = {
            **comparison,
            "inputs": {
                "baseline_results": {
                    "path": str(args.baseline_results.expanduser().resolve()),
                    "sha256": sha256_file(args.baseline_results),
                },
                "candidate_results": {
                    "path": str(args.candidate_results.expanduser().resolve()),
                    "sha256": sha256_file(args.candidate_results),
                },
            },
        }
        _write_new_json(args.output, comparison)
        print(canonical_json(comparison))
        return 0 if comparison["status"] == "passed" else 2
    except (ProbeError, OSError, ValueError) as exc:
        print(
            canonical_json(
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
