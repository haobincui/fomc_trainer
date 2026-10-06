"""Validation-only hard-gate probe for paper Model chk-2 checkpoints.

The probe is deliberately separate from training and from the immutable data
release.  It authenticates the paper-chk2 release, selects a deterministic
validation panel, generates greedily from one LoRA checkpoint, and records
structural, deterministic-fidelity, and repetition gates.  It never generates
from the sealed test split and it does not treat these deterministic checks as
a substitute for fresh semantic Validator A/B calls.

Generated text is retained in a private (mode 0600) result file so a later,
fresh Validator-A -> Validator-B pass can judge the exact student output.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jobs.generation.generate_chk3_sft_targets import (
    _attribution_categories,
    _date_values,
    _numeric_values,
)
from jobs.generation.generate_paper_chk2_chk1_analysis_rewrite import (
    _CITATION_RE,
    _FINAL_BAD_RE,
    _SECRET_RE,
    _WORD_RE,
)
from jobs.retrain_v2 import probe_chk1_sft_degeneration as common_probe
from open_r1.provenance import sha256_file
from open_r1.trainer.dataset_release import verify_paper_chk2_sft_release


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_"
    "v6_downstream128_recovery_v1_20260831"
)
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
RELEASE_MANIFEST_FILE_SHA256 = (
    "cb022dcd379e069e3d5ad54d7c7fdbc9e2ee29f958c85d5ff45d066f7728c097"
)
TRAINING_CONFIG = REPO_ROOT / (
    "configs/retrain_v2/"
    "paper_chk2_minutes_sft_chk1_cp200_v6_recovery_full3ep_lr1e6_20260901.yaml"
)
PARENT_MODEL = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
ADAPTER_ROOT = REPO_ROOT / (
    "output/training/retrain_v2/"
    "paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901/"
    "adapters/chk2"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_minutes_checkpoint_hard_gate_probe_v1_20260901"
)

SCHEMA = "paper-chk2-minutes-checkpoint-probe-v1"
MANIFEST_SCHEMA = "paper-chk2-minutes-checkpoint-probe-manifest-v1"
RESULT_SCHEMA = "paper-chk2-minutes-checkpoint-probe-result-v1"
SUMMARY_SCHEMA = "paper-chk2-minutes-checkpoint-probe-summary-v1"
BOUNDARY = "</think>"
CHECKPOINTS = (40, 50, 60)
DEFAULT_SAMPLE_COUNT = 8
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TAIL_TOKENS = 1024
DEFAULT_SEED = 20260901
MAX_SEQUENCE_TOKENS = 4096
MIN_GPU_FREE_GIB = 12
SELECTION_SALT = "paper-chk2-minutes-hard-gate-validation-v1"


class PaperChk2ProbeError(RuntimeError):
    """A paper-chk2 probe input or output failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2ProbeError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperChk2ProbeError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        _require(bool(raw), f"blank row in {label}:{line_number}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PaperChk2ProbeError(
                f"invalid JSON in {label}:{line_number}"
            ) from exc
        _require(isinstance(value, dict), f"non-object row in {label}:{line_number}")
        rows.append(value)
    return rows


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _load_training_config() -> dict[str, Any]:
    _require(
        TRAINING_CONFIG.is_file() and not TRAINING_CONFIG.is_symlink(),
        f"missing training config: {TRAINING_CONFIG}",
    )
    value = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "training config must be a mapping")
    return value


def extract_source_analysis(prompt: str) -> str:
    prefix = "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
    _require(isinstance(prompt, str) and prompt.startswith(prefix), "prompt prefix drift")
    try:
        payload = json.loads(prompt[len(prefix) :])
    except json.JSONDecodeError as exc:
        raise PaperChk2ProbeError("prompt payload is not valid JSON") from exc
    _require(
        isinstance(payload, dict) and set(payload) == {"analysis"},
        "prompt payload must contain only analysis",
    )
    analysis = payload.get("analysis")
    _require(isinstance(analysis, str) and bool(analysis.strip()), "analysis is empty")
    return analysis.strip()


def _prompt_ids(tokenizer: Any, *, system_prompt: str, prompt: str) -> list[int]:
    try:
        values = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise PaperChk2ProbeError(f"chat-template rendering failed: {exc}") from exc
    if hasattr(values, "tolist"):
        values = values.tolist()
    _require(isinstance(values, list) and values, "chat template returned no IDs")
    _require(
        not values or not isinstance(values[0], list),
        "chat template unexpectedly returned batched IDs",
    )
    return [int(value) for value in values]


def _counter_payload(values: Counter[str]) -> list[list[Any]]:
    return [[key, values[key]] for key in sorted(values)]


def _feature_record(source: str) -> dict[str, Any]:
    numbers = _numeric_values(source)
    dates = _date_values(source)
    attributions = _attribution_categories(source)
    return {
        "numeric_occurrences": sum(numbers.values()),
        "date_values": len(dates),
        "attribution_categories": sorted(attributions),
        "numeric_multiset_sha256": _sha256_text(
            _canonical_json(_counter_payload(numbers))
        ),
        "date_set_sha256": _sha256_text(_canonical_json(sorted(dates))),
        "attribution_set_sha256": _sha256_text(
            _canonical_json(sorted(attributions))
        ),
    }


def _ranked_pick(
    ordered: Sequence[Mapping[str, Any]],
    *,
    selected: list[dict[str, Any]],
    tag: str,
) -> None:
    used = {str(item["sample_id"]) for item in selected}
    used_meetings = {str(item["meeting_date"]) for item in selected}
    alternatives = [item for item in ordered if str(item["sample_id"]) not in used]
    if not alternatives:
        return
    distinct = [item for item in alternatives if str(item["meeting_date"]) not in used_meetings]
    chosen = distinct[0] if distinct else alternatives[0]
    selected.append({**dict(chosen), "selection_tag": tag})


def select_probe_rows(
    candidates: Sequence[Mapping[str, Any]], sample_count: int
) -> list[dict[str, Any]]:
    """Select a deterministic, diverse validation panel without target access."""

    _require(1 <= sample_count <= len(candidates), "invalid sample_count")
    if sample_count == len(candidates):
        return [
            {**dict(row), "selection_tag": "full_validation"}
            for row in sorted(candidates, key=lambda item: int(item["line_number"]))
        ]
    selected: list[dict[str, Any]] = []
    length_order = sorted(
        candidates,
        key=lambda row: (int(row["prompt_token_count"]), str(row["sample_id"])),
    )
    quantiles = (
        ("length_min", 0.0),
        ("length_q25", 0.25),
        ("length_median", 0.5),
        ("length_q75", 0.75),
        ("length_max", 1.0),
    )
    for tag, quantile in quantiles:
        target = round((len(length_order) - 1) * quantile)
        ranked = sorted(
            length_order,
            key=lambda row: (
                abs(length_order.index(row) - target),
                _sha256_text(f"{SELECTION_SALT}:{row['sample_id']}"),
            ),
        )
        _ranked_pick(ranked, selected=selected, tag=tag)
        if len(selected) >= sample_count:
            return selected
    numeric_order = sorted(
        candidates,
        key=lambda row: (
            -int(row["source_features"]["numeric_occurrences"]),
            str(row["sample_id"]),
        ),
    )
    _ranked_pick(numeric_order, selected=selected, tag="numeric_heavy")
    if len(selected) >= sample_count:
        return selected
    date_order = sorted(
        candidates,
        key=lambda row: (
            -int(row["source_features"]["date_values"]),
            str(row["sample_id"]),
        ),
    )
    _ranked_pick(date_order, selected=selected, tag="date_heavy")
    if len(selected) >= sample_count:
        return selected
    attribution_order = sorted(
        candidates,
        key=lambda row: (
            -len(row["source_features"]["attribution_categories"]),
            str(row["sample_id"]),
        ),
    )
    _ranked_pick(attribution_order, selected=selected, tag="attribution")
    while len(selected) < sample_count:
        used_styles = {str(item["section_style_id"]) for item in selected}
        used_topics = {str(item["atomic_topic"]) for item in selected}
        fill = sorted(
            candidates,
            key=lambda row: (
                str(row["section_style_id"]) in used_styles,
                str(row["atomic_topic"]) in used_topics,
                _sha256_text(f"{SELECTION_SALT}:fill:{row['sample_id']}"),
            ),
        )
        before = len(selected)
        _ranked_pick(fill, selected=selected, tag="diversity_fill")
        _require(len(selected) > before, "cannot complete diverse panel")
    _require(len({row["sample_id"] for row in selected}) == sample_count, "duplicate panel ID")
    return selected


def prepare(output_root: Path, *, sample_count: int) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "probe_manifest.json"
    _require(not output_root.exists() and not output_root.is_symlink(), f"output exists: {output_root}")
    config = _load_training_config()
    system_prompt = config.get("system_prompt")
    _require(isinstance(system_prompt, str) and bool(system_prompt), "system prompt missing")
    binding = verify_paper_chk2_sft_release(
        dataset_dir=RELEASE_ROOT / "minutes_alignment",
        manifest_path=RELEASE_MANIFEST,
        expected_manifest_sha256=RELEASE_MANIFEST_FILE_SHA256,
        training_scope=str(config.get("dataset_paper_chk2_scope")),
        system_prompt=system_prompt,
        model_path=PARENT_MODEL,
    )
    validation_data = RELEASE_ROOT / "minutes_alignment/validation.jsonl"
    validation_sidecar = RELEASE_ROOT / "minutes_alignment/manifests/validation.jsonl"
    rows = _read_jsonl(validation_data, label="validation data")
    sidecars = _read_jsonl(validation_sidecar, label="validation sidecar")
    _require(len(rows) == len(sidecars) == 42, "validation population drift")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(PARENT_MODEL),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=False,
    )
    candidates: list[dict[str, Any]] = []
    for index, (row, sidecar) in enumerate(zip(rows, sidecars, strict=True), start=1):
        _require(set(row) == {"prompt", "response"}, f"validation schema drift:{index}")
        prompt = row["prompt"]
        _require(
            sidecar.get("prompt_sha256") == _sha256_text(prompt),
            f"prompt hash drift:{index}",
        )
        source = extract_source_analysis(prompt)
        prompt_ids = _prompt_ids(tokenizer, system_prompt=system_prompt, prompt=prompt)
        candidates.append(
            {
                "sample_id": sidecar["sample_id"],
                "line_number": index,
                "meeting_date": sidecar["meeting_date"],
                "atomic_topic": sidecar["atomic_topic"],
                "section_style_id": sidecar["section_style_id"],
                "prompt_sha256": sidecar["prompt_sha256"],
                "source_analysis_sha256": sidecar["source_analysis_sha256"],
                "prompt_token_count": len(prompt_ids),
                "source_features": _feature_record(source),
            }
        )
    selected = select_probe_rows(candidates, sample_count)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "selection_only": True,
        "evaluation_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
        "test_generation_performed": False,
        "sample_selection": {
            "population": "validation",
            "population_rows": len(rows),
            "selected_rows": sample_count,
            "strategy": "deterministic_length_fact_density_attribution_style_topic_meeting_diversity_v1",
            "salt_sha256": _sha256_text(SELECTION_SALT),
        },
        "release": {
            "root": str(RELEASE_ROOT),
            "manifest": str(RELEASE_MANIFEST),
            "manifest_file_sha256": RELEASE_MANIFEST_FILE_SHA256,
            "validation_data": {
                "path": str(validation_data),
                "sha256": sha256_file(validation_data),
                "rows": len(rows),
            },
            "validation_sidecar": {
                "path": str(validation_sidecar),
                "sha256": sha256_file(validation_sidecar),
                "rows": len(sidecars),
            },
            "binding_scope": binding["scope"],
        },
        "training_config": {
            "path": str(TRAINING_CONFIG),
            "sha256": sha256_file(TRAINING_CONFIG),
            "system_prompt_sha256": _sha256_text(system_prompt),
        },
        "parent_model": {"path": str(PARENT_MODEL)},
        "checkpoints": {
            str(step): {
                "path": str(ADAPTER_ROOT / f"checkpoint-{step}"),
                "adapter_config_sha256": sha256_file(
                    ADAPTER_ROOT / f"checkpoint-{step}/adapter_config.json"
                ),
                "adapter_model_sha256": sha256_file(
                    ADAPTER_ROOT / f"checkpoint-{step}/adapter_model.safetensors"
                ),
            }
            for step in CHECKPOINTS
        },
        "samples": selected,
    }
    manifest["manifest_sha256"] = _sha256_text(_canonical_json(manifest))
    _write_exclusive_json(manifest_path, manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path, label="probe manifest")
    stored = manifest.pop("manifest_sha256", None)
    _require(
        manifest.get("schema_version") == MANIFEST_SCHEMA
        and stored == _sha256_text(_canonical_json(manifest)),
        "probe manifest self-hash drift",
    )
    manifest["manifest_sha256"] = stored
    release = manifest.get("release")
    _require(isinstance(release, dict), "probe release binding missing")
    for name in ("validation_data", "validation_sidecar"):
        record = release.get(name)
        _require(isinstance(record, dict), f"probe {name} binding missing")
        path_value = Path(str(record.get("path")))
        _require(sha256_file(path_value) == record.get("sha256"), f"{name} hash drift")
    return manifest


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _sentence_list(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", value.strip())
        if item.strip()
    ]


def _word_ngram_repetition(value: str, n: int) -> float:
    words = [item.casefold() for item in _WORD_RE.findall(value)]
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def analyze_student_completion(
    *,
    completion: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int] | set[int] | None,
    max_new_tokens: int,
    prompt_token_count: int,
    source_analysis: str,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> dict[str, Any]:
    """Apply deterministic hard gates without claiming semantic entailment."""

    base = common_probe.analyze_completion(
        text=completion,
        generated_token_ids=generated_token_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=max_new_tokens,
        tail_tokens=tail_tokens,
    )
    boundary_count = int(base["think_boundary_count"])
    reasoning = completion.split(BOUNDARY, 1)[0].strip() if boundary_count == 1 else ""
    answer = completion.split(BOUNDARY, 1)[1].strip() if boundary_count == 1 else ""
    answer_words = _WORD_RE.findall(answer)
    source_numbers = _numeric_values(source_analysis)
    answer_numbers = _numeric_values(answer)
    source_dates = _date_values(source_analysis)
    answer_dates = _date_values(answer)
    source_attributions = _attribution_categories(source_analysis)
    answer_attributions = _attribution_categories(answer)
    sentences = _sentence_list(answer)
    normalized_sentences = [_normalize_text(item) for item in sentences]
    duplicate_sentences = sorted(
        item for item, count in Counter(normalized_sentences).items() if item and count > 1
    )
    failures = list(base.get("catastrophic_reasons") or [])
    if not bool(base.get("hit_eos")):
        failures.append("missing_terminal_eos")
    if boundary_count != 1:
        failures.append("think_boundary_count_not_one")
    if not reasoning:
        failures.append("empty_native_reasoning")
    if not answer:
        failures.append("empty_final_answer")
    if answer and re.search(r"[\r\n\u2028\u2029]", answer):
        failures.append("final_answer_not_single_paragraph")
    if answer and not 20 <= len(answer_words) <= 400:
        failures.append("final_answer_word_count_outside_20_400")
    if answer and _FINAL_BAD_RE.search(answer):
        failures.append("final_answer_heading_list_json_or_meta")
    if answer and _CITATION_RE.search(answer):
        failures.append("final_answer_contains_citation")
    if answer and _SECRET_RE.search(answer):
        failures.append("final_answer_contains_credential_pattern")
    if answer and any(token in answer for token in ("<think>", "</think>", "<answer>", "</answer>")):
        failures.append("final_answer_contains_control_marker")
    if answer and _normalize_text(answer) == _normalize_text(source_analysis):
        failures.append("final_answer_exact_full_source_copy")
    if source_numbers != answer_numbers:
        failures.append("numeric_multiset_not_preserved")
    if source_dates != answer_dates:
        failures.append("date_set_not_preserved")
    if source_attributions != answer_attributions:
        failures.append("attribution_set_not_preserved")
    if duplicate_sentences:
        failures.append("duplicate_final_sentence")
    if prompt_token_count + int(base["raw_generated_tokens"]) > MAX_SEQUENCE_TOKENS:
        failures.append("sequence_exceeds_4096_tokens")
    failures = sorted(set(failures))
    return {
        **dict(base),
        "finish_reason": "eos" if base["hit_eos"] else "length" if base["cap_reached"] else "other",
        "prompt_tokens": prompt_token_count,
        "total_tokens": prompt_token_count + int(base["raw_generated_tokens"]),
        "native_reasoning_nonempty": bool(reasoning),
        "final_answer_nonempty": bool(answer),
        "final_answer_single_paragraph": bool(answer) and not bool(re.search(r"[\r\n\u2028\u2029]", answer)),
        "final_answer_word_count": len(answer_words),
        "numeric_multiset_preserved": source_numbers == answer_numbers,
        "date_set_preserved": source_dates == answer_dates,
        "attribution_set_preserved": source_attributions == answer_attributions,
        "answer_word_3gram_repetition": round(_word_ngram_repetition(answer, 3), 8),
        "answer_word_4gram_repetition": round(_word_ngram_repetition(answer, 4), 8),
        "duplicate_final_sentence_count": len(duplicate_sentences),
        "deterministic_hard_gate_pass": not failures,
        "deterministic_hard_gate_failures": failures,
        "semantic_factual_fidelity_status": "PENDING_FRESH_VALIDATOR_A",
        "minutes_style_status": "PENDING_FRESH_VALIDATOR_B",
    }


def _adapter_record(manifest: Mapping[str, Any], checkpoint: int) -> Mapping[str, Any]:
    checkpoints = manifest.get("checkpoints")
    _require(isinstance(checkpoints, dict), "checkpoint bindings missing")
    value = checkpoints.get(str(checkpoint))
    _require(isinstance(value, dict), f"checkpoint-{checkpoint} binding missing")
    path = Path(str(value.get("path")))
    _require(
        sha256_file(path / "adapter_config.json") == value.get("adapter_config_sha256"),
        f"checkpoint-{checkpoint} adapter config drift",
    )
    _require(
        sha256_file(path / "adapter_model.safetensors") == value.get("adapter_model_sha256"),
        f"checkpoint-{checkpoint} adapter weights drift",
    )
    return value


def generate(
    manifest_path: Path,
    output_root: Path,
    *,
    checkpoint: int,
    max_new_tokens: int,
    tail_tokens: int,
    seed: int,
) -> dict[str, Any]:
    _require(checkpoint in CHECKPOINTS, f"unsupported checkpoint: {checkpoint}")
    _require(512 <= max_new_tokens <= 3072, "max_new_tokens must be in 512..3072")
    _require(tail_tokens > 0, "tail_tokens must be positive")
    manifest = _load_manifest(manifest_path.expanduser().resolve())
    adapter_record = _adapter_record(manifest, checkpoint)
    target = output_root.expanduser().resolve() / "generations" / f"checkpoint-{checkpoint}"
    _require(not target.exists() and not target.is_symlink(), f"generation exists: {target}")
    staging = target.parent / f".{target.name}.staging-{os.getpid()}"
    _require(not staging.exists() and not staging.is_symlink(), f"staging exists: {staging}")
    staging.mkdir(parents=True, mode=0o700)

    validation_data = Path(str(manifest["release"]["validation_data"]["path"]))
    validation_sidecar = Path(str(manifest["release"]["validation_sidecar"]["path"]))
    rows = _read_jsonl(validation_data, label="bound validation data")
    sidecars = _read_jsonl(validation_sidecar, label="bound validation sidecar")
    samples = manifest.get("samples")
    _require(isinstance(samples, list) and samples, "probe samples missing")
    config = _load_training_config()
    system_prompt = str(config["system_prompt"])
    launch = {
        "schema_version": SCHEMA,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "checkpoint": checkpoint,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "adapter": dict(adapter_record),
        "generation": {
            "mode": "greedy",
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "tail_tokens": tail_tokens,
            "seed": seed,
            "load_in_4bit": True,
            "attn_implementation": "sdpa",
        },
        "scope": {
            "split": "validation",
            "test_generation_performed": False,
            "selection_only": True,
            "evaluation_eligible": False,
            "suitable_for_leakage_safe_evaluation": False,
        },
    }
    _write_exclusive_json(staging / "launch.json", launch)

    model = None
    results: list[dict[str, Any]] = []
    try:
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        _require(torch.cuda.is_available(), "CUDA is required")
        _require(torch.cuda.device_count() == 1, "expose exactly one GPU")
        free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
        _require(
            free_bytes >= MIN_GPU_FREE_GIB * 1024**3,
            f"GPU free memory below {MIN_GPU_FREE_GIB} GiB",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(PARENT_MODEL),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
            fix_mistral_regex=False,
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
            str(PARENT_MODEL),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=quantization,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            str(Path(str(adapter_record["path"]))),
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()
        model.config.use_cache = True
        eos_value = model.generation_config.eos_token_id
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_ids = common_probe._normalize_eos_ids(eos_value)
        _require(bool(eos_ids), "model/tokenizer has no EOS")
        torch.cuda.reset_peak_memory_stats()
        result_path = staging / "results.jsonl"
        descriptor = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for index, sample in enumerate(samples):
                line_index = int(sample["line_number"]) - 1
                row = rows[line_index]
                sidecar = sidecars[line_index]
                _require(sidecar["sample_id"] == sample["sample_id"], "sample order drift")
                _require(_sha256_text(row["prompt"]) == sample["prompt_sha256"], "sample prompt drift")
                source = extract_source_analysis(row["prompt"])
                prompt_ids = _prompt_ids(tokenizer, system_prompt=system_prompt, prompt=row["prompt"])
                _require(len(prompt_ids) == sample["prompt_token_count"], "prompt token drift")
                _require(
                    len(prompt_ids) + max_new_tokens <= MAX_SEQUENCE_TOKENS,
                    "probe token budget exceeds training max_length",
                )
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
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
                completion = common_probe.decode_completion_preserving_boundary(
                    tokenizer, generated_ids, eos_ids
                )
                metrics = analyze_student_completion(
                    completion=completion,
                    generated_token_ids=generated_ids,
                    eos_token_ids=eos_ids,
                    max_new_tokens=max_new_tokens,
                    prompt_token_count=len(prompt_ids),
                    source_analysis=source,
                    tail_tokens=tail_tokens,
                )
                reasoning = completion.split(BOUNDARY, 1)[0].strip() if completion.count(BOUNDARY) == 1 else ""
                answer = completion.split(BOUNDARY, 1)[1].strip() if completion.count(BOUNDARY) == 1 else ""
                result = {
                    "schema_version": RESULT_SCHEMA,
                    "checkpoint": checkpoint,
                    "sample_id": sample["sample_id"],
                    "meeting_date": sample["meeting_date"],
                    "atomic_topic": sample["atomic_topic"],
                    "section_style_id": sample["section_style_id"],
                    "selection_tag": sample["selection_tag"],
                    "seed": case_seed,
                    "prompt_sha256": sample["prompt_sha256"],
                    "source_analysis_sha256": sample["source_analysis_sha256"],
                    "completion_sha256": _sha256_text(completion),
                    "reasoning_sha256": _sha256_text(reasoning),
                    "rewritten_minutes_sha256": _sha256_text(answer),
                    "source_analysis": source,
                    "student_native_reasoning": reasoning,
                    "rewritten_minutes": answer,
                    "completion": completion,
                    "metrics": metrics,
                }
                handle.write(_canonical_json(result) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                results.append(result)
                print(
                    f"checkpoint={checkpoint} case={index + 1}/{len(samples)} "
                    f"sample={sample['sample_id']} pass={metrics['deterministic_hard_gate_pass']} "
                    f"tokens={metrics['raw_generated_tokens']}",
                    flush=True,
                )
                del input_ids, attention_mask, sequences
        total = len(results)
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "checkpoint": checkpoint,
            "cases": total,
            "deterministic_pass_count": sum(
                bool(row["metrics"]["deterministic_hard_gate_pass"]) for row in results
            ),
            "deterministic_pass_rate": sum(
                bool(row["metrics"]["deterministic_hard_gate_pass"]) for row in results
            ) / total,
            "failure_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in results
                        for reason in row["metrics"]["deterministic_hard_gate_failures"]
                    ).items()
                )
            ),
            "mean_generated_tokens": statistics.fmean(
                int(row["metrics"]["raw_generated_tokens"]) for row in results
            ),
            "max_full_token_4gram_repetition": max(
                float(row["metrics"]["full_token_4gram_repetition"]) for row in results
            ),
            "max_tail_token_4gram_repetition": max(
                float(row["metrics"]["tail_token_4gram_repetition"]) for row in results
            ),
            "fresh_validator_a_status": "PENDING",
            "fresh_validator_b_status": "PENDING",
            "checkpoint_hard_gate_status": "INCOMPLETE_PENDING_FRESH_LLM_VALIDATORS",
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "results": {
                "path": "results.jsonl",
                "sha256": sha256_file(result_path),
                "rows": total,
            },
        }
        _write_exclusive_json(staging / "summary.json", summary)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return summary
    except BaseException as exc:
        try:
            _write_exclusive_json(
                staging / "failure.json",
                {
                    **launch,
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_cases": len(results),
                },
            )
        finally:
            pass
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def status(output_root: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "probe_manifest.json"
    manifest = _load_manifest(manifest_path)
    checkpoints: dict[str, Any] = {}
    for step in CHECKPOINTS:
        summary_path = output_root / "generations" / f"checkpoint-{step}/summary.json"
        checkpoints[str(step)] = (
            _read_json(summary_path, label=f"checkpoint-{step} summary")
            if summary_path.is_file()
            else {"status": "not_run"}
        )
    return {
        "schema_version": SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_rows": len(manifest["samples"]),
        "checkpoints": checkpoints,
        "test_generation_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--manifest", type=Path, required=True)
    generate_parser.add_argument("--output-root", type=Path, required=True)
    generate_parser.add_argument("--checkpoint", type=int, choices=CHECKPOINTS, required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    generate_parser.add_argument("--tail-tokens", type=int, default=DEFAULT_TAIL_TOKENS)
    generate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_root, sample_count=args.sample_count)
    elif args.command == "generate":
        result = generate(
            args.manifest,
            args.output_root,
            checkpoint=args.checkpoint,
            max_new_tokens=args.max_new_tokens,
            tail_tokens=args.tail_tokens,
            seed=args.seed,
        )
    else:
        result = status(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
