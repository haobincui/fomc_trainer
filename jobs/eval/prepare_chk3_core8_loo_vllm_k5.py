"""Seal the exact-token CHK3 Core8 LOO K=5 vLLM input cohort.

This module prepares a new stochastic K=5 cohort from the already sealed
N=2,176 Core8 leave-one-out matrix.  The historical greedy generation
metadata in that source manifest is provenance only: no greedy output is
reused as one of the five stochastic replicates.

The complete FOMC chat-template token IDs are materialized once here.  The
vLLM workers must consume those IDs directly and must not render the chat
template again at generation time.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_core8_loo_stochastic_k10 as source_contract
from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-cp318-core8-loo-n128-vllm-k5-dual-dp1-v1"
COHORT_SCHEMA = "chk3-core8-loo-vllm-k5-cohort-v2"
LEDGER_ROW_SCHEMA = "chk3-core8-loo-vllm-prompt-token-ledger-row-v1"
PREPARATION_SCHEMA = "chk3-core8-loo-vllm-k5-preparation-v2"
EXECUTION_POLICY_SCHEMA = "chk3-core8-loo-vllm-k5-execution-policy-v2"

DEFAULT_SOURCE_SAMPLE_MANIFEST = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/"
    "inputs_v1/samples.json"
)
SOURCE_SAMPLE_MANIFEST_SHA256 = (
    "5ddaf97ef9ee7d957ca65a977d6527ea5cedc93e2b22fc15bc08627ea9e91efe"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1/preparation_v2"
)
HISTORICAL_GREEDY_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1"
)
HISTORICAL_GREEDY_GENERATION_MANIFEST = (
    HISTORICAL_GREEDY_ROOT / "generation_v1/manifest.json"
)
HISTORICAL_GREEDY_GENERATION_MANIFEST_SHA256 = (
    "b610716e5f35a4ea8012e9a4b5c2a41b0d557e2a08bac79d709839f87ec969b6"
)
HISTORICAL_GREEDY_GENERATIONS = (
    HISTORICAL_GREEDY_ROOT / "generation_v1/generations.jsonl"
)
HISTORICAL_GREEDY_GENERATIONS_SHA256 = (
    "9901a90210d81ae7f97a08965e201d267b9360dcc449d8813d2d0348550608af"
)
HISTORICAL_GREEDY_SCORE_MANIFEST = HISTORICAL_GREEDY_ROOT / "score_v1/manifest.json"
HISTORICAL_GREEDY_SCORE_MANIFEST_SHA256 = (
    "4226551fd40ea9b02678624756295d868929fecec8664352ff5282134dde8df8"
)

MODEL_ID = "chk3"
EXPECTED_PROMPTS = 2_176
EXPECTED_MEETINGS = 128
VARIANTS_PER_MEETING = 17
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
)
EXPECTED_CASES = EXPECTED_PROMPTS * len(REPLICATE_SEEDS)
MAX_NEW_TOKENS = 2_560
MAX_MODEL_LEN = 4_096
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS
# Two complete 85-case, sample-major meeting blocks per global window.
ABSOLUTE_CHUNK_SIZE = VARIANTS_PER_MEETING * len(REPLICATE_SEEDS) * 2
EXPECTED_CHUNKS = EXPECTED_CASES // ABSOLUTE_CHUNK_SIZE
MAX_NUM_SEQS_PER_WORKER = 16
GPU_MEMORY_UTILIZATION_PER_WORKER = 0.95
EXPECTED_VLLM_VERSION = "0.8.5.post1"
MAX_NUM_BATCHED_TOKENS_PER_WORKER = 4_096
EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f"
)

LEDGER_FILENAME = "prompt_token_ledger_n2176.v2.jsonl"
COHORT_FILENAME = "cohort_n2176_k5.v2.json"
PREPARATION_FILENAME = "preparation.v2.json"
EXECUTION_POLICY_FILENAME = "execution_policy.v2.json"


class LooVllmK5PreparationError(RuntimeError):
    """The sealed LOO exact-token preparation contract was violated."""


def canonical_case_coordinates(absolute_case_index: int) -> dict[str, int]:
    """Map a sample-major absolute case index to its paired GPU shard."""

    if (
        not isinstance(absolute_case_index, int)
        or isinstance(absolute_case_index, bool)
        or absolute_case_index < 0
        or absolute_case_index >= EXPECTED_CASES
    ):
        raise LooVllmK5PreparationError("absolute case index is out of range")
    replicate_count = len(REPLICATE_SEEDS)
    absolute_prompt_index = absolute_case_index // replicate_count
    replicate_id = absolute_case_index % replicate_count
    meeting_index = absolute_prompt_index // VARIANTS_PER_MEETING
    variant_rank = absolute_prompt_index % VARIANTS_PER_MEETING
    paired_block_index = meeting_index * replicate_count + replicate_id
    return {
        "absolute_case_index": absolute_case_index,
        "absolute_prompt_index": absolute_prompt_index,
        "replicate_id": replicate_id,
        "meeting_index": meeting_index,
        "variant_rank": variant_rank,
        "paired_block_index": paired_block_index,
        "shard_id": paired_block_index % 2,
    }


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise LooVllmK5PreparationError(
            f"value cannot be represented as canonical JSON: {exc}"
        ) from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_new_bytes(path: Path, body: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise LooVllmK5PreparationError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise LooVllmK5PreparationError(
                f"refusing to overwrite artifact: {path}"
            ) from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(path, _encoded_json(value))


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new_bytes(
        path,
        "".join(_canonical(dict(row)) + "\n" for row in rows).encode("utf-8"),
    )


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LooVllmK5PreparationError(
            f"cannot bind missing/non-regular artifact: {resolved}"
        )
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LooVllmK5PreparationError(f"missing regular JSON artifact: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LooVllmK5PreparationError(f"invalid JSON artifact {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise LooVllmK5PreparationError(f"JSON artifact is not an object: {resolved}")
    return value


def _token_id_sha256(token_ids: Sequence[int]) -> str:
    return _sha256_text(_canonical(list(token_ids)))


def load_excluded_historical_greedy_evidence() -> dict[str, Any]:
    """Deep-bind the completed K=1 greedy run that is excluded from K=5."""

    expected_files = (
        (
            HISTORICAL_GREEDY_GENERATION_MANIFEST,
            HISTORICAL_GREEDY_GENERATION_MANIFEST_SHA256,
        ),
        (HISTORICAL_GREEDY_GENERATIONS, HISTORICAL_GREEDY_GENERATIONS_SHA256),
        (
            HISTORICAL_GREEDY_SCORE_MANIFEST,
            HISTORICAL_GREEDY_SCORE_MANIFEST_SHA256,
        ),
    )
    for path, expected_sha256 in expected_files:
        observed = _sha256_file(path)
        if observed != expected_sha256:
            raise LooVllmK5PreparationError(
                f"excluded historical greedy artifact changed: {path}"
            )

    generation_manifest = _read_json(HISTORICAL_GREEDY_GENERATION_MANIFEST)
    score_manifest = _read_json(HISTORICAL_GREEDY_SCORE_MANIFEST)
    try:
        generation_payload_sha = validate_manifest_integrity(generation_manifest)
        score_payload_sha = validate_manifest_integrity(score_manifest)
    except Exception as exc:
        raise LooVllmK5PreparationError(
            f"excluded historical greedy manifest integrity failed: {exc}"
        ) from exc
    generation_contract = generation_manifest.get("generation_contract")
    generation_artifacts = generation_manifest.get("artifacts")
    score_run = score_manifest.get("run_manifest")
    if (
        generation_manifest.get("schema_version")
        != "chk3-core8-loo-full-generation-run-v1"
        or generation_manifest.get("status") != "complete"
        or generation_manifest.get("stage_id") != MODEL_ID
        or not isinstance(generation_contract, Mapping)
        or generation_contract.get("mode") != "greedy"
        or generation_contract.get("do_sample") is not False
        or not isinstance(generation_artifacts, Mapping)
        or not isinstance(generation_artifacts.get("generations"), Mapping)
        or generation_artifacts["generations"].get("sha256")
        != HISTORICAL_GREEDY_GENERATIONS_SHA256
        or generation_artifacts["generations"].get("rows") != EXPECTED_PROMPTS
        or score_manifest.get("schema_version")
        != "chk3-core8-loo-full-score-manifest-v1"
        or score_manifest.get("status") != "complete"
        or not isinstance(score_run, Mapping)
        or score_run.get("sha256")
        != HISTORICAL_GREEDY_GENERATION_MANIFEST_SHA256
    ):
        raise LooVllmK5PreparationError(
            "excluded historical greedy K=1 evidence contract drift"
        )
    for manifest in (generation_manifest, score_manifest):
        sample_binding = manifest.get("sample_manifest")
        if (
            not isinstance(sample_binding, Mapping)
            or sample_binding.get("sha256") != SOURCE_SAMPLE_MANIFEST_SHA256
        ):
            raise LooVllmK5PreparationError(
                "excluded greedy evidence does not use the sealed LOO cohort"
            )

    generation_binding = {
        **_binding(HISTORICAL_GREEDY_GENERATION_MANIFEST),
        "payload_sha256": generation_payload_sha,
    }
    generations_binding = _binding(
        HISTORICAL_GREEDY_GENERATIONS, rows=EXPECTED_PROMPTS
    )
    score_binding = {
        **_binding(HISTORICAL_GREEDY_SCORE_MANIFEST),
        "payload_sha256": score_payload_sha,
    }
    return {
        "role": "excluded_historical_greedy_k1_evidence_only",
        "included_in_stochastic_replicates": False,
        "exclusion_reason": (
            "greedy_do_sample_false_is_not_from_the_vllm_stochastic_distribution"
        ),
        "generation_manifest": generation_binding,
        "generations": generations_binding,
        "score_manifest": score_binding,
    }


def load_model_runtime_inventory(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind every CHK3 file that the BF16 vLLM runtime may consume."""

    try:
        anchor = source_contract._load_anchor()
    except source_contract.Core8StochasticK10Error as exc:
        raise LooVllmK5PreparationError(str(exc)) from exc
    model_path = Path(str(anchor["model_path"])).expanduser().resolve()
    tokenizer_binding = manifest.get("tokenizer")
    model_anchor_binding = manifest.get("model_anchor")
    if (
        not isinstance(tokenizer_binding, Mapping)
        or Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
        != model_path
        or not isinstance(model_anchor_binding, Mapping)
        or anchor["anchor"].get("sha256") != model_anchor_binding.get("sha256")
        or anchor["anchor"].get("payload_sha256")
        != model_anchor_binding.get("payload_sha256")
    ):
        raise LooVllmK5PreparationError("CHK3 runtime model anchor/path drift")

    model = anchor.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("files"), Mapping):
        raise LooVllmK5PreparationError("CHK3 anchor model inventory is missing")
    files: dict[str, str] = {
        str(name): str(digest) for name, digest in model["files"].items()
    }
    tokenizer_files = tokenizer_binding.get("files")
    if not isinstance(tokenizer_files, Mapping):
        raise LooVllmK5PreparationError("FOMC tokenizer inventory is missing")
    for name, digest in tokenizer_files.items():
        existing = files.get(str(name))
        if existing is not None and existing != digest:
            raise LooVllmK5PreparationError(
                f"runtime model/tokenizer hash conflict at {name}"
            )
        files[str(name)] = str(digest)

    chat_template = model_path / "chat_template.jinja"
    if (
        not chat_template.is_file()
        or chat_template.is_symlink()
        or _sha256_file(chat_template) != EXPECTED_CHAT_TEMPLATE_SHA256
    ):
        raise LooVllmK5PreparationError("runtime chat_template.jinja hash drift")
    files[chat_template.name] = EXPECTED_CHAT_TEMPLATE_SHA256

    for name, digest in files.items():
        candidate = model_path / name
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or _sha256_file(candidate) != digest
        ):
            raise LooVllmK5PreparationError(f"runtime model file changed: {name}")
    normalized = dict(sorted(files.items()))
    return {
        "path": str(model_path),
        "files": normalized,
        "file_count": len(normalized),
        "file_hash_inventory_sha256": _sha256_text(_canonical(normalized)),
        "chat_template_jinja_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
        "load_role": "exact_merged_chk3_cp318_bf16_vllm_runtime",
    }


def _validate_source_matrix(
    sample_manifest: Mapping[str, Any], bound_rows: Sequence[Mapping[str, Any]]
) -> None:
    samples = sample_manifest.get("samples")
    selection = sample_manifest.get("selection")
    if (
        not isinstance(samples, list)
        or len(samples) != EXPECTED_PROMPTS
        or len(bound_rows) != EXPECTED_PROMPTS
        or not isinstance(selection, Mapping)
        or selection.get("rows") != EXPECTED_PROMPTS
        or selection.get("meetings") != EXPECTED_MEETINGS
        or selection.get("variants_per_meeting") != VARIANTS_PER_MEETING
    ):
        raise LooVllmK5PreparationError(
            "sealed source matrix is not N2176 / 128 meetings / 17 variants"
        )
    sample_ids = [sample.get("sample_id") for sample in samples]
    row_ids = [row.get("sample_id") for row in bound_rows]
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise LooVllmK5PreparationError("source sample identity is invalid")
    if len(set(sample_ids)) != EXPECTED_PROMPTS or sample_ids != row_ids:
        raise LooVllmK5PreparationError(
            "sample/dataset identity, uniqueness, or order drift"
        )

    meeting_counts: Counter[str] = Counter()
    meeting_full_counts: Counter[str] = Counter()
    meeting_ranks: dict[str, set[int]] = {}
    for sample, row in zip(samples, bound_rows, strict=True):
        if not isinstance(sample, Mapping) or not isinstance(row, Mapping):
            raise LooVllmK5PreparationError("source matrix row is not an object")
        meeting_id = sample.get("meeting_id")
        if (
            not isinstance(meeting_id, str)
            or not meeting_id
            or row.get("meeting_id") != meeting_id
            or row.get("arm") != sample.get("arm")
            or row.get("intervention_topic") != sample.get("intervention_topic")
        ):
            raise LooVllmK5PreparationError("source LOO identity drift")
        for text_key, hash_key in (
            ("prompt", "prompt_sha256"),
            ("source_analysis", "source_analysis_sha256"),
            ("reference_minutes", "reference_minutes_sha256"),
        ):
            text = row.get(text_key)
            if (
                not isinstance(text, str)
                or not text
                or _sha256_text(text) != sample.get(hash_key)
            ):
                raise LooVllmK5PreparationError(
                    f"source text/hash drift at {text_key}"
                )
        prompt_count = sample.get("prompt_token_count")
        if (
            not isinstance(prompt_count, int)
            or isinstance(prompt_count, bool)
            or prompt_count <= 0
            or prompt_count + MAX_NEW_TOKENS > MAX_MODEL_LEN
        ):
            raise LooVllmK5PreparationError(
                f"prompt+generation exceeds {MAX_MODEL_LEN} for {sample['sample_id']}"
            )
        variant_rank = sample.get("variant_rank")
        if (
            not isinstance(variant_rank, int)
            or isinstance(variant_rank, bool)
            or variant_rank < 0
            or variant_rank >= VARIANTS_PER_MEETING
        ):
            raise LooVllmK5PreparationError("LOO variant rank drift")
        meeting_counts[meeting_id] += 1
        meeting_ranks.setdefault(meeting_id, set()).add(variant_rank)
        if sample.get("arm") == "full":
            meeting_full_counts[meeting_id] += 1

    if (
        len(meeting_counts) != EXPECTED_MEETINGS
        or set(meeting_counts.values()) != {VARIANTS_PER_MEETING}
        or set(meeting_full_counts.values()) != {1}
        or set(meeting_full_counts) != set(meeting_counts)
        or any(
            ranks != set(range(VARIANTS_PER_MEETING))
            for ranks in meeting_ranks.values()
        )
    ):
        raise LooVllmK5PreparationError("per-meeting 17-variant inventory drift")


def load_source_inputs(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], Any, str]:
    """Deep-validate the frozen LOO source and return its bound tokenizer."""

    if expected_sha256 != SOURCE_SAMPLE_MANIFEST_SHA256:
        raise LooVllmK5PreparationError(
            "source sample manifest must use the preregistered SHA-256"
        )
    try:
        # Do not ask the historical loader to render every prompt.  This
        # preparer materializes each prompt exactly once in build_token_ledger.
        manifest, rows, _unused_tokenizer, observed = source_contract._load_inputs(
            path, expected_sha256, load_tokenizer=False
        )
    except source_contract.Core8StochasticK10Error as exc:
        raise LooVllmK5PreparationError(str(exc)) from exc
    _validate_source_matrix(manifest, rows)
    tokenizer_binding = manifest.get("tokenizer")
    if not isinstance(tokenizer_binding, Mapping):
        raise LooVllmK5PreparationError("sample manifest tokenizer binding missing")
    tokenizer_path = Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
    try:
        observed_tokenizer = dict(native_probe._tokenizer_fingerprint(tokenizer_path))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
    except (native_probe.Chk3ProbeError, OSError, ValueError) as exc:
        raise LooVllmK5PreparationError(str(exc)) from exc
    if observed_tokenizer != dict(tokenizer_binding):
        raise LooVllmK5PreparationError("FOMC tokenizer file inventory changed")
    return manifest, rows, tokenizer, observed


def _load_prompt_contract(
    manifest: Mapping[str, Any], tokenizer: Any
) -> tuple[str, str | None, dict[str, Any]]:
    prompt_contract = manifest.get("prompt_contract")
    tokenizer_binding = manifest.get("tokenizer")
    if not isinstance(prompt_contract, Mapping) or not isinstance(
        tokenizer_binding, Mapping
    ):
        raise LooVllmK5PreparationError("source prompt/tokenizer binding is missing")
    config_path = Path(str(prompt_contract.get("training_config"))).expanduser().resolve()
    tokenizer_path = Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
    try:
        system_prompt, suffix, config_sha = native_probe._load_prompt_contract(
            config_path
        )
        tokenizer_fingerprint = dict(native_probe._tokenizer_fingerprint(tokenizer_path))
        tokenizer_semantics = native_eval._tokenizer_semantics(tokenizer)
    except (
        native_probe.Chk3ProbeError,
        native_eval.NativeThreeModelEvalError,
        OSError,
        ValueError,
    ) as exc:
        raise LooVllmK5PreparationError(str(exc)) from exc
    if (
        config_sha != prompt_contract.get("training_config_sha256")
        or _sha256_text(system_prompt) != prompt_contract.get("system_prompt_sha256")
        or (_sha256_text(suffix) if suffix else None)
        != prompt_contract.get("user_prompt_suffix_sha256")
    ):
        raise LooVllmK5PreparationError("training prompt contract changed")
    if tokenizer_fingerprint != dict(tokenizer_binding):
        raise LooVllmK5PreparationError("FOMC tokenizer file inventory changed")
    return system_prompt, suffix, tokenizer_semantics


def build_token_ledger(
    *,
    sample_manifest: Mapping[str, Any],
    bound_rows: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    user_prompt_suffix: str | None,
) -> list[dict[str, Any]]:
    """Materialize and validate every exact chat-template input token ID."""

    samples = sample_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_PROMPTS:
        raise LooVllmK5PreparationError("source sample inventory is not N=2,176")
    if len(bound_rows) != EXPECTED_PROMPTS or set(bound_rows) != {
        str(sample.get("sample_id")) for sample in samples
    }:
        raise LooVllmK5PreparationError("bound-row coverage differs from samples")

    ledger: list[dict[str, Any]] = []
    for absolute_prompt_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise LooVllmK5PreparationError("source sample is not an object")
        sample_id = str(sample.get("sample_id") or "")
        source = bound_rows.get(sample_id)
        if not sample_id or not isinstance(source, Mapping):
            raise LooVllmK5PreparationError(
                f"source row is missing for sample {sample_id!r}"
            )
        source_prompt = source.get("prompt")
        if (
            not isinstance(source_prompt, str)
            or _sha256_text(source_prompt) != sample.get("prompt_sha256")
        ):
            raise LooVllmK5PreparationError(f"source prompt hash drift for {sample_id}")
        try:
            messages = native_probe._messages(
                source,
                system_prompt=system_prompt,
                user_prompt_suffix=user_prompt_suffix,
            )
            prompt_token_ids = native_probe._prompt_ids(tokenizer, messages)
        except native_probe.Chk3ProbeError as exc:
            raise LooVllmK5PreparationError(str(exc)) from exc
        if (
            not prompt_token_ids
            or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in prompt_token_ids
            )
            or len(prompt_token_ids) != sample.get("prompt_token_count")
        ):
            raise LooVllmK5PreparationError(
                f"prompt-token contract drift for {sample_id}"
            )
        if len(prompt_token_ids) + MAX_NEW_TOKENS > MAX_MODEL_LEN:
            raise LooVllmK5PreparationError(
                f"prompt+generation exceeds {MAX_MODEL_LEN} for {sample_id}"
            )
        ledger.append(
            {
                "schema_version": LEDGER_ROW_SCHEMA,
                "absolute_prompt_index": absolute_prompt_index,
                "source_sample_line_number": absolute_prompt_index + 1,
                "sample_id": sample_id,
                "meeting_id": sample.get("meeting_id"),
                "arm": sample.get("arm"),
                "intervention_topic": sample.get("intervention_topic"),
                "variant_rank": sample.get("variant_rank"),
                "prompt_sha256": sample.get("prompt_sha256"),
                "messages_sha256": _sha256_text(_canonical(messages)),
                "prompt_token_count": len(prompt_token_ids),
                "prompt_token_ids": list(prompt_token_ids),
                "prompt_token_ids_sha256": _token_id_sha256(prompt_token_ids),
            }
        )
    return ledger


def _copy_binding(manifest: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise LooVllmK5PreparationError(f"source manifest has no {key} binding")
    return dict(value)


def build_execution_policy(
    *,
    source_sample_manifest: Mapping[str, Any],
    model_runtime_inventory: Mapping[str, Any],
    excluded_historical_greedy_k1: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the sealed user-approved dual-worker execution policy."""

    return seal_manifest(
        {
            "schema_version": EXECUTION_POLICY_SCHEMA,
            "status": "complete",
            "sealed": True,
            "immutable": True,
            "evaluation_id": EVALUATION_ID,
            "source_sample_manifest": dict(source_sample_manifest),
            "model_runtime_inventory": dict(model_runtime_inventory),
            "excluded_historical_greedy_k1": dict(
                excluded_historical_greedy_k1
            ),
            "topology": {
                "workers": 2,
                "worker_type": "two_independent_full_model_vllm_instances",
                "physical_gpu_indexes": [0, 1],
                "data_parallel_size_per_worker": 1,
                "tensor_parallel_size_per_worker": 1,
                "pipeline_parallel_size_per_worker": 1,
                "cross_worker_data_parallel_collective": False,
                "dtype": "bfloat16",
                "quantization": None,
            },
            "engine": {
                "backend": "vllm-v1-async-engine",
                "vllm_version": EXPECTED_VLLM_VERSION,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_seqs_per_worker": MAX_NUM_SEQS_PER_WORKER,
                "max_num_batched_tokens_per_worker": (
                    MAX_NUM_BATCHED_TOKENS_PER_WORKER
                ),
                "gpu_memory_utilization_per_worker": (
                    GPU_MEMORY_UTILIZATION_PER_WORKER
                ),
                "swap_space_gib": 0,
                "cpu_offload_gib": 0,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "enforce_eager": False,
                "cuda_graphs": True,
                "async_output_processing": True,
                "input_contract": (
                    "consume_exact_persisted_fomc_prompt_token_ids_"
                    "without_runtime_chat_template"
                ),
            },
            "sampling": {
                "generation_mode": "stochastic_sampling",
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "replicate_seeds": list(REPLICATE_SEEDS),
                "replicates": len(REPLICATE_SEEDS),
                "all_replicates_same_distribution": True,
                "all_replicates_fresh_vllm_generations": True,
                "historical_greedy_k1_excluded": True,
                "historical_nf4_k10_partial_excluded": True,
                "row_seed": "derive_row_seed(replicate_seed,meeting_full_sample_id)",
                "common_random_numbers": (
                    "same-meeting-replicate-across-all-17-arms-v1"
                ),
            },
            "scheduling_and_replay": {
                "canonical_case_order": (
                    "meeting_order_then_variant_rank_then_replicate_id"
                ),
                "absolute_prompt_index": "absolute_case_index//5",
                "replicate_id": "absolute_case_index%5",
                "meeting_index": "absolute_prompt_index//17",
                "variant_rank": "absolute_prompt_index%17",
                "paired_block_index": "meeting_index*5+replicate_id",
                "shard_unit": "meeting_replicate_17_arm_block",
                "shard_function": "paired_block_mod_2",
                "shard_id": "(meeting_index*5+replicate_id)%2",
                "schedule_sensitive_sampling_acknowledged": True,
                "schedule_sensitive_token_identity_replay_gate": "non_blocking",
                "user_waiver": (
                    "2026-08-17:user_explicitly_accepted_output_token_id_"
                    "variation_across_batch_concurrency"
                ),
                "token_identity_across_launches_or_batch_schedules_required": False,
                "resume_dispatch": "only_missing_tuple_keys",
                "fsynced_wal_rows": "immutable_validate_only_never_redispatch",
                "duplicate_tuple_policy": "fail_closed",
            },
            "coverage": {
                "meetings": EXPECTED_MEETINGS,
                "prompts": EXPECTED_PROMPTS,
                "variants_per_meeting": VARIANTS_PER_MEETING,
                "replicates": len(REPLICATE_SEEDS),
                "generation_rows": EXPECTED_CASES,
            },
            "mutation_policy": "write-once-new-directory-no-overwrite",
        }
    )


def prepare(
    *,
    source_sample_manifest_path: Path = DEFAULT_SOURCE_SAMPLE_MANIFEST,
    source_sample_manifest_sha256: str = SOURCE_SAMPLE_MANIFEST_SHA256,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Create a new immutable, sealed LOO K=5 preparation directory."""

    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink():
        raise LooVllmK5PreparationError("refusing a symlink output directory")
    output_dir = unresolved_output.resolve()
    if output_dir.exists():
        raise LooVllmK5PreparationError(f"output already exists: {output_dir}")

    manifest, input_rows, tokenizer, observed_source_sha = load_source_inputs(
        source_sample_manifest_path, source_sample_manifest_sha256
    )
    if observed_source_sha != SOURCE_SAMPLE_MANIFEST_SHA256:
        raise LooVllmK5PreparationError("observed source manifest SHA-256 drift")
    system_prompt, suffix, tokenizer_semantics = _load_prompt_contract(
        manifest, tokenizer
    )
    model_runtime_inventory = load_model_runtime_inventory(manifest)
    excluded_historical_greedy_k1 = load_excluded_historical_greedy_evidence()
    rows_by_id = {str(row["sample_id"]): row for row in input_rows}
    ledger_rows = build_token_ledger(
        sample_manifest=manifest,
        bound_rows=rows_by_id,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt_suffix=suffix,
    )
    prompt_counts = [int(row["prompt_token_count"]) for row in ledger_rows]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    ledger_path = output_dir / LEDGER_FILENAME
    cohort_path = output_dir / COHORT_FILENAME
    preparation_path = output_dir / PREPARATION_FILENAME
    execution_policy_path = output_dir / EXECUTION_POLICY_FILENAME
    try:
        _write_new_jsonl(ledger_path, ledger_rows)
        ledger_binding = _binding(ledger_path, rows=EXPECTED_PROMPTS)
        source_manifest_binding = _binding(source_sample_manifest_path)
        if source_manifest_binding["sha256"] != SOURCE_SAMPLE_MANIFEST_SHA256:
            raise LooVllmK5PreparationError("source manifest changed during preparation")
        source_manifest_binding["payload_sha256"] = manifest["integrity"][
            "payload_sha256"
        ]
        execution_policy = build_execution_policy(
            source_sample_manifest=source_manifest_binding,
            model_runtime_inventory=model_runtime_inventory,
            excluded_historical_greedy_k1=excluded_historical_greedy_k1,
        )
        validate_manifest_integrity(execution_policy)
        _write_new_json(execution_policy_path, execution_policy)
        execution_policy_binding = {
            **_binding(execution_policy_path),
            "payload_sha256": execution_policy["integrity"]["payload_sha256"],
        }

        generation_design = {
            "backend": "vllm-v1-two-independent-bf16-dp1-workers",
            "model_id": MODEL_ID,
            "meetings": EXPECTED_MEETINGS,
            "prompts": EXPECTED_PROMPTS,
            "variants_per_meeting": VARIANTS_PER_MEETING,
            "replicate_seeds": list(REPLICATE_SEEDS),
            "replicates": len(REPLICATE_SEEDS),
            "rows": EXPECTED_CASES,
            "all_replicates_same_distribution": True,
            "greedy_rows_reused": False,
            "canonical_case_order": (
                "meeting_order_then_variant_rank_then_replicate_id"
            ),
            "absolute_prompt_index": "absolute_case_index//5",
            "replicate_id": "absolute_case_index%5",
            "meeting_index": "absolute_prompt_index//17",
            "variant_rank": "absolute_prompt_index%17",
            "paired_block_index": "meeting_index*5+replicate_id",
            "shard_function": "paired_block_mod_2",
            "shard_id": "(meeting_index*5+replicate_id)%2",
            "absolute_chunk_size": ABSOLUTE_CHUNK_SIZE,
            "absolute_chunks": EXPECTED_CHUNKS,
            "row_seed": "derive_row_seed(replicate_seed,meeting_full_sample_id)",
            "common_random_numbers": "same-meeting-replicate-across-17-arms-v1",
            "prompt_input": "exact_persisted_token_ids_no_runtime_chat_template",
            "decoding": {
                "generation_mode": "stochastic_sampling",
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_model_len": MAX_MODEL_LEN,
                "dtype": "bfloat16",
                "quantization": None,
            },
        }
        cohort = seal_manifest(
            {
                "schema_version": COHORT_SCHEMA,
                "status": "complete",
                "sealed": True,
                "immutable": True,
                "evaluation_id": EVALUATION_ID,
                "task_contract_id": manifest["task_contract_id"],
                "source_sample_manifest": source_manifest_binding,
                "source_dataset": _copy_binding(manifest, "dataset"),
                "source_release": _copy_binding(manifest, "source_release"),
                "source_panel": _copy_binding(manifest, "source_panel"),
                "model_anchor": _copy_binding(manifest, "model_anchor"),
                "exact_merge_evidence": _copy_binding(
                    manifest, "exact_merge_evidence"
                ),
                "prompt_contract": dict(manifest["prompt_contract"]),
                "fomc_tokenizer": {
                    **dict(manifest["tokenizer"]),
                    "semantics": tokenizer_semantics,
                    "role": "single_frozen_input_and_output_tokenizer",
                },
                "model_runtime_inventory": model_runtime_inventory,
                "excluded_historical_greedy_k1": excluded_historical_greedy_k1,
                "execution_policy": execution_policy_binding,
                "token_ledger": {
                    **ledger_binding,
                    "row_schema_version": LEDGER_ROW_SCHEMA,
                    "full_prompt_token_ids_persisted": True,
                    "vllm_runtime_chat_templating": False,
                },
                "coverage": {
                    "meetings": EXPECTED_MEETINGS,
                    "prompts": EXPECTED_PROMPTS,
                    "variants_per_meeting": VARIANTS_PER_MEETING,
                    "replicates": len(REPLICATE_SEEDS),
                    "generation_rows": EXPECTED_CASES,
                },
                "prompt_budget": {
                    "minimum_prompt_tokens": min(prompt_counts),
                    "maximum_prompt_tokens": max(prompt_counts),
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "maximum_total_tokens": max(prompt_counts) + MAX_NEW_TOKENS,
                    "max_model_len": MAX_MODEL_LEN,
                    "over_budget_rows": 0,
                },
                "generation_design": generation_design,
                "preparation_runtime": {
                    "python": platform.python_version(),
                    "transformers": _package_version("transformers"),
                    "tokenizers": _package_version("tokenizers"),
                    "prompt_token_ids_materialized_once": True,
                },
                "implementation": {
                    "preparer": _binding(Path(__file__).resolve()),
                    "sealed_source_loader": _binding(
                        Path(str(source_contract.__file__)).resolve()
                    ),
                },
                "mutation_policy": "write-once-new-directory-no-overwrite",
            }
        )
        _write_new_json(cohort_path, cohort)
        cohort_binding = {
            **_binding(cohort_path),
            "payload_sha256": cohort["integrity"]["payload_sha256"],
        }
        preparation = seal_manifest(
            {
                "schema_version": PREPARATION_SCHEMA,
                "status": "complete",
                "sealed": True,
                "immutable": True,
                "evaluation_id": EVALUATION_ID,
                "cohort": cohort_binding,
                "execution_policy": execution_policy_binding,
                "token_ledger": ledger_binding,
                "source_sample_manifest": source_manifest_binding,
                "coverage": {
                    "meetings": EXPECTED_MEETINGS,
                    "prompts": EXPECTED_PROMPTS,
                    "variants_per_meeting": VARIANTS_PER_MEETING,
                    "replicates": len(REPLICATE_SEEDS),
                    "generation_rows": EXPECTED_CASES,
                    "absolute_chunks": EXPECTED_CHUNKS,
                },
                "validation": {
                    "sealed_source_manifest_sha256": "passed",
                    "deep_source_matrix_validation": "passed",
                    "exact_prompt_token_ids_persisted": True,
                    "prompt_token_count_mismatches": 0,
                    "prompt_budget_violations": 0,
                    "runtime_chat_template_application": False,
                    "all_five_replicates_stochastic_same_distribution": True,
                    "greedy_replicate_reuse": False,
                    "excluded_historical_greedy_k1_evidence_bound": True,
                },
                "mutation_policy": "write-once-new-directory-no-overwrite",
            }
        )
        # Assert both payloads are valid before publishing the terminal marker.
        validate_manifest_integrity(cohort)
        validate_manifest_integrity(preparation)
        _write_new_json(preparation_path, preparation)
        _fsync_directory(output_dir)
        return preparation
    except Exception:
        # Roll back only the three explicitly owned files.  Refuse to remove
        # the directory if an unexpected concurrent artifact appeared.
        for path in (
            preparation_path,
            cohort_path,
            execution_policy_path,
            ledger_path,
        ):
            with contextlib.suppress(OSError):
                path.unlink()
        with contextlib.suppress(OSError):
            output_dir.rmdir()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sample-manifest",
        type=Path,
        default=DEFAULT_SOURCE_SAMPLE_MANIFEST,
    )
    parser.add_argument(
        "--source-sample-manifest-sha256",
        default=SOURCE_SAMPLE_MANIFEST_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            source_sample_manifest_path=args.source_sample_manifest,
            source_sample_manifest_sha256=args.source_sample_manifest_sha256,
            output_dir=args.output_dir,
        )
    except (LooVllmK5PreparationError, OSError, ValueError) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        )
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABSOLUTE_CHUNK_SIZE",
    "COHORT_SCHEMA",
    "DEFAULT_SOURCE_SAMPLE_MANIFEST",
    "DEFAULT_OUTPUT_DIR",
    "EVALUATION_ID",
    "EXPECTED_CASES",
    "EXPECTED_CHUNKS",
    "EXPECTED_MEETINGS",
    "EXPECTED_PROMPTS",
    "EXECUTION_POLICY_SCHEMA",
    "HISTORICAL_GREEDY_GENERATION_MANIFEST_SHA256",
    "HISTORICAL_GREEDY_GENERATIONS_SHA256",
    "HISTORICAL_GREEDY_SCORE_MANIFEST_SHA256",
    "LEDGER_ROW_SCHEMA",
    "LooVllmK5PreparationError",
    "MAX_MODEL_LEN",
    "MAX_NEW_TOKENS",
    "MAX_PROMPT_TOKENS",
    "PREPARATION_SCHEMA",
    "REPLICATE_SEEDS",
    "SOURCE_SAMPLE_MANIFEST_SHA256",
    "VARIANTS_PER_MEETING",
    "build_token_ledger",
    "build_execution_policy",
    "canonical_case_coordinates",
    "load_model_runtime_inventory",
    "load_excluded_historical_greedy_evidence",
    "load_source_inputs",
    "prepare",
]
