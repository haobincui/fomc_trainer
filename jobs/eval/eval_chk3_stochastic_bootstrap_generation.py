"""Generate the sealed CHK0/CHK1/CHK3 N190 x K5 bootstrap cohort.

This is a versioned stochastic companion to ``eval_chk3_native_three_model``.
The historical greedy runner and its N12 evidence remain untouched.  A formal
invocation runs the three immutable, exact-merged model artifacts in the
preregistered order CHK1 -> CHK3 -> CHK0 on physical GPU0 only.

Generation is intentionally performed only once.  Hierarchical bootstrap
resampling and semantic scoring are CPU/offline follow-up steps over the
persisted full text and token IDs produced here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as semantic_gate
from open_r1.validator import loo_generation_spec as loo_spec
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-native-stochastic-bootstrap-n190-k5-v1"
TASK_CONTRACT_ID = native_eval.TASK_CONTRACT_ID
SAMPLE_MANIFEST_SCHEMA_VERSION = "chk3-stochastic-bootstrap-full-test-samples-v1"
ROW_SCHEMA_VERSION = "chk3-stochastic-bootstrap-generation-row-v1"
RUN_MANIFEST_SCHEMA_VERSION = "chk3-stochastic-bootstrap-generation-run-manifest-v1"
SUITE_MANIFEST_SCHEMA_VERSION = "chk3-stochastic-bootstrap-generation-suite-v1"
STATE_SCHEMA_VERSION = "chk3-stochastic-bootstrap-generation-state-v1"

MODEL_ORDER = ("chk1", "chk3", "chk0")
MODEL_LABELS = {
    "chk0": "chk0-base",
    "chk1": "chk1-cp200-exact-merged",
    "chk3": "chk3-cp318-exact-merged",
}
ANCHOR_SPECS = {
    "chk0": {
        "path": ROOT
        / "output/evaluation/main/chk3_native_analysis_to_minutes_n12_cp250_20260811_v1/chk0/manifest.json",
        "sha256": "745af8f2f01063d3143b1a7bd8c1037b632d5dea70303df673e1075ba4b4846c",
        "payload_sha256": "828da316a080a88fd75fd6f9bb070e6fdf9ea24c8b16b3ee24ca8e39434aa216",
    },
    "chk1": {
        "path": ROOT
        / "output/evaluation/main/chk3_native_analysis_to_minutes_n12_cp250_20260811_v1/chk1/manifest.json",
        "sha256": "0bb7736ed01e9efc91b6d3d2594af34ddc04e6c89087edcd63d1ee283f776085",
        "payload_sha256": "3c993e281f66943cee8c84670f50ece8f2f77dcdf1aa68c2c521665031878c5f",
    },
    "chk3": {
        "path": ROOT
        / "output/evaluation/main/chk3_checkpoint_selection_cp200_cp318_20260811_v1/native_n12_merged/cp318/manifest.json",
        "sha256": "8bc24bbf7827bf210ffffeb443c88740baf3e9773c66f9456fdaf13bbd4105c4",
        "payload_sha256": "ce0e0410f1829d00f6958134290a70c74dae420cc25e5668943237f209c809aa",
    },
}

REPLICATE_SEEDS = (20260811, 21260811, 22260811, 23260811, 24260811)
BOOTSTRAP_SEED = 20260812
EXPECTED_TEST_ROWS = 190
EXPECTED_MEETING_IDS = (
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
)
EXPECTED_IDENTITY_ROWS = 19

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 3072
TAIL_TOKENS = 1024
ATTN_IMPLEMENTATION = "sdpa"
GPU_LOCK_PATH = Path("/tmp/fomc_trainer_chk3_stochastic_bootstrap_gpu0.lock")
MEETING_ID_RE = re.compile(r"^chk1-analysis-(\d{4}-\d{2}-\d{2})-[0-9a-f]+$")
MODEL_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StochasticBootstrapGenerationError(RuntimeError):
    """The frozen stochastic-generation contract was violated."""


class ExternalGpuProcessDetectedError(StochasticBootstrapGenerationError):
    """Another process entered GPU0 after this run loaded its model."""


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
        raise StochasticBootstrapGenerationError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return native_eval._read_json(path)
    except native_eval.NativeThreeModelEvalError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return native_eval._read_jsonl(path)
    except native_eval.NativeThreeModelEvalError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise StochasticBootstrapGenerationError(f"refusing to overwrite artifact: {path}")
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
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise StochasticBootstrapGenerationError(
                f"refusing to overwrite artifact: {path}"
            ) from exc
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        native_eval._atomic_write_json(path, value)
        _fsync_directory(path.parent)
    except native_eval.NativeThreeModelEvalError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc


def _sha256_file(path: Path) -> str:
    return native_eval.native_probe.common_probe.sha256_file(path)


def _sha256_text(text: str) -> str:
    return native_eval.native_probe.common_probe.sha256_text(text)


def _file_binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    try:
        return native_eval._file_binding(path, payload_sha256=payload_sha256)
    except native_eval.NativeThreeModelEvalError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((_canonical_json(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _jsonl_rows_bytes(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(len((_canonical_json(row) + "\n").encode("utf-8")) for row in rows)


def _meeting_id(sample_id: str) -> str:
    match = MEETING_ID_RE.fullmatch(sample_id)
    if match is None or match.group(1) not in EXPECTED_MEETING_IDS:
        raise StochasticBootstrapGenerationError(
            f"sample ID has no expected meeting cluster: {sample_id!r}"
        )
    return match.group(1)


def _sampling_contract() -> dict[str, Any]:
    return {
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "batch_size": 1,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "bfloat16",
        "bnb_4bit_use_double_quant": True,
        "model_dtype": "bfloat16",
        "attn_implementation": ATTN_IMPLEMENTATION,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "row_seed": "derive_row_seed(replicate_seed,sample_id)",
        "rng_reset": "transformers.set_seed+torch.manual_seed+torch.cuda.manual_seed_all-per-tuple-v1",
        "canonical_tuple_order": "sample_manifest_order_then_replicate_id",
        "recommended_model_order": list(MODEL_ORDER),
    }


def _length_buckets(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        records,
        key=lambda row: (int(row["completion_tokens"]), str(row["sample_id"])),
    )
    boundaries = (0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered))
    result: dict[str, str] = {}
    for index, bucket in enumerate(("short", "medium", "long")):
        for row in ordered[boundaries[index] : boundaries[index + 1]]:
            result[str(row["sample_id"])] = bucket
    return result


def build_full_test_sample_manifest(
    *,
    test_data: Path,
    test_row_manifest: Path,
    release_manifest: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    training_config: Path,
) -> dict[str, Any]:
    """Seal all 190 release-ordered held-out test rows and their 13 meetings."""

    test_data = test_data.expanduser().resolve()
    test_row_manifest = test_row_manifest.expanduser().resolve()
    release_manifest = release_manifest.expanduser().resolve()
    training_config = training_config.expanduser().resolve()
    release = _read_json(release_manifest)
    if release.get("schema_version") != "chk3-minutes-training-release-v1":
        raise StochasticBootstrapGenerationError("unsupported chk3 release schema")
    if release.get("immutable") is not True or release.get("quality_status") != "passed":
        raise StochasticBootstrapGenerationError("release is not immutable and passed")
    split_counts = release.get("split_counts")
    if not isinstance(split_counts, Mapping) or split_counts.get("test") != EXPECTED_TEST_ROWS:
        raise StochasticBootstrapGenerationError("formal release test split must be N190")
    data_rows = _read_jsonl(test_data)
    row_records = _read_jsonl(test_row_manifest)
    if len(data_rows) != EXPECTED_TEST_ROWS or len(row_records) != EXPECTED_TEST_ROWS:
        raise StochasticBootstrapGenerationError("test data/manifest must each have 190 rows")
    try:
        data_binding = native_eval._verify_release_file(
            release=release,
            release_root=release_manifest.parent,
            relative_path="minutes_alignment/test.jsonl",
            actual_path=test_data,
            expected_rows=EXPECTED_TEST_ROWS,
        )
        row_binding = native_eval._verify_release_file(
            release=release,
            release_root=release_manifest.parent,
            relative_path="minutes_alignment/manifests/test.jsonl",
            actual_path=test_row_manifest,
            expected_rows=EXPECTED_TEST_ROWS,
        )
        system_prompt, suffix, config_sha = native_eval.native_probe._load_prompt_contract(
            training_config
        )
    except (native_eval.NativeThreeModelEvalError, native_eval.native_probe.Chk3ProbeError) as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc

    buckets = _length_buckets(row_records)
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, (row, record) in enumerate(
        zip(data_rows, row_records, strict=True), start=1
    ):
        prompt = row.get("prompt")
        response = row.get("response")
        sample_id = record.get("sample_id")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise StochasticBootstrapGenerationError(f"invalid test row {line_number}")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise StochasticBootstrapGenerationError(
                f"invalid/duplicate sample ID at row {line_number}"
            )
        seen_ids.add(sample_id)
        if record.get("split") != "test" or record.get("source_split") != "test":
            raise StochasticBootstrapGenerationError("non-test row in held-out manifest")
        prompt_sha = _sha256_text(prompt)
        response_sha = _sha256_text(response)
        if record.get("prompt_sha256") != prompt_sha or record.get(
            "response_sha256"
        ) != response_sha:
            raise StochasticBootstrapGenerationError(
                f"release row hash drift at row {line_number}"
            )
        analysis = native_eval.native_probe.extract_source_analysis(prompt)
        reference = native_eval._final_answer(response)
        if not reference:
            raise StochasticBootstrapGenerationError("reference has no Minutes answer")
        try:
            prompt_ids = native_eval.native_probe._prompt_ids(
                tokenizer,
                native_eval.native_probe._messages(
                    row, system_prompt=system_prompt, user_prompt_suffix=suffix
                ),
            )
        except native_eval.native_probe.Chk3ProbeError as exc:
            raise StochasticBootstrapGenerationError(str(exc)) from exc
        samples.append(
            {
                "sample_id": sample_id,
                "meeting_id": _meeting_id(sample_id),
                "line_number": line_number,
                "release_row_manifest_line_number": line_number,
                "length_bucket": buckets[sample_id],
                "prompt_sha256": prompt_sha,
                "response_sha256": response_sha,
                "analysis_sha256": _sha256_text(analysis),
                "release_analysis_sha256": record.get("analysis_sha256"),
                "reference_minutes_sha256": _sha256_text(reference),
                "completion_tokens": record.get("completion_tokens"),
                "prompt_token_count": len(prompt_ids),
                "analysis_reference_exact_identity": analysis == reference,
                "normalized_identity": native_eval._normalized_identity_text(analysis)
                == native_eval._normalized_identity_text(reference),
                "punctuation_insensitive_identity": (
                    native_eval._punctuation_insensitive_identity_text(analysis)
                    == native_eval._punctuation_insensitive_identity_text(reference)
                ),
                "source_numeric_multiset_sha256": native_eval.native_probe._numeric_hash(
                    analysis
                ),
                "source_date_set_sha256": native_eval.native_probe._date_hash(analysis),
                "source_numeric_occurrences": sum(
                    native_eval.native_probe._numeric_values(analysis).values()
                ),
                "source_date_values": len(native_eval.native_probe._date_values(analysis)),
            }
        )
    meetings = sorted({str(sample["meeting_id"]) for sample in samples})
    if tuple(meetings) != EXPECTED_MEETING_IDS:
        raise StochasticBootstrapGenerationError("held-out meeting inventory drift")
    if sum(bool(sample["normalized_identity"]) for sample in samples) != EXPECTED_IDENTITY_ROWS:
        raise StochasticBootstrapGenerationError("normalized-identity count is not N19")
    try:
        tokenizer_binding = dict(native_eval.native_probe._tokenizer_fingerprint(tokenizer_path))
    except native_eval.native_probe.Chk3ProbeError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc
    payload = {
        "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "immutable": True,
        "release": {
            "path": str(release_manifest),
            "sha256": _sha256_file(release_manifest),
            "release_id": release.get("release_id"),
            "schema_version": release.get("schema_version"),
        },
        "dataset": {**data_binding, "split": "test"},
        "release_row_manifest": row_binding,
        "prompt_contract": {
            "training_config": str(training_config),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": _sha256_text(system_prompt),
            "user_prompt_suffix_sha256": _sha256_text(suffix) if suffix is not None else None,
        },
        "tokenizer": tokenizer_binding,
        "selection": {
            "algorithm": "all-held-out-test-rows-release-order-v1",
            "evaluation_split": "test",
            "checkpoint_selection_split": "validation",
            "rows": EXPECTED_TEST_ROWS,
            "meeting_clusters": len(EXPECTED_MEETING_IDS),
            "meeting_ids": list(EXPECTED_MEETING_IDS),
            "ordering": "release_jsonl_line_number_ascending",
            "normalized_identity_rows": EXPECTED_IDENTITY_ROWS,
            "identity_normalization": "NFKC+casefold+whitespace-collapse-v1",
        },
        "generation_design": {
            "models": list(MODEL_ORDER),
            "replicate_seeds": list(REPLICATE_SEEDS),
            "rows_per_model": EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS),
            "total_rows": EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS) * len(MODEL_ORDER),
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "samples": samples,
    }
    return seal_manifest(payload)


def load_full_test_sample_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    """Load and deeply validate the externally hash-pinned N190 manifest."""

    resolved = path.expanduser().resolve()
    observed_sha = _sha256_file(resolved)
    if observed_sha != expected_sha256:
        raise StochasticBootstrapGenerationError(
            "sample manifest file SHA256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha}"
        )
    manifest = _read_json(resolved)
    try:
        validate_manifest_integrity(manifest)
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"sample manifest integrity failed: {exc}"
        ) from exc
    if manifest.get("schema_version") != SAMPLE_MANIFEST_SCHEMA_VERSION:
        raise StochasticBootstrapGenerationError("unsupported full-test manifest schema")
    if (
        manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("task_contract_id") != TASK_CONTRACT_ID
        or manifest.get("immutable") is not True
    ):
        raise StochasticBootstrapGenerationError("full-test manifest contract drift")
    selection = manifest.get("selection")
    design = manifest.get("generation_design")
    samples = manifest.get("samples")
    if not isinstance(selection, Mapping) or not isinstance(design, Mapping):
        raise StochasticBootstrapGenerationError("full-test selection/design is missing")
    if not isinstance(samples, list) or len(samples) != EXPECTED_TEST_ROWS:
        raise StochasticBootstrapGenerationError("full-test manifest is not N190")
    expected_selection = {
        "algorithm": "all-held-out-test-rows-release-order-v1",
        "evaluation_split": "test",
        "checkpoint_selection_split": "validation",
        "rows": EXPECTED_TEST_ROWS,
        "meeting_clusters": len(EXPECTED_MEETING_IDS),
        "meeting_ids": list(EXPECTED_MEETING_IDS),
        "ordering": "release_jsonl_line_number_ascending",
        "normalized_identity_rows": EXPECTED_IDENTITY_ROWS,
        "identity_normalization": "NFKC+casefold+whitespace-collapse-v1",
    }
    if dict(selection) != expected_selection:
        raise StochasticBootstrapGenerationError("full-test selection contract drift")
    if dict(design) != {
        "models": list(MODEL_ORDER),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "rows_per_model": EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS),
        "total_rows": EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS) * len(MODEL_ORDER),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }:
        raise StochasticBootstrapGenerationError("full-test generation design drift")
    sample_ids = [sample.get("sample_id") for sample in samples if isinstance(sample, Mapping)]
    if len(sample_ids) != EXPECTED_TEST_ROWS or len(set(sample_ids)) != EXPECTED_TEST_ROWS:
        raise StochasticBootstrapGenerationError("sample IDs are missing or duplicated")
    if [sample.get("line_number") for sample in samples] != list(
        range(1, EXPECTED_TEST_ROWS + 1)
    ):
        raise StochasticBootstrapGenerationError("sample manifest is out of release order")
    meetings = sorted({sample.get("meeting_id") for sample in samples})
    if meetings != list(EXPECTED_MEETING_IDS):
        raise StochasticBootstrapGenerationError("sample meeting inventory drift")
    if sum(bool(sample.get("normalized_identity")) for sample in samples) != EXPECTED_IDENTITY_ROWS:
        raise StochasticBootstrapGenerationError("sample identity denominator drift")
    _load_bound_rows(manifest)
    return manifest, observed_sha


def _load_bound_rows(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Rebind every sample to the immutable release row and recompute hashes."""

    release_binding = manifest.get("release")
    dataset_binding = manifest.get("dataset")
    row_manifest_binding = manifest.get("release_row_manifest")
    prompt_contract = manifest.get("prompt_contract")
    tokenizer_binding = manifest.get("tokenizer")
    if not all(
        isinstance(value, Mapping)
        for value in (
            release_binding,
            dataset_binding,
            row_manifest_binding,
            prompt_contract,
            tokenizer_binding,
        )
    ):
        raise StochasticBootstrapGenerationError("sample source bindings are incomplete")
    paths_and_bindings = (
        (Path(str(release_binding.get("path"))).expanduser().resolve(), release_binding),
        (Path(str(dataset_binding.get("path"))).expanduser().resolve(), dataset_binding),
        (
            Path(str(row_manifest_binding.get("path"))).expanduser().resolve(),
            row_manifest_binding,
        ),
    )
    for source_path, binding in paths_and_bindings:
        if _sha256_file(source_path) != binding.get("sha256"):
            raise StochasticBootstrapGenerationError(
                f"bound source artifact changed: {source_path}"
            )
    release_path, data_path, row_path = (item[0] for item in paths_and_bindings)
    release = _read_json(release_path)
    if release.get("immutable") is not True or release.get("quality_status") != "passed":
        raise StochasticBootstrapGenerationError("bound release status changed")
    data_rows = _read_jsonl(data_path)
    row_records = _read_jsonl(row_path)
    if len(data_rows) != EXPECTED_TEST_ROWS or len(row_records) != EXPECTED_TEST_ROWS:
        raise StochasticBootstrapGenerationError("bound source row count changed")
    config_path = Path(str(prompt_contract.get("training_config"))).expanduser().resolve()
    try:
        system_prompt, suffix, config_sha = native_eval.native_probe._load_prompt_contract(
            config_path
        )
    except native_eval.native_probe.Chk3ProbeError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc
    if (
        config_sha != prompt_contract.get("training_config_sha256")
        or _sha256_text(system_prompt) != prompt_contract.get("system_prompt_sha256")
        or (_sha256_text(suffix) if suffix is not None else None)
        != prompt_contract.get("user_prompt_suffix_sha256")
    ):
        raise StochasticBootstrapGenerationError("bound prompt contract changed")
    tokenizer_path = Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
    tokenizer_files = tokenizer_binding.get("files")
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise StochasticBootstrapGenerationError("tokenizer file binding is missing")
    for name, expected_sha in tokenizer_files.items():
        file_path = tokenizer_path / str(name)
        if _sha256_file(file_path) != expected_sha:
            raise StochasticBootstrapGenerationError(
                f"bound tokenizer file changed: {name}"
            )

    bound: dict[str, Mapping[str, Any]] = {}
    for index, (sample, row, record) in enumerate(
        zip(manifest["samples"], data_rows, row_records, strict=True), start=1
    ):
        if not isinstance(sample, Mapping):
            raise StochasticBootstrapGenerationError("sample row is not an object")
        sample_id = sample.get("sample_id")
        prompt = row.get("prompt")
        response = row.get("response")
        if (
            sample.get("line_number") != index
            or sample.get("release_row_manifest_line_number") != index
            or record.get("sample_id") != sample_id
            or not isinstance(sample_id, str)
            or not isinstance(prompt, str)
            or not isinstance(response, str)
        ):
            raise StochasticBootstrapGenerationError(
                f"release-order sample binding drift at row {index}"
            )
        analysis = native_eval.native_probe.extract_source_analysis(prompt)
        reference = native_eval._final_answer(response)
        checks = {
            "meeting_id": _meeting_id(sample_id),
            "prompt_sha256": _sha256_text(prompt),
            "response_sha256": _sha256_text(response),
            "analysis_sha256": _sha256_text(analysis),
            "reference_minutes_sha256": _sha256_text(reference),
            "completion_tokens": record.get("completion_tokens"),
            "source_numeric_multiset_sha256": native_eval.native_probe._numeric_hash(
                analysis
            ),
            "source_date_set_sha256": native_eval.native_probe._date_hash(analysis),
            "source_numeric_occurrences": sum(
                native_eval.native_probe._numeric_values(analysis).values()
            ),
            "source_date_values": len(native_eval.native_probe._date_values(analysis)),
            "analysis_reference_exact_identity": analysis == reference,
            "normalized_identity": native_eval._normalized_identity_text(analysis)
            == native_eval._normalized_identity_text(reference),
            "punctuation_insensitive_identity": (
                native_eval._punctuation_insensitive_identity_text(analysis)
                == native_eval._punctuation_insensitive_identity_text(reference)
            ),
        }
        for key, expected in checks.items():
            if sample.get(key) != expected:
                raise StochasticBootstrapGenerationError(
                    f"bound sample field drift at {key} (row {index})"
                )
        if record.get("prompt_sha256") != checks["prompt_sha256"] or record.get(
            "response_sha256"
        ) != checks["response_sha256"]:
            raise StochasticBootstrapGenerationError("release manifest row hash drift")
        bound[sample_id] = row
    return bound


def _source_hashes(manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    tokenizer_files = manifest["tokenizer"]["files"]
    return {
        "sample_manifest_sha256": manifest_sha256,
        "release_manifest_sha256": manifest["release"]["sha256"],
        "test_data_sha256": manifest["dataset"]["sha256"],
        "test_row_manifest_sha256": manifest["release_row_manifest"]["sha256"],
        "training_config_sha256": manifest["prompt_contract"][
            "training_config_sha256"
        ],
        "tokenizer_file_inventory_sha256": _sha256_text(
            _canonical_json(tokenizer_files)
        ),
        "implementation_sources": {
            "stochastic_runner": _file_binding(Path(__file__).resolve()),
            "native_generation_contract": _file_binding(
                Path(str(native_eval.__file__)).resolve()
            ),
            "hard_gate_contract": _file_binding(
                Path(str(semantic_gate.__file__)).resolve()
            ),
            "row_seed_contract": _file_binding(Path(str(loo_spec.__file__)).resolve()),
        },
    }


def load_frozen_anchor(model_id: str) -> dict[str, Any]:
    """Validate one historical sealed run and its exact current model inventory."""

    if model_id not in MODEL_ORDER:
        raise StochasticBootstrapGenerationError(f"unsupported model ID: {model_id}")
    spec = ANCHOR_SPECS[model_id]
    path = Path(spec["path"]).expanduser().resolve()
    if _sha256_file(path) != spec["sha256"]:
        raise StochasticBootstrapGenerationError(f"{model_id} anchor file SHA drift")
    anchor = _read_json(path)
    try:
        payload_sha = validate_manifest_integrity(
            anchor, expected_payload_sha256=str(spec["payload_sha256"])
        )
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"{model_id} anchor integrity failed: {exc}"
        ) from exc
    if (
        anchor.get("schema_version") != native_eval.RUN_MANIFEST_SCHEMA_VERSION
        or anchor.get("status") != "complete"
        or anchor.get("task_contract_id") != TASK_CONTRACT_ID
        or anchor.get("stage_id") != model_id
        or anchor.get("adapter") is not None
    ):
        raise StochasticBootstrapGenerationError(f"{model_id} anchor contract drift")
    model = anchor.get("model")
    if not isinstance(model, Mapping):
        raise StochasticBootstrapGenerationError(f"{model_id} anchor has no model")
    model_path = Path(str(model.get("path"))).expanduser().resolve()
    try:
        observed = dict(native_eval.native_probe._fingerprint_model(model_path))
    except native_eval.native_probe.Chk3ProbeError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc
    if observed != dict(model):
        raise StochasticBootstrapGenerationError(
            f"{model_id} current model inventory differs from sealed anchor"
        )
    return {
        "model_id": model_id,
        "model_label": MODEL_LABELS[model_id],
        "model_path": model_path,
        "model": observed,
        "anchor": {
            "path": str(path),
            "sha256": spec["sha256"],
            "payload_sha256": payload_sha,
        },
    }


def _generation_metrics(row: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    try:
        audit = semantic_gate._recompute_core_hard_gate(row)
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"generation hard-gate recomputation failed: {exc}"
        ) from exc
    metrics = {
        "structure_delivery": bool(audit["delivery_valid"])
        and bool(audit["native_structure_valid"]),
        "numeric_fidelity": bool(audit["numeric_multiset_preserved"]),
        "date_fidelity": bool(audit["date_set_preserved"]),
        "degeneration_free": bool(audit["degeneration_free"]),
    }
    if all(metrics.values()) is not bool(audit["preregistered_core_valid"]):
        raise StochasticBootstrapGenerationError(
            "stored four-metric gate disagrees with preregistered core"
        )
    return metrics, audit


def build_stochastic_result(
    *,
    text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int] | set[int] | None,
    source_prompt: str,
    source_analysis: str,
    reference_response: str,
    pad_token_id: int,
    model_id: str,
    model_label: str,
    sample_manifest_sha256: str,
    sample: Mapping[str, Any],
    replicate_id: int,
    replicate_seed: int,
    source_artifact_sha256s: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one fully persisted stochastic row and its four hard-gate slots."""

    if model_id not in MODEL_ORDER or not MODEL_LABEL_RE.fullmatch(model_label):
        raise StochasticBootstrapGenerationError("invalid model identity")
    if (
        isinstance(replicate_id, bool)
        or not isinstance(replicate_id, int)
        or replicate_id < 0
        or replicate_id >= len(REPLICATE_SEEDS)
        or replicate_seed != REPLICATE_SEEDS[replicate_id]
    ):
        raise StochasticBootstrapGenerationError("invalid replicate identity")
    sample_id = sample.get("sample_id")
    meeting_id = sample.get("meeting_id")
    if not isinstance(sample_id, str) or meeting_id != _meeting_id(sample_id):
        raise StochasticBootstrapGenerationError("invalid sample/meeting identity")
    row_seed = derive_row_seed(replicate_seed, sample_id)
    try:
        base = native_eval.build_full_result(
            text=text,
            generated_token_ids=generated_token_ids,
            eos_token_ids=eos_token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            tail_tokens=TAIL_TOKENS,
            source_prompt=source_prompt,
            source_analysis=source_analysis,
            reference_response=reference_response,
            load_in_4bit=True,
            attn_implementation=ATTN_IMPLEMENTATION,
            pad_token_id=pad_token_id,
            stage_id=model_id,
            model_label=model_label,
            sample_manifest_sha256=sample_manifest_sha256,
            sample=sample,
            seed=row_seed,
        )
    except native_eval.NativeThreeModelEvalError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc
    sampling = _sampling_contract()
    row: dict[str, Any] = {
        **base,
        "schema_version": ROW_SCHEMA_VERSION,
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "meeting_id": meeting_id,
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "row_seed": row_seed,
        "generation_mode": sampling["generation_mode"],
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "batch_size": 1,
        "input_token_count": sample.get("prompt_token_count"),
        "input_truncated": False,
        "sampling_contract_sha256": _sha256_text(_canonical_json(sampling)),
        "source_artifact_sha256s": dict(source_artifact_sha256s),
        "semantic_status": "pending_offline",
    }
    generation_metrics, audit = _generation_metrics(row)
    row["generation_metrics"] = generation_metrics
    row["preregistered_core_valid"] = bool(audit["preregistered_core_valid"])
    row["preregistered_core_failures"] = list(
        audit["preregistered_core_failures"]
    )
    row["six_metrics"] = {
        **generation_metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }
    return row


def validate_stochastic_result(
    row: Mapping[str, Any],
    *,
    model_id: str,
    model_label: str,
    sample_manifest_sha256: str,
    sample: Mapping[str, Any],
    source_analysis: str,
    reference_response: str,
    replicate_id: int,
    replicate_seed: int,
    source_artifact_sha256s: Mapping[str, Any],
    tokenizer: Any | None = None,
) -> None:
    """Deep-recompute a row from persisted text and tokens; reject any drift."""

    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise StochasticBootstrapGenerationError("unsupported stochastic row schema")
    sample_id = sample.get("sample_id")
    expected_identity = {
        "evaluation_id": EVALUATION_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "model_id": model_id,
        "stage_id": model_id,
        "model_label": model_label,
        "sample_id": sample_id,
        "meeting_id": sample.get("meeting_id"),
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "seed": derive_row_seed(replicate_seed, str(sample_id)),
        "row_seed": derive_row_seed(replicate_seed, str(sample_id)),
        "sample_manifest_sha256": sample_manifest_sha256,
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "batch_size": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "load_in_4bit": True,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "input_token_count": sample.get("prompt_token_count"),
        "input_truncated": False,
        "sampling_contract_sha256": _sha256_text(
            _canonical_json(_sampling_contract())
        ),
        "source_artifact_sha256s": dict(source_artifact_sha256s),
        "semantic_status": "pending_offline",
    }
    for key, expected in expected_identity.items():
        if row.get(key) != expected:
            raise StochasticBootstrapGenerationError(
                f"stochastic row contract drift at {key}"
            )
    base = copy.deepcopy(dict(row))
    base["schema_version"] = native_eval.ROW_SCHEMA_VERSION
    base["generation_mode"] = "greedy"
    base["do_sample"] = False
    try:
        native_eval.validate_full_result(
            base,
            stage_id=model_id,
            model_label=model_label,
            sample_manifest_sha256=sample_manifest_sha256,
            sample=sample,
            source_analysis=source_analysis,
            reference_response=reference_response,
            tokenizer=tokenizer,
        )
    except native_eval.NativeThreeModelEvalError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc
    generation_metrics, audit = _generation_metrics(row)
    if row.get("generation_metrics") != generation_metrics:
        raise StochasticBootstrapGenerationError("stored generation metrics drift")
    expected_six = {
        **generation_metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }
    if row.get("six_metrics") != expected_six:
        raise StochasticBootstrapGenerationError("six-metric placeholder drift")
    if row.get("preregistered_core_valid") is not bool(
        audit["preregistered_core_valid"]
    ) or row.get("preregistered_core_failures") != list(
        audit["preregistered_core_failures"]
    ):
        raise StochasticBootstrapGenerationError("stored core-gate audit drift")


def _canonical_cases(
    samples: Sequence[Mapping[str, Any]], *, smoke: bool
) -> list[tuple[Mapping[str, Any], int, int]]:
    selected_samples = list(samples[:1] if smoke else samples)
    selected_seeds = REPLICATE_SEEDS[:1] if smoke else REPLICATE_SEEDS
    return [
        (sample, replicate_id, replicate_seed)
        for sample in selected_samples
        for replicate_id, replicate_seed in enumerate(selected_seeds)
    ]


def validate_progress_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    model_label: str,
    sample_manifest_sha256: str,
    samples: Sequence[Mapping[str, Any]],
    bound_rows: Mapping[str, Mapping[str, Any]],
    source_artifact_sha256s: Mapping[str, Any],
    smoke: bool,
    tokenizer: Any | None = None,
    complete: bool = False,
) -> None:
    """Require an exact canonical tuple prefix; gaps/reordering are fatal."""

    cases = _canonical_cases(samples, smoke=smoke)
    if len(rows) > len(cases) or (complete and len(rows) != len(cases)):
        raise StochasticBootstrapGenerationError("generation tuple count drift")
    observed_keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        sample, replicate_id, replicate_seed = cases[index]
        expected_key = (model_id, sample["sample_id"], replicate_id)
        observed_key = (
            row.get("model_id"),
            row.get("sample_id"),
            row.get("replicate_id"),
        )
        if observed_key in observed_keys:
            raise StochasticBootstrapGenerationError("duplicate generation tuple")
        observed_keys.add(observed_key)
        if observed_key != expected_key:
            raise StochasticBootstrapGenerationError(
                f"out-of-order/missing generation tuple at index {index}"
            )
        source = bound_rows[str(sample["sample_id"])]
        validate_stochastic_result(
            row,
            model_id=model_id,
            model_label=model_label,
            sample_manifest_sha256=sample_manifest_sha256,
            sample=sample,
            source_analysis=native_eval.native_probe.extract_source_analysis(
                str(source["prompt"])
            ),
            reference_response=str(source["response"]),
            replicate_id=replicate_id,
            replicate_seed=replicate_seed,
            source_artifact_sha256s=source_artifact_sha256s,
            tokenizer=tokenizer,
        )


def _external_gpu0_compute_processes() -> list[dict[str, Any]]:
    """Return physical GPU0 compute processes without changing GPU state."""

    command = (
        "nvidia-smi",
        "--id=0",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StochasticBootstrapGenerationError(
            f"cannot inspect physical GPU0 with nvidia-smi: {exc}"
        ) from exc
    processes: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or "No running processes" in line or line.startswith("N/A"):
            continue
        columns = [value.strip() for value in line.split(",", 1)]
        if not columns or not columns[0].isdigit():
            raise StochasticBootstrapGenerationError(
                f"unexpected nvidia-smi compute row: {line!r}"
            )
        pid = int(columns[0])
        if pid == os.getpid():
            continue
        processes.append(
            {
                "pid": pid,
                "process_name": columns[1] if len(columns) == 2 else "",
            }
        )
    return processes


def _physical_gpu0_identity() -> dict[str, str]:
    command = (
        "nvidia-smi",
        "--id=0",
        "--query-gpu=uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StochasticBootstrapGenerationError(
            f"cannot bind physical GPU0 identity: {exc}"
        ) from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise StochasticBootstrapGenerationError("physical GPU0 identity is ambiguous")
    columns = [value.strip() for value in rows[0].split(",")]
    if len(columns) != 2 or not columns[0].startswith("GPU-"):
        raise StochasticBootstrapGenerationError("invalid physical GPU0 identity row")
    return {"uuid": columns[0], "pci_bus_id": columns[1].lower()}


def _normalise_gpu_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.hex()
    compact = re.sub(r"[^0-9a-f]", "", str(value).casefold())
    if len(compact) != 32:
        raise StochasticBootstrapGenerationError(
            f"logical CUDA device UUID is unavailable/invalid: {value!r}"
        )
    return compact


def _verify_logical_cuda0_is_physical_gpu0(
    torch_module: Any, physical_identity: Mapping[str, str]
) -> str:
    properties = torch_module.cuda.get_device_properties(0)
    logical_uuid = getattr(properties, "uuid", None)
    expected_uuid = str(physical_identity["uuid"])
    if _normalise_gpu_uuid(logical_uuid) != _normalise_gpu_uuid(expected_uuid):
        raise StochasticBootstrapGenerationError(
            "logical cuda:0 UUID does not match nvidia-smi physical GPU0"
        )
    return expected_uuid


@contextlib.contextmanager
def exclusive_gpu0_lease(
    *,
    lock_path: Path = GPU_LOCK_PATH,
    timeout_seconds: int,
    poll_seconds: int,
    on_wait: Callable[[str, Sequence[Mapping[str, Any]]], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Wait for both the cooperative lock and an idle physical GPU0."""

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise StochasticBootstrapGenerationError("GPU wait timeout/poll must be positive")
    lock_path = lock_path.expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise StochasticBootstrapGenerationError(f"GPU lock path is a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    started = time.monotonic()
    last_notice = 0.0
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise StochasticBootstrapGenerationError(
                        "timed out waiting for the exclusive GPU0 generation lock"
                    )
                if on_wait is not None and elapsed - last_notice >= 60:
                    on_wait("waiting_for_flock", [])
                    last_notice = elapsed
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        consecutive_idle = 0
        while consecutive_idle < 2:
            processes = _external_gpu0_compute_processes()
            elapsed = time.monotonic() - started
            if processes:
                consecutive_idle = 0
                if on_wait is not None and elapsed - last_notice >= 60:
                    on_wait("waiting_for_external_compute_pids", processes)
                    last_notice = elapsed
            else:
                consecutive_idle += 1
            if consecutive_idle < 2:
                if elapsed >= timeout_seconds:
                    raise StochasticBootstrapGenerationError(
                        "timed out waiting for physical GPU0 to become idle"
                    )
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        lease = {
            "lock_path": str(lock_path),
            "acquired_at_utc": _utc_now(),
            "wait_seconds": round(time.monotonic() - started, 3),
            "physical_gpu_index": 0,
            "external_compute_pids_at_acquire": [],
        }
        yield lease
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _completion_matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    matrix: dict[str, list[int]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        matrix.setdefault(sample_id, []).append(int(row["replicate_id"]))
    return [
        {"sample_id": sample_id, "replicate_ids": replicate_ids}
        for sample_id, replicate_ids in matrix.items()
    ]


def _state_payload(
    *,
    status: str,
    model_id: str,
    results: Sequence[Mapping[str, Any]],
    expected_cases: int,
    resume_count: int,
    progress_path: Path,
    **extra: Any,
) -> dict[str, Any]:
    binding = _file_binding(progress_path) if progress_path.is_file() else None
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": status,
        "updated_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "completed_cases": len(results),
        "expected_cases": expected_cases,
        "resume_count": resume_count,
        "completion_matrix": _completion_matrix(results),
        "partial_results": binding,
        **extra,
    }


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(path, seal_manifest(payload))


def _validate_state_integrity(state: Mapping[str, Any]) -> str:
    try:
        return validate_manifest_integrity(state)
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"state integrity failed: {exc}"
        ) from exc


def _validate_resume_state(
    *,
    state: Mapping[str, Any],
    model_id: str,
    rows: Sequence[Mapping[str, Any]],
    progress_path: Path,
    expected_cases: int,
    canonical_partial_path: Path | None = None,
) -> int:
    _validate_state_integrity(state)
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("evaluation_id") != EVALUATION_ID
        or state.get("model_id") != model_id
        or state.get("expected_cases") != expected_cases
    ):
        raise StochasticBootstrapGenerationError("resume state contract drift")
    recorded = state.get("completed_cases")
    if not isinstance(recorded, int) or isinstance(recorded, bool) or recorded < 0:
        raise StochasticBootstrapGenerationError("resume completed-case count is invalid")
    if len(rows) not in {recorded, recorded + 1}:
        raise StochasticBootstrapGenerationError(
            "partial/state mismatch exceeds the one-row fsync-before-state window"
        )
    binding = state.get("partial_results")
    if not isinstance(binding, Mapping):
        raise StochasticBootstrapGenerationError("resume state has no partial binding")
    recorded_sha = binding.get("sha256")
    if not isinstance(recorded_sha, str) or SHA256_RE.fullmatch(recorded_sha) is None:
        raise StochasticBootstrapGenerationError("resume partial SHA is invalid")
    bound_path = Path(str(binding.get("path"))).expanduser().resolve()
    actual_path = progress_path.expanduser().resolve()
    path_matches = bound_path == actual_path
    rename_before_state_window = (
        canonical_partial_path is not None
        and bound_path == canonical_partial_path.expanduser().resolve()
        and actual_path.name == "generations.jsonl"
        and len(rows) == expected_cases
        and recorded == expected_cases
    )
    if not path_matches and not rename_before_state_window:
        raise StochasticBootstrapGenerationError("resume partial path binding drift")
    recorded_bytes = binding.get("bytes")
    if not isinstance(recorded_bytes, int) or isinstance(recorded_bytes, bool):
        raise StochasticBootstrapGenerationError("resume partial byte count is invalid")
    observed_sha = _sha256_file(progress_path)
    if len(rows) == recorded:
        if observed_sha != recorded_sha or progress_path.stat().st_size != recorded_bytes:
            raise StochasticBootstrapGenerationError("resume partial SHA/size mismatch")
    elif (
        _jsonl_rows_sha256(rows[:recorded]) != recorded_sha
        or _jsonl_rows_bytes(rows[:recorded]) != recorded_bytes
    ):
        raise StochasticBootstrapGenerationError(
            "fsynced one-row-ahead partial does not extend the state-bound prefix"
        )
    resume_count = state.get("resume_count", 0)
    if not isinstance(resume_count, int) or isinstance(resume_count, bool) or resume_count < 0:
        raise StochasticBootstrapGenerationError("resume count is invalid")
    return resume_count + 1


def _generation_contract(
    rows: Sequence[Mapping[str, Any]], *, smoke: bool
) -> dict[str, Any]:
    cases = [
        {
            "sample_id": row.get("sample_id"),
            "meeting_id": row.get("meeting_id"),
            "replicate_id": row.get("replicate_id"),
            "replicate_seed": row.get("replicate_seed"),
            "seed": row.get("seed"),
        }
        for row in rows
    ]
    expected_cases = 1 if smoke else EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS)
    if len(cases) != expected_cases:
        raise StochasticBootstrapGenerationError("generation contract row count drift")
    common_keys = ("eos_token_ids", "pad_token_id")
    common = {key: rows[0].get(key) for key in common_keys}
    for row in rows:
        for key, expected in common.items():
            if row.get(key) != expected:
                raise StochasticBootstrapGenerationError(
                    f"within-run generation contract drift at {key}"
                )
    return {
        **_sampling_contract(),
        **common,
        "evaluation_scope": "infrastructure_smoke" if smoke else "formal_full_test",
        "expected_samples": 1 if smoke else EXPECTED_TEST_ROWS,
        "expected_cases": expected_cases,
        "cases": cases,
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, smoke: bool) -> dict[str, Any]:
    expected = 1 if smoke else EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS)
    if len(rows) != expected:
        raise StochasticBootstrapGenerationError("cannot summarize incomplete run")
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(row["preregistered_core_failures"])
    return {
        "status": "complete",
        "evaluation_scope": "infrastructure_smoke" if smoke else "formal_full_test",
        "cases": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_meetings": len({row["meeting_id"] for row in rows}),
        "replicates": len({row["replicate_id"] for row in rows}),
        "input_truncation_cases": sum(bool(row["input_truncated"]) for row in rows),
        "core_valid_cases": sum(bool(row["preregistered_core_valid"]) for row in rows),
        "generation_metric_rates": {
            metric: sum(bool(row["generation_metrics"][metric]) for row in rows)
            / len(rows)
            for metric in (
                "structure_delivery",
                "numeric_fidelity",
                "date_fidelity",
                "degeneration_free",
            )
        },
        "core_failure_counts": dict(sorted(failures.items())),
    }


def reset_row_rng(transformers_module: Any, torch_module: Any, row_seed: int) -> None:
    """Reset every Transformers/PyTorch RNG immediately before one tuple."""

    if isinstance(row_seed, bool) or not isinstance(row_seed, int) or row_seed < 0:
        raise StochasticBootstrapGenerationError("row seed must be non-negative")
    transformers_module.set_seed(row_seed)
    torch_module.manual_seed(row_seed)
    torch_module.cuda.manual_seed_all(row_seed)


def run_model(
    *,
    model_id: str,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    resume: bool,
    smoke: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    gpu_lock_path: Path = GPU_LOCK_PATH,
) -> dict[str, Any]:
    """Run one anchor-bound model; formal callers use :func:`run_suite`."""

    if model_id not in MODEL_ORDER:
        raise StochasticBootstrapGenerationError(f"unsupported model ID: {model_id}")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if visible != ["0"]:
        raise StochasticBootstrapGenerationError(
            "CUDA_VISIBLE_DEVICES must expose physical GPU0 only; expected ['0'], "
            f"observed={visible}"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise StochasticBootstrapGenerationError(
            "CUDA_DEVICE_ORDER must be exactly PCI_BUS_ID for physical GPU binding"
        )
    physical_gpu_identity = _physical_gpu0_identity()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.is_symlink():
        raise StochasticBootstrapGenerationError(f"output directory is a symlink: {output_dir}")
    if output_dir.exists() and not resume:
        raise StochasticBootstrapGenerationError(f"output already exists: {output_dir}")
    if not output_dir.exists() and resume:
        raise StochasticBootstrapGenerationError(f"resume output does not exist: {output_dir}")
    manifest, observed_sample_sha = load_full_test_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    bound_rows = _load_bound_rows(manifest)
    source_hashes = _source_hashes(manifest, observed_sample_sha)
    anchor = load_frozen_anchor(model_id)
    model_label = str(anchor["model_label"])
    cases = _canonical_cases(manifest["samples"], smoke=smoke)
    expected_cases = len(cases)
    launch_path = output_dir / "launch.json"
    partial_dir = output_dir / ".partial"
    progress_path = partial_dir / "generations.progress.v1.jsonl"
    final_path = output_dir / "generations.jsonl"
    state_path = output_dir / "state.progress.v1.json"
    run_manifest_path = output_dir / "manifest.json"
    launch_payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "task_contract_id": TASK_CONTRACT_ID,
        "evaluation_scope": "infrastructure_smoke" if smoke else "formal_full_test",
        "model_id": model_id,
        "stage_id": model_id,
        "model_label": model_label,
        "model": anchor["model"],
        "adapter": None,
        "sealed_anchor_manifest": anchor["anchor"],
        "sample_manifest": {
            "path": str(sample_manifest_path.expanduser().resolve()),
            "sha256": observed_sample_sha,
        },
        "source_artifact_sha256s": source_hashes,
        "generation": {
            **_sampling_contract(),
            "expected_samples": 1 if smoke else EXPECTED_TEST_ROWS,
            "expected_cases": expected_cases,
        },
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": visible,
            "physical_gpu_index": 0,
            "gpu_lock_path": str(gpu_lock_path.expanduser().resolve()),
            "gpu_busy_policy": "wait-for-zero-external-compute-pids-no-override-v1",
            "cuda_device_order": "PCI_BUS_ID",
            "physical_gpu_identity": physical_gpu_identity,
        },
        "persistence": {
            "append_flush_fsync_per_row": True,
            "full_generated_text": True,
            "answer": True,
            "generated_token_ids": True,
            "source_text_and_all_hashes": True,
            "completion_matrix_state": True,
            "semantic_metrics": "pending_offline",
        },
    }
    launch_compare_keys = (
        "schema_version",
        "evaluation_id",
        "task_contract_id",
        "evaluation_scope",
        "model_id",
        "stage_id",
        "model_label",
        "model",
        "adapter",
        "sealed_anchor_manifest",
        "sample_manifest",
        "source_artifact_sha256s",
        "generation",
        "runtime",
        "persistence",
    )
    results: list[dict[str, Any]] = []
    resume_count = 0
    state_frozen = False
    finalization_recovery = False
    active_results_path = progress_path
    if resume:
        if run_manifest_path.exists() or run_manifest_path.is_symlink():
            raise StochasticBootstrapGenerationError("refusing to resume a sealed model run")
        if not launch_path.is_file() or launch_path.is_symlink():
            raise StochasticBootstrapGenerationError("resume launch manifest is missing")
        prior_launch = _read_json(launch_path)
        try:
            validate_manifest_integrity(prior_launch)
        except Exception as exc:
            raise StochasticBootstrapGenerationError(
                f"resume launch integrity failed: {exc}"
            ) from exc
        for key in launch_compare_keys:
            if prior_launch.get(key) != launch_payload.get(key):
                raise StochasticBootstrapGenerationError(
                    f"resume launch contract drift at {key}"
                )
        if progress_path.is_file() and not progress_path.is_symlink():
            active_results_path = progress_path
        elif final_path.is_file() and not final_path.is_symlink():
            active_results_path = final_path
        else:
            raise StochasticBootstrapGenerationError("resume generation prefix is missing")
        if progress_path.is_file() and final_path.exists():
            raise StochasticBootstrapGenerationError("partial and final generation files coexist")
        results = _read_jsonl(active_results_path)
        validate_progress_prefix(
            results,
            model_id=model_id,
            model_label=model_label,
            sample_manifest_sha256=observed_sample_sha,
            samples=manifest["samples"],
            bound_rows=bound_rows,
            source_artifact_sha256s=source_hashes,
            smoke=smoke,
        )
        if not state_path.is_file() or state_path.is_symlink():
            raise StochasticBootstrapGenerationError("resume state is missing")
        prior_state = _read_json(state_path)
        resume_count = _validate_resume_state(
            state=prior_state,
            model_id=model_id,
            rows=results,
            progress_path=active_results_path,
            expected_cases=expected_cases,
            canonical_partial_path=progress_path,
        )
        if active_results_path == final_path and len(results) != expected_cases:
            raise StochasticBootstrapGenerationError("incomplete rows were renamed final")
        state_frozen = prior_state.get("status") == "generation_complete"
        if state_frozen and active_results_path != final_path:
            raise StochasticBootstrapGenerationError(
                "generation-complete state does not bind the final JSONL"
            )
        if state_frozen:
            finalization_recovery = True
            resume_count = int(prior_state.get("resume_count", 0))
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        partial_dir.mkdir()
        _fsync_directory(output_dir.parent)
        _fsync_directory(output_dir)
        _write_new_json(launch_path, seal_manifest(launch_payload))
        with progress_path.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial_dir)
        _write_state(
            state_path,
            _state_payload(
                status="initializing",
                model_id=model_id,
                results=results,
                expected_cases=expected_cases,
                resume_count=resume_count,
                progress_path=progress_path,
            ),
        )

    prompt_contract = manifest["prompt_contract"]
    config_path = Path(str(prompt_contract["training_config"])).expanduser().resolve()
    try:
        system_prompt, suffix, config_sha = native_eval.native_probe._load_prompt_contract(
            config_path
        )
    except native_eval.native_probe.Chk3ProbeError as exc:
        raise StochasticBootstrapGenerationError(str(exc)) from exc
    if config_sha != prompt_contract["training_config_sha256"]:
        raise StochasticBootstrapGenerationError("training prompt config drift")

    def on_gpu_wait(reason: str, processes: Sequence[Mapping[str, Any]]) -> None:
        target = active_results_path if active_results_path.is_file() else progress_path
        _write_state(
            state_path,
            _state_payload(
                status=reason,
                model_id=model_id,
                results=results,
                expected_cases=expected_cases,
                resume_count=resume_count,
                progress_path=target,
                external_compute_processes=[dict(value) for value in processes],
            ),
        )
        print(
            _canonical_json(
                {
                    "status": reason,
                    "model_id": model_id,
                    "external_compute_processes": list(processes),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    model = None
    tokenizer = None
    lease: dict[str, Any] | None = None
    try:
        with exclusive_gpu0_lease(
            lock_path=gpu_lock_path,
            timeout_seconds=gpu_wait_timeout_seconds,
            poll_seconds=gpu_poll_seconds,
            on_wait=on_gpu_wait,
        ) as acquired_lease:
            lease = acquired_lease
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise StochasticBootstrapGenerationError(
                    "exactly one visible CUDA GPU (physical GPU0) is required"
                )
            logical_gpu_uuid = _verify_logical_cuda0_is_physical_gpu0(
                torch, physical_gpu_identity
            )
            tokenizer_path = Path(str(manifest["tokenizer"]["path"])).expanduser().resolve()
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True, trust_remote_code=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer_semantics = native_eval._tokenizer_semantics(tokenizer)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(anchor["model_path"]),
                dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
                quantization_config=quantization_config,
                device_map={"": 0},
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model.eval()
            model.config.use_cache = True
            if int(getattr(model.config, "vocab_size", 0) or 0) != int(
                tokenizer_semantics["vocabulary_size"]
            ):
                raise StochasticBootstrapGenerationError(
                    "model vocabulary is incompatible with the frozen tokenizer"
                )
            eos_value = model.generation_config.eos_token_id
            if eos_value is None:
                eos_value = tokenizer.eos_token_id
            try:
                eos_ids = native_eval.native_probe.common_probe._normalize_eos_ids(
                    eos_value
                )
            except native_eval.native_probe.common_probe.ProbeError as exc:
                raise StochasticBootstrapGenerationError(str(exc)) from exc
            if not eos_ids:
                raise StochasticBootstrapGenerationError("model/tokenizer has no EOS IDs")
            validate_progress_prefix(
                results,
                model_id=model_id,
                model_label=model_label,
                sample_manifest_sha256=observed_sample_sha,
                samples=manifest["samples"],
                bound_rows=bound_rows,
                source_artifact_sha256s=source_hashes,
                smoke=smoke,
                tokenizer=tokenizer,
            )
            for result in results:
                if result.get("eos_token_ids") != sorted(eos_ids) or result.get(
                    "pad_token_id"
                ) != int(tokenizer.pad_token_id):
                    raise StochasticBootstrapGenerationError(
                        "resume tokenizer stop contract drift"
                    )
            torch.cuda.reset_peak_memory_stats()
            remaining = cases[len(results) :]
            if remaining and active_results_path == final_path:
                raise StochasticBootstrapGenerationError(
                    "cannot append to a prematurely finalized generation file"
                )
            if remaining:
                with progress_path.open("a", encoding="utf-8") as output_handle:
                    for sample, replicate_id, replicate_seed in remaining:
                        sample_id = str(sample["sample_id"])
                        source_row = bound_rows[sample_id]
                        try:
                            prompt_ids = native_eval.native_probe._prompt_ids(
                                tokenizer,
                                native_eval.native_probe._messages(
                                    source_row,
                                    system_prompt=system_prompt,
                                    user_prompt_suffix=suffix,
                                ),
                            )
                        except native_eval.native_probe.Chk3ProbeError as exc:
                            raise StochasticBootstrapGenerationError(str(exc)) from exc
                        if len(prompt_ids) != sample.get("prompt_token_count"):
                            raise StochasticBootstrapGenerationError(
                                f"prompt token-count drift for {sample_id}"
                            )
                        context_limit = int(
                            getattr(model.config, "max_position_embeddings", 0) or 0
                        )
                        if context_limit and len(prompt_ids) + MAX_NEW_TOKENS > context_limit:
                            raise StochasticBootstrapGenerationError(
                                f"input would truncate/overflow for {sample_id}: "
                                f"prompt={len(prompt_ids)}, completion={MAX_NEW_TOKENS}, "
                                f"limit={context_limit}"
                            )
                        input_ids = torch.tensor(
                            [prompt_ids], dtype=torch.long, device="cuda:0"
                        )
                        attention_mask = torch.ones_like(input_ids)
                        external_processes = _external_gpu0_compute_processes()
                        if external_processes:
                            _write_state(
                                state_path,
                                _state_payload(
                                    status="external_process_detected_fail_closed",
                                    model_id=model_id,
                                    results=results,
                                    expected_cases=expected_cases,
                                    resume_count=resume_count,
                                    progress_path=progress_path,
                                    external_compute_processes=external_processes,
                                ),
                            )
                            raise ExternalGpuProcessDetectedError(
                                "external GPU0 compute process appeared after model load; "
                                f"unloading without sharing: {external_processes}"
                            )
                        row_seed = derive_row_seed(replicate_seed, sample_id)
                        reset_row_rng(transformers, torch, row_seed)
                        with torch.inference_mode():
                            sequences = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                do_sample=True,
                                temperature=TEMPERATURE,
                                top_p=TOP_P,
                                top_k=TOP_K,
                                repetition_penalty=REPETITION_PENALTY,
                                max_new_tokens=MAX_NEW_TOKENS,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=sorted(eos_ids),
                                use_cache=True,
                            )
                        generated_ids = sequences[0, input_ids.shape[1] :].tolist()
                        try:
                            text = native_eval.native_probe.common_probe.decode_completion_preserving_boundary(
                                tokenizer, generated_ids, eos_ids
                            )
                        except native_eval.native_probe.common_probe.ProbeError as exc:
                            raise StochasticBootstrapGenerationError(str(exc)) from exc
                        source_analysis = native_eval.native_probe.extract_source_analysis(
                            str(source_row["prompt"])
                        )
                        result = build_stochastic_result(
                            text=text,
                            generated_token_ids=generated_ids,
                            eos_token_ids=eos_ids,
                            source_prompt=str(source_row["prompt"]),
                            source_analysis=source_analysis,
                            reference_response=str(source_row["response"]),
                            pad_token_id=int(tokenizer.pad_token_id),
                            model_id=model_id,
                            model_label=model_label,
                            sample_manifest_sha256=observed_sample_sha,
                            sample=sample,
                            replicate_id=replicate_id,
                            replicate_seed=replicate_seed,
                            source_artifact_sha256s=source_hashes,
                        )
                        validate_stochastic_result(
                            result,
                            model_id=model_id,
                            model_label=model_label,
                            sample_manifest_sha256=observed_sample_sha,
                            sample=sample,
                            source_analysis=source_analysis,
                            reference_response=str(source_row["response"]),
                            replicate_id=replicate_id,
                            replicate_seed=replicate_seed,
                            source_artifact_sha256s=source_hashes,
                            tokenizer=tokenizer,
                        )
                        output_handle.write(_canonical_json(result) + "\n")
                        output_handle.flush()
                        os.fsync(output_handle.fileno())
                        results.append(result)
                        _write_state(
                            state_path,
                            _state_payload(
                                status="generating",
                                model_id=model_id,
                                results=results,
                                expected_cases=expected_cases,
                                resume_count=resume_count,
                                progress_path=progress_path,
                                last_tuple={
                                    "model_id": model_id,
                                    "sample_id": sample_id,
                                    "replicate_id": replicate_id,
                                },
                            ),
                        )
                        print(
                            _canonical_json(
                                {
                                    "status": "generating",
                                    "model_id": model_id,
                                    "completed_cases": len(results),
                                    "expected_cases": expected_cases,
                                    "last_tuple": {
                                        "sample_id": sample_id,
                                        "replicate_id": replicate_id,
                                    },
                                    "state_path": str(state_path),
                                    "partial_results_sha256": _sha256_file(progress_path),
                                }
                            ),
                            flush=True,
                        )
                        del input_ids, attention_mask, sequences
            validate_progress_prefix(
                results,
                model_id=model_id,
                model_label=model_label,
                sample_manifest_sha256=observed_sample_sha,
                samples=manifest["samples"],
                bound_rows=bound_rows,
                source_artifact_sha256s=source_hashes,
                smoke=smoke,
                tokenizer=tokenizer,
                complete=True,
            )
            peak_allocated_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            peak_reserved_gib = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            runtime_versions = {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }

        if active_results_path != final_path:
            os.replace(progress_path, final_path)
            _fsync_directory(output_dir)
        summary = _summary(results, smoke=smoke)
        contract = _generation_contract(results, smoke=smoke)
        finalization = {
            "generation_contract_sha256": _sha256_text(_canonical_json(contract)),
            "summary_sha256": _sha256_text(_canonical_json(summary)),
            "tokenizer_semantics": tokenizer_semantics,
            "generation_runtime": {
                "physical_gpu_index": 0,
                "cuda_visible_devices": visible,
                "gpu_lease": lease,
                "physical_gpu_identity": physical_gpu_identity,
                "logical_cuda0_uuid": logical_gpu_uuid,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                **runtime_versions,
            },
        }
        if not state_frozen:
            _write_state(
                state_path,
                _state_payload(
                    status="generation_complete",
                    model_id=model_id,
                    results=results,
                    expected_cases=expected_cases,
                    resume_count=resume_count,
                    progress_path=final_path,
                    finalization=finalization,
                ),
            )
            state_frozen = True
        frozen_state = _read_json(state_path)
        frozen_state_payload_sha = _validate_state_integrity(frozen_state)
        if frozen_state.get("status") != "generation_complete":
            raise StochasticBootstrapGenerationError(
                "final state is not frozen at generation_complete"
            )
        frozen_finalization = frozen_state.get("finalization")
        if not isinstance(frozen_finalization, Mapping) or (
            frozen_finalization.get("generation_contract_sha256")
            != finalization["generation_contract_sha256"]
            or frozen_finalization.get("summary_sha256")
            != finalization["summary_sha256"]
        ):
            raise StochasticBootstrapGenerationError(
                "frozen generation state finalization drift"
            )
        run_payload = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "task_contract_id": TASK_CONTRACT_ID,
            "evaluation_scope": "infrastructure_smoke" if smoke else "formal_full_test",
            "model_id": model_id,
            "stage_id": model_id,
            "model_label": model_label,
            "model": anchor["model"],
            "adapter": None,
            "sealed_anchor_manifest": anchor["anchor"],
            "sample_manifest": {
                "path": str(sample_manifest_path.expanduser().resolve()),
                "sha256": observed_sample_sha,
            },
            "source_artifact_sha256s": source_hashes,
            "prompt_contract": dict(prompt_contract),
            "tokenizer_semantics": tokenizer_semantics,
            "generation_contract": contract,
            "runtime": {
                "physical_gpu_index": 0,
                "cuda_visible_devices": visible,
                "gpu_lease": lease,
                "physical_gpu_identity": physical_gpu_identity,
                "logical_cuda0_uuid": logical_gpu_uuid,
                "resume_count": resume_count,
                "finalization_recovery": finalization_recovery,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
                **runtime_versions,
            },
            "persistence": launch_payload["persistence"],
            "artifacts": {
                "launch": _file_binding(
                    launch_path,
                    payload_sha256=validate_manifest_integrity(_read_json(launch_path)),
                ),
                "generations": {**_file_binding(final_path), "rows": len(results)},
                "generation_complete_state": _file_binding(
                    state_path, payload_sha256=frozen_state_payload_sha
                ),
            },
            "summary": summary,
            "row_bindings": [
                {
                    "model_id": row["model_id"],
                    "sample_id": row["sample_id"],
                    "replicate_id": row["replicate_id"],
                    "row_seed": row["row_seed"],
                    "completion_sha256": row["completion_sha256"],
                    "answer_sha256": row["answer_sha256"],
                    "generated_token_ids_sha256": row["generated_token_ids_sha256"],
                }
                for row in results
            ],
            "limitations": {
                "semantic_metrics_pending_offline": True,
                "smoke_not_formal_evidence": smoke,
                "checkpoint_318_post_selection_robustness": not smoke,
            },
        }
        sealed_run = seal_manifest(run_payload)
        _write_new_json(run_manifest_path, sealed_run)
        return sealed_run
    except Exception as exc:
        target = final_path if final_path.is_file() else progress_path
        if state_path.parent.exists() and target.is_file() and not state_frozen:
            failure_status = (
                "external_process_detected_fail_closed"
                if isinstance(exc, ExternalGpuProcessDetectedError)
                else "failed"
            )
            _write_state(
                state_path,
                _state_payload(
                    status=failure_status,
                    model_id=model_id,
                    results=results,
                    expected_cases=expected_cases,
                    resume_count=resume_count,
                    progress_path=target,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    failed_at_utc=_utc_now(),
                ),
            )
        raise
    finally:
        del model, tokenizer
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
    expected_model_id: str,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    expected_scope: str = "formal_full_test",
) -> dict[str, Any]:
    """CPU-deep-validate a sealed run and return rows for offline scoring."""

    if expected_model_id not in MODEL_ORDER:
        raise StochasticBootstrapGenerationError("invalid expected model ID")
    if expected_scope not in {"formal_full_test", "infrastructure_smoke"}:
        raise StochasticBootstrapGenerationError("invalid expected evaluation scope")
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    try:
        payload_sha = validate_manifest_integrity(manifest)
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"sealed run integrity failed: {exc}"
        ) from exc
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("task_contract_id") != TASK_CONTRACT_ID
        or manifest.get("evaluation_scope") != expected_scope
        or manifest.get("model_id") != expected_model_id
        or manifest.get("stage_id") != expected_model_id
        or manifest.get("model_label") != MODEL_LABELS[expected_model_id]
        or manifest.get("adapter") is not None
    ):
        raise StochasticBootstrapGenerationError("sealed run identity/contract drift")
    anchor = load_frozen_anchor(expected_model_id)
    if (
        manifest.get("model") != anchor["model"]
        or manifest.get("sealed_anchor_manifest") != anchor["anchor"]
    ):
        raise StochasticBootstrapGenerationError("sealed run anchor/model drift")
    sample_binding = manifest.get("sample_manifest")
    if not isinstance(sample_binding, Mapping) or (
        Path(str(sample_binding.get("path"))).expanduser().resolve()
        != sample_manifest_path.expanduser().resolve()
        or sample_binding.get("sha256") != sample_manifest_sha256
    ):
        raise StochasticBootstrapGenerationError("sealed run sample binding drift")
    sample_manifest, observed_sample_sha = load_full_test_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    source_hashes = _source_hashes(sample_manifest, observed_sample_sha)
    if manifest.get("source_artifact_sha256s") != source_hashes:
        raise StochasticBootstrapGenerationError("sealed run source-hash drift")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("physical_gpu_index") != 0
        or runtime.get("cuda_visible_devices") != ["0"]
    ):
        raise StochasticBootstrapGenerationError("run was not executed on GPU0 only")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StochasticBootstrapGenerationError("sealed run artifact bindings are missing")
    launch_binding = artifacts.get("launch")
    if not isinstance(launch_binding, Mapping):
        raise StochasticBootstrapGenerationError("launch binding is missing")
    launch_path = Path(str(launch_binding.get("path"))).expanduser().resolve()
    if launch_path.parent != manifest_path.parent or launch_path.name != "launch.json":
        raise StochasticBootstrapGenerationError("launch is outside the run directory")
    launch = _read_json(launch_path)
    try:
        launch_payload_sha = validate_manifest_integrity(launch)
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"launch integrity failed: {exc}"
        ) from exc
    if dict(launch_binding) != _file_binding(
        launch_path, payload_sha256=launch_payload_sha
    ):
        raise StochasticBootstrapGenerationError("launch file binding drift")
    launch_to_run = {
        "evaluation_id": "evaluation_id",
        "task_contract_id": "task_contract_id",
        "evaluation_scope": "evaluation_scope",
        "model_id": "model_id",
        "stage_id": "stage_id",
        "model_label": "model_label",
        "model": "model",
        "adapter": "adapter",
        "sealed_anchor_manifest": "sealed_anchor_manifest",
        "sample_manifest": "sample_manifest",
        "source_artifact_sha256s": "source_artifact_sha256s",
        "persistence": "persistence",
    }
    for launch_key, run_key in launch_to_run.items():
        if launch.get(launch_key) != manifest.get(run_key):
            raise StochasticBootstrapGenerationError(
                f"launch/run contract drift at {launch_key}"
            )
    launch_generation = launch.get("generation")
    if not isinstance(launch_generation, Mapping):
        raise StochasticBootstrapGenerationError("launch generation contract is missing")
    expected_launch_generation = {
        **_sampling_contract(),
        "expected_samples": 1 if expected_scope == "infrastructure_smoke" else EXPECTED_TEST_ROWS,
        "expected_cases": 1
        if expected_scope == "infrastructure_smoke"
        else EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS),
    }
    if dict(launch_generation) != expected_launch_generation:
        raise StochasticBootstrapGenerationError("launch generation contract drift")
    launch_runtime = launch.get("runtime")
    if (
        not isinstance(launch_runtime, Mapping)
        or launch_runtime.get("cuda_visible_devices") != ["0"]
        or launch_runtime.get("physical_gpu_index") != 0
        or launch_runtime.get("cuda_device_order") != "PCI_BUS_ID"
        or not isinstance(launch_runtime.get("physical_gpu_identity"), Mapping)
    ):
        raise StochasticBootstrapGenerationError("launch physical-GPU contract drift")
    generations = artifacts.get("generations")
    if not isinstance(generations, Mapping):
        raise StochasticBootstrapGenerationError("generation binding is missing")
    generations_path = Path(str(generations.get("path"))).expanduser().resolve()
    if (
        generations_path.parent != manifest_path.parent
        or generations_path.name != "generations.jsonl"
    ):
        raise StochasticBootstrapGenerationError("generations are outside the run directory")
    observed_bytes = generations_path.stat().st_size
    if (
        _sha256_file(generations_path) != generations.get("sha256")
        or generations.get("bytes") != observed_bytes
    ):
        raise StochasticBootstrapGenerationError("generation file hash/size drift")
    rows = _read_jsonl(generations_path)
    smoke = expected_scope == "infrastructure_smoke"
    expected_cases = 1 if smoke else EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS)
    if len(rows) != expected_cases or generations.get("rows") != expected_cases:
        raise StochasticBootstrapGenerationError("sealed generation row count drift")
    from transformers import AutoTokenizer

    tokenizer_path = Path(str(sample_manifest["tokenizer"]["path"])).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bound_rows = _load_bound_rows(sample_manifest)
    validate_progress_prefix(
        rows,
        model_id=expected_model_id,
        model_label=MODEL_LABELS[expected_model_id],
        sample_manifest_sha256=observed_sample_sha,
        samples=sample_manifest["samples"],
        bound_rows=bound_rows,
        source_artifact_sha256s=source_hashes,
        smoke=smoke,
        tokenizer=tokenizer,
        complete=True,
    )
    observed_contract = _generation_contract(rows, smoke=smoke)
    if manifest.get("generation_contract") != observed_contract:
        raise StochasticBootstrapGenerationError("sealed generation contract drift")
    observed_summary = _summary(rows, smoke=smoke)
    if manifest.get("summary") != observed_summary:
        raise StochasticBootstrapGenerationError("sealed run summary drift")
    row_bindings = [
        {
            "model_id": row["model_id"],
            "sample_id": row["sample_id"],
            "replicate_id": row["replicate_id"],
            "row_seed": row["row_seed"],
            "completion_sha256": row["completion_sha256"],
            "answer_sha256": row["answer_sha256"],
            "generated_token_ids_sha256": row["generated_token_ids_sha256"],
        }
        for row in rows
    ]
    if manifest.get("row_bindings") != row_bindings:
        raise StochasticBootstrapGenerationError("sealed row-binding inventory drift")
    if any(bool(row.get("input_truncated")) for row in rows):
        raise StochasticBootstrapGenerationError("formal run contains input truncation")
    state_binding = artifacts.get("generation_complete_state")
    if not isinstance(state_binding, Mapping):
        raise StochasticBootstrapGenerationError("generation-complete state binding missing")
    state_path = Path(str(state_binding.get("path"))).expanduser().resolve()
    if (
        state_path.parent != manifest_path.parent
        or state_path.name != "state.progress.v1.json"
    ):
        raise StochasticBootstrapGenerationError("frozen state is outside run directory")
    frozen_state = _read_json(state_path)
    state_payload_sha = _validate_state_integrity(frozen_state)
    if dict(state_binding) != _file_binding(
        state_path, payload_sha256=state_payload_sha
    ):
        raise StochasticBootstrapGenerationError("frozen state file binding drift")
    if (
        frozen_state.get("schema_version") != STATE_SCHEMA_VERSION
        or frozen_state.get("status") != "generation_complete"
        or frozen_state.get("evaluation_id") != EVALUATION_ID
        or frozen_state.get("model_id") != expected_model_id
        or frozen_state.get("completed_cases") != expected_cases
        or frozen_state.get("expected_cases") != expected_cases
        or frozen_state.get("completion_matrix") != _completion_matrix(rows)
    ):
        raise StochasticBootstrapGenerationError("frozen state contract drift")
    final_binding = {**_file_binding(generations_path), "rows": expected_cases}
    state_results_binding = frozen_state.get("partial_results")
    if not isinstance(state_results_binding, Mapping) or dict(
        state_results_binding
    ) != {key: value for key, value in final_binding.items() if key != "rows"}:
        raise StochasticBootstrapGenerationError(
            "frozen state does not exactly bind final generations"
        )
    finalization = frozen_state.get("finalization")
    if (
        not isinstance(finalization, Mapping)
        or finalization.get("generation_contract_sha256")
        != _sha256_text(_canonical_json(observed_contract))
        or finalization.get("summary_sha256")
        != _sha256_text(_canonical_json(observed_summary))
        or finalization.get("tokenizer_semantics")
        != manifest.get("tokenizer_semantics")
    ):
        raise StochasticBootstrapGenerationError("frozen state finalization drift")
    generation_runtime = finalization.get("generation_runtime")
    if (
        not isinstance(generation_runtime, Mapping)
        or generation_runtime.get("physical_gpu_index") != 0
        or generation_runtime.get("cuda_visible_devices") != ["0"]
        or generation_runtime.get("physical_gpu_identity")
        != launch_runtime.get("physical_gpu_identity")
        or generation_runtime.get("logical_cuda0_uuid")
        != launch_runtime.get("physical_gpu_identity", {}).get("uuid")
    ):
        raise StochasticBootstrapGenerationError("frozen generation GPU runtime drift")
    for runtime_record in (generation_runtime, runtime):
        lease = runtime_record.get("gpu_lease")
        if (
            not isinstance(lease, Mapping)
            or lease.get("physical_gpu_index") != 0
            or lease.get("external_compute_pids_at_acquire") != []
        ):
            raise StochasticBootstrapGenerationError("GPU lease audit drift")
    return {
        "manifest": manifest,
        "manifest_binding": _file_binding(
            manifest_path, payload_sha256=payload_sha
        ),
        "results": rows,
        "generation_contract": observed_contract,
        "summary": observed_summary,
        "model_id": expected_model_id,
        "model_label": MODEL_LABELS[expected_model_id],
    }


def run_suite(
    *,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    resume: bool,
    smoke: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    gpu_lock_path: Path = GPU_LOCK_PATH,
) -> dict[str, Any]:
    """Run and seal CHK1 -> CHK3 -> CHK0; reject any out-of-order prefix."""

    output_dir = output_dir.expanduser().resolve()
    suite_manifest_path = output_dir / "manifest.json"
    launch_path = output_dir / "launch.json"
    state_path = output_dir / "state.progress.v1.json"
    if output_dir.exists() and not resume:
        raise StochasticBootstrapGenerationError(f"suite output exists: {output_dir}")
    if not output_dir.exists() and resume:
        raise StochasticBootstrapGenerationError("suite resume output does not exist")
    if resume and suite_manifest_path.exists():
        raise StochasticBootstrapGenerationError("refusing to resume a sealed suite")
    sample_manifest, observed_sample_sha = load_full_test_sample_manifest(
        sample_manifest_path, sample_manifest_sha256
    )
    anchors = {model_id: load_frozen_anchor(model_id) for model_id in MODEL_ORDER}
    launch_payload = {
        "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "evaluation_scope": "infrastructure_smoke" if smoke else "formal_full_test",
        "task_contract_id": TASK_CONTRACT_ID,
        "model_order": list(MODEL_ORDER),
        "sample_manifest": {
            "path": str(sample_manifest_path.expanduser().resolve()),
            "sha256": observed_sample_sha,
        },
        "sealed_anchor_manifests": {
            model_id: anchors[model_id]["anchor"] for model_id in MODEL_ORDER
        },
        "generation_contract": _sampling_contract(),
        "expected_rows_per_model": 1
        if smoke
        else EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS),
        "expected_total_rows": len(MODEL_ORDER)
        if smoke
        else EXPECTED_TEST_ROWS * len(REPLICATE_SEEDS) * len(MODEL_ORDER),
        "gpu_policy": {
            "physical_gpu_index": 0,
            "cuda_visible_devices": ["0"],
            "cuda_device_order": "PCI_BUS_ID",
            "lock_path": str(gpu_lock_path.expanduser().resolve()),
            "external_compute_policy": "wait-before-load-fail-closed-after-load-v1",
        },
    }
    suite_state_frozen = False
    if resume:
        launch = _read_json(launch_path)
        try:
            validate_manifest_integrity(launch)
        except Exception as exc:
            raise StochasticBootstrapGenerationError(
                f"suite launch integrity failed: {exc}"
            ) from exc
        for key, expected in launch_payload.items():
            if key in {"created_at_utc", "status"}:
                continue
            if launch.get(key) != expected:
                raise StochasticBootstrapGenerationError(
                    f"suite resume contract drift at {key}"
                )
        if state_path.is_file():
            prior_state = _read_json(state_path)
            _validate_state_integrity(prior_state)
            suite_state_frozen = prior_state.get("status") == "generation_complete"
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _fsync_directory(output_dir.parent)
        _write_new_json(launch_path, seal_manifest(launch_payload))
        _write_state(
            state_path,
            {
                "schema_version": f"{STATE_SCHEMA_VERSION}-suite",
                "status": "initializing",
                "evaluation_id": EVALUATION_ID,
                "evaluation_scope": launch_payload["evaluation_scope"],
                "model_order": list(MODEL_ORDER),
                "completed_model_order": [],
            },
        )

    completed: list[str] = []
    run_records: dict[str, dict[str, Any]] = {}
    for model_index, model_id in enumerate(MODEL_ORDER):
        model_dir = output_dir / model_id
        model_manifest = model_dir / "manifest.json"
        later_existing = [
            later
            for later in MODEL_ORDER[model_index + 1 :]
            if (output_dir / later).exists()
        ]
        if not model_manifest.is_file() and later_existing:
            raise StochasticBootstrapGenerationError(
                f"out-of-order suite prefix before {model_id}; later outputs exist: "
                f"{later_existing}"
            )
        if model_manifest.is_file():
            validated = load_and_validate_run(
                model_manifest,
                expected_model_id=model_id,
                sample_manifest_path=sample_manifest_path,
                sample_manifest_sha256=observed_sample_sha,
                expected_scope=launch_payload["evaluation_scope"],
            )
        else:
            _run_model_with_external_retry(
                model_id=model_id,
                sample_manifest_path=sample_manifest_path,
                sample_manifest_sha256=observed_sample_sha,
                output_dir=model_dir,
                smoke=smoke,
                gpu_wait_timeout_seconds=gpu_wait_timeout_seconds,
                gpu_poll_seconds=gpu_poll_seconds,
                gpu_lock_path=gpu_lock_path,
            )
            validated = load_and_validate_run(
                model_manifest,
                expected_model_id=model_id,
                sample_manifest_path=sample_manifest_path,
                sample_manifest_sha256=observed_sample_sha,
                expected_scope=launch_payload["evaluation_scope"],
            )
        completed.append(model_id)
        run_records[model_id] = validated
        if not suite_state_frozen:
            _write_state(
                state_path,
                {
                    "schema_version": f"{STATE_SCHEMA_VERSION}-suite",
                    "status": "running",
                    "evaluation_id": EVALUATION_ID,
                    "evaluation_scope": launch_payload["evaluation_scope"],
                    "model_order": list(MODEL_ORDER),
                    "completed_model_order": list(completed),
                    "run_manifests": {
                        key: run_records[key]["manifest_binding"] for key in completed
                    },
                },
            )
    if completed != list(MODEL_ORDER):
        raise StochasticBootstrapGenerationError("suite did not complete canonical order")
    contracts = [run_records[model_id]["generation_contract"] for model_id in MODEL_ORDER]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise StochasticBootstrapGenerationError("cross-model generation contract drift")
    tuple_inventories = [
        [
            (row["sample_id"], row["replicate_id"], row["row_seed"])
            for row in run_records[model_id]["results"]
        ]
        for model_id in MODEL_ORDER
    ]
    if any(inventory != tuple_inventories[0] for inventory in tuple_inventories[1:]):
        raise StochasticBootstrapGenerationError("cross-model paired tuple drift")
    suite_state_payload = {
        "schema_version": f"{STATE_SCHEMA_VERSION}-suite",
        "status": "generation_complete",
        "evaluation_id": EVALUATION_ID,
        "evaluation_scope": launch_payload["evaluation_scope"],
        "model_order": list(MODEL_ORDER),
        "completed_model_order": list(completed),
        "run_manifests": {
            model_id: run_records[model_id]["manifest_binding"]
            for model_id in MODEL_ORDER
        },
        "total_rows": sum(
            len(run_records[model_id]["results"]) for model_id in MODEL_ORDER
        ),
    }
    prior_suite_state = _read_json(state_path)
    if prior_suite_state.get("status") != "generation_complete":
        _write_state(state_path, suite_state_payload)
    suite_state = _read_json(state_path)
    state_payload_sha = _validate_state_integrity(suite_state)
    if {key: value for key, value in suite_state.items() if key != "integrity"} != suite_state_payload:
        raise StochasticBootstrapGenerationError("frozen suite state drift")
    suite_payload = {
        "schema_version": SUITE_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "evaluation_scope": launch_payload["evaluation_scope"],
        "task_contract_id": TASK_CONTRACT_ID,
        "model_order": list(MODEL_ORDER),
        "sample_manifest": launch_payload["sample_manifest"],
        "generation_contract": contracts[0],
        "coverage": {
            "models": len(MODEL_ORDER),
            "rows_per_model": len(run_records[MODEL_ORDER[0]]["results"]),
            "total_rows": suite_state_payload["total_rows"],
            "paired_tuple_coverage_identical": True,
            "input_truncation_cases": 0,
        },
        "artifacts": {
            "launch": _file_binding(
                launch_path,
                payload_sha256=validate_manifest_integrity(_read_json(launch_path)),
            ),
            "generation_complete_state": _file_binding(
                state_path, payload_sha256=state_payload_sha
            ),
            "model_runs": {
                model_id: run_records[model_id]["manifest_binding"]
                for model_id in MODEL_ORDER
            },
        },
        "limitations": {
            "semantic_scoring_pending_offline": True,
            "checkpoint_318_post_selection_robustness": not smoke,
            "smoke_not_formal_evidence": smoke,
        },
    }
    sealed_suite = seal_manifest(suite_payload)
    _write_new_json(suite_manifest_path, sealed_suite)
    return sealed_suite


def _run_model_with_external_retry(
    *,
    model_id: str,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    smoke: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    gpu_lock_path: Path,
) -> dict[str, Any]:
    """Resume only the dedicated post-load external-PID failure condition."""

    retries = 0
    while True:
        try:
            return run_model(
                model_id=model_id,
                sample_manifest_path=sample_manifest_path,
                sample_manifest_sha256=sample_manifest_sha256,
                output_dir=output_dir,
                resume=output_dir.exists(),
                smoke=smoke,
                gpu_wait_timeout_seconds=gpu_wait_timeout_seconds,
                gpu_poll_seconds=gpu_poll_seconds,
                gpu_lock_path=gpu_lock_path,
            )
        except ExternalGpuProcessDetectedError:
            retries += 1
            print(
                _canonical_json(
                    {
                        "status": "external_process_unloaded_waiting_to_resume",
                        "model_id": model_id,
                        "retry": retries,
                        "completed_prefix_preserved": True,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )


def load_and_validate_suite(
    manifest_path: Path,
    *,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    expected_scope: str = "formal_full_test",
) -> dict[str, Any]:
    """Deep-validate the sealed ordered suite and all three child runs."""

    if expected_scope not in {"formal_full_test", "infrastructure_smoke"}:
        raise StochasticBootstrapGenerationError("invalid suite scope")
    manifest_path = manifest_path.expanduser().resolve()
    suite = _read_json(manifest_path)
    try:
        payload_sha = validate_manifest_integrity(suite)
    except Exception as exc:
        raise StochasticBootstrapGenerationError(
            f"suite integrity failed: {exc}"
        ) from exc
    if (
        suite.get("schema_version") != SUITE_MANIFEST_SCHEMA_VERSION
        or suite.get("status") != "complete"
        or suite.get("evaluation_id") != EVALUATION_ID
        or suite.get("task_contract_id") != TASK_CONTRACT_ID
        or suite.get("evaluation_scope") != expected_scope
        or suite.get("model_order") != list(MODEL_ORDER)
    ):
        raise StochasticBootstrapGenerationError("suite identity/order drift")
    sample_binding = suite.get("sample_manifest")
    if not isinstance(sample_binding, Mapping) or (
        Path(str(sample_binding.get("path"))).expanduser().resolve()
        != sample_manifest_path.expanduser().resolve()
        or sample_binding.get("sha256") != sample_manifest_sha256
    ):
        raise StochasticBootstrapGenerationError("suite sample binding drift")
    load_full_test_sample_manifest(sample_manifest_path, sample_manifest_sha256)
    artifacts = suite.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StochasticBootstrapGenerationError("suite artifact bindings missing")
    launch_binding = artifacts.get("launch")
    state_binding = artifacts.get("generation_complete_state")
    child_bindings = artifacts.get("model_runs")
    if not all(
        isinstance(value, Mapping)
        for value in (launch_binding, state_binding, child_bindings)
    ):
        raise StochasticBootstrapGenerationError("suite artifact inventory malformed")
    launch_path = Path(str(launch_binding.get("path"))).expanduser().resolve()
    state_path = Path(str(state_binding.get("path"))).expanduser().resolve()
    if (
        launch_path.parent != manifest_path.parent
        or launch_path.name != "launch.json"
        or state_path.parent != manifest_path.parent
        or state_path.name != "state.progress.v1.json"
    ):
        raise StochasticBootstrapGenerationError("suite launch/state path drift")
    launch = _read_json(launch_path)
    launch_payload_sha = validate_manifest_integrity(launch)
    if dict(launch_binding) != _file_binding(
        launch_path, payload_sha256=launch_payload_sha
    ):
        raise StochasticBootstrapGenerationError("suite launch binding drift")
    if (
        launch.get("schema_version") != SUITE_MANIFEST_SCHEMA_VERSION
        or launch.get("model_order") != list(MODEL_ORDER)
        or launch.get("evaluation_scope") != expected_scope
        or launch.get("sample_manifest") != sample_binding
    ):
        raise StochasticBootstrapGenerationError("suite launch contract drift")
    state = _read_json(state_path)
    state_payload_sha = _validate_state_integrity(state)
    if dict(state_binding) != _file_binding(
        state_path, payload_sha256=state_payload_sha
    ):
        raise StochasticBootstrapGenerationError("suite state binding drift")
    if (
        state.get("status") != "generation_complete"
        or state.get("model_order") != list(MODEL_ORDER)
        or state.get("completed_model_order") != list(MODEL_ORDER)
        or state.get("evaluation_scope") != expected_scope
    ):
        raise StochasticBootstrapGenerationError("suite frozen state order drift")
    _validate_suite_child_binding_inventory(child_bindings)
    runs: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        binding = child_bindings.get(model_id)
        if not isinstance(binding, Mapping):
            raise StochasticBootstrapGenerationError("suite child binding missing")
        child_path = Path(str(binding.get("path"))).expanduser().resolve()
        if (
            child_path != manifest_path.parent / model_id / "manifest.json"
            or dict(binding)
            != _file_binding(
                child_path,
                payload_sha256=validate_manifest_integrity(_read_json(child_path)),
            )
        ):
            raise StochasticBootstrapGenerationError(
                f"suite {model_id} manifest binding drift"
            )
        runs[model_id] = load_and_validate_run(
            child_path,
            expected_model_id=model_id,
            sample_manifest_path=sample_manifest_path,
            sample_manifest_sha256=sample_manifest_sha256,
            expected_scope=expected_scope,
        )
    expected_state_runs = {
        model_id: runs[model_id]["manifest_binding"] for model_id in MODEL_ORDER
    }
    if state.get("run_manifests") != expected_state_runs:
        raise StochasticBootstrapGenerationError("suite state child-chain drift")
    contracts = [runs[model_id]["generation_contract"] for model_id in MODEL_ORDER]
    if any(value != contracts[0] for value in contracts[1:]) or suite.get(
        "generation_contract"
    ) != contracts[0]:
        raise StochasticBootstrapGenerationError("suite cross-model contract drift")
    tuple_inventories = [
        [
            (row["sample_id"], row["replicate_id"], row["row_seed"])
            for row in runs[model_id]["results"]
        ]
        for model_id in MODEL_ORDER
    ]
    if any(value != tuple_inventories[0] for value in tuple_inventories[1:]):
        raise StochasticBootstrapGenerationError("suite paired tuple drift")
    rows_per_model = 1 if expected_scope == "infrastructure_smoke" else 950
    expected_coverage = {
        "models": 3,
        "rows_per_model": rows_per_model,
        "total_rows": rows_per_model * 3,
        "paired_tuple_coverage_identical": True,
        "input_truncation_cases": 0,
    }
    if suite.get("coverage") != expected_coverage or state.get(
        "total_rows"
    ) != expected_coverage["total_rows"]:
        raise StochasticBootstrapGenerationError("suite coverage drift")
    if any(
        row.get("input_truncated") is not False
        for model_id in MODEL_ORDER
        for row in runs[model_id]["results"]
    ):
        raise StochasticBootstrapGenerationError("suite contains input truncation")
    return {
        "manifest": suite,
        "manifest_binding": _file_binding(
            manifest_path, payload_sha256=payload_sha
        ),
        "runs": runs,
        "generation_contract": contracts[0],
        "coverage": expected_coverage,
    }


def _validate_suite_child_binding_inventory(
    child_bindings: Mapping[str, Any],
) -> None:
    """Validate object membership; ordered arrays carry execution order.

    Sealed JSON is written with ``sort_keys=True``, so JSON object key order is
    canonical lexicographic order rather than the generation order.  The
    authoritative ordering evidence is ``model_order`` together with the
    frozen state's ``completed_model_order``; both are checked separately.
    """

    if set(child_bindings) != set(MODEL_ORDER) or any(
        not isinstance(child_bindings.get(model_id), Mapping)
        for model_id in MODEL_ORDER
    ):
        raise StochasticBootstrapGenerationError(
            "suite child binding inventory drift"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--test-data", required=True, type=Path)
    prepare.add_argument("--test-row-manifest", required=True, type=Path)
    prepare.add_argument("--release-manifest", required=True, type=Path)
    prepare.add_argument("--tokenizer", required=True, type=Path)
    prepare.add_argument("--training-config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run-model")
    run.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    run.add_argument("--sample-manifest", required=True, type=Path)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--gpu-wait-timeout-seconds", type=int, default=172800)
    run.add_argument("--gpu-poll-seconds", type=int, default=30)
    run.add_argument("--gpu-lock-path", type=Path, default=GPU_LOCK_PATH)

    suite = subparsers.add_parser("run-suite")
    suite.add_argument("--sample-manifest", required=True, type=Path)
    suite.add_argument("--sample-manifest-sha256", required=True)
    suite.add_argument("--output-dir", required=True, type=Path)
    suite.add_argument("--resume", action="store_true")
    suite.add_argument("--smoke", action="store_true")
    suite.add_argument("--gpu-wait-timeout-seconds", type=int, default=172800)
    suite.add_argument("--gpu-poll-seconds", type=int, default=30)
    suite.add_argument("--gpu-lock-path", type=Path, default=GPU_LOCK_PATH)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    validate.add_argument("--sample-manifest", required=True, type=Path)
    validate.add_argument("--sample-manifest-sha256", required=True)
    validate.add_argument(
        "--scope",
        choices=("formal_full_test", "infrastructure_smoke"),
        default="formal_full_test",
    )
    validate_suite = subparsers.add_parser("validate-suite")
    validate_suite.add_argument("--manifest", required=True, type=Path)
    validate_suite.add_argument("--sample-manifest", required=True, type=Path)
    validate_suite.add_argument("--sample-manifest-sha256", required=True)
    validate_suite.add_argument(
        "--scope",
        choices=("formal_full_test", "infrastructure_smoke"),
        default="formal_full_test",
    )
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
            result = build_full_test_sample_manifest(
                test_data=args.test_data,
                test_row_manifest=args.test_row_manifest,
                release_manifest=args.release_manifest,
                tokenizer=tokenizer,
                tokenizer_path=args.tokenizer,
                training_config=args.training_config,
            )
            _write_new_json(args.output, result)
            payload = {
                "status": "prepared",
                "schema_version": result["schema_version"],
                "path": str(args.output.expanduser().resolve()),
                "sha256": _sha256_file(args.output),
                "payload_sha256": result["integrity"]["payload_sha256"],
                "rows": len(result["samples"]),
            }
        elif args.command == "run-model":
            if not args.smoke:
                raise StochasticBootstrapGenerationError(
                    "formal generation must use run-suite to enforce CHK1->CHK3->CHK0"
                )
            result = run_model(
                model_id=args.model_id,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output_dir=args.output_dir,
                resume=args.resume,
                smoke=args.smoke,
                gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
                gpu_lock_path=args.gpu_lock_path,
            )
            payload = {
                "status": result["status"],
                "schema_version": result["schema_version"],
                "model_id": result["model_id"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
        elif args.command == "run-suite":
            result = run_suite(
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                output_dir=args.output_dir,
                resume=args.resume,
                smoke=args.smoke,
                gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
                gpu_lock_path=args.gpu_lock_path,
            )
            payload = {
                "status": result["status"],
                "schema_version": result["schema_version"],
                "model_order": result["model_order"],
                "total_rows": result["coverage"]["total_rows"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
        elif args.command == "validate":
            validated = load_and_validate_run(
                args.manifest,
                expected_model_id=args.model_id,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                expected_scope=args.scope,
            )
            payload = {
                "status": "validated",
                "schema_version": validated["manifest"]["schema_version"],
                "model_id": validated["model_id"],
                "rows": len(validated["results"]),
                "manifest_sha256": validated["manifest_binding"]["sha256"],
            }
        else:
            validated_suite = load_and_validate_suite(
                args.manifest,
                sample_manifest_path=args.sample_manifest,
                sample_manifest_sha256=args.sample_manifest_sha256,
                expected_scope=args.scope,
            )
            payload = {
                "status": "validated",
                "schema_version": validated_suite["manifest"]["schema_version"],
                "model_order": list(validated_suite["runs"]),
                "total_rows": validated_suite["coverage"]["total_rows"],
                "manifest_sha256": validated_suite["manifest_binding"]["sha256"],
            }
        print(_canonical_json(payload))
        return 0
    except (StochasticBootstrapGenerationError, OSError, ValueError) as exc:
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
