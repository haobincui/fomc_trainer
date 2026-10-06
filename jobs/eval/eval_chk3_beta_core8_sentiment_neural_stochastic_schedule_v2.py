"""Neural Core8 sentiment scorer with an explicit BERT special-token contract.

This amendment leaves the sealed v1 inventory, model snapshots, and lexicon
diagnostic byte-for-byte unchanged.  It replaces a Transformers-version-
dependent tokenizer method with the frozen sequence contract
``[CLS] + up to 510 content tokens + [SEP]``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jobs.eval import (
    eval_chk3_beta_core8_sentiment_stochastic_schedule_v1 as sentiment_v1,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATH = Path(__file__).resolve()
IMPLEMENTATION_SHA256_AT_IMPORT = sha256_file(IMPLEMENTATION_PATH)
DEFAULT_OUTPUT_ROOT = sentiment_v1.DEFAULT_OUTPUT_ROOT
DEFAULT_INVENTORY = DEFAULT_OUTPUT_ROOT / "preparation/manifest.json"
DEFAULT_MODEL_ROOT = sentiment_v1.DEFAULT_MODEL_ROOT

EVALUATION_ID = "chk3-beta-core8-sentiment-neural-stochastic-schedule-v2"
WINDOW_SCHEMA = "chk3-beta-core8-sentiment-neural-window-score-row-v2"
TOPIC_SCHEMA = "chk3-beta-core8-sentiment-neural-topic-score-row-v2"
MEETING_SCHEMA = "chk3-beta-core8-sentiment-neural-meeting-score-row-v2"
BACKEND_MANIFEST_SCHEMA = "chk3-beta-core8-sentiment-neural-backend-manifest-v2"

NEURAL_BACKENDS = (sentiment_v1.DISTIL_BACKEND, sentiment_v1.FINBERT_BACKEND)
EXPECTED_SPECIAL_TOKEN_IDS = {
    sentiment_v1.DISTIL_BACKEND: {"cls_token_id": 101, "sep_token_id": 102},
    sentiment_v1.FINBERT_BACKEND: {"cls_token_id": 101, "sep_token_id": 102},
}
CONTENT_TOKENS = sentiment_v1.CONTENT_TOKENS
OVERLAP_TOKENS = sentiment_v1.OVERLAP_TOKENS
WINDOW_STEP = sentiment_v1.WINDOW_STEP


class NeuralSentimentV2Error(RuntimeError):
    """The v2 neural sentiment contract or a bound artifact is invalid."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _record_sha256(value: Mapping[str, Any]) -> str:
    return sha256_text(_canonical(dict(value)))


def _token_ids_sha256(values: Sequence[int]) -> str:
    return sha256_text(_canonical(list(values)))


def _binding(
    path: Path, *, rows: int | None = None, payload_sha256: str | None = None
) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise NeuralSentimentV2Error(f"bound path is a symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise NeuralSentimentV2Error(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _implementation_binding() -> dict[str, Any]:
    binding = _binding(IMPLEMENTATION_PATH)
    if binding["sha256"] != IMPLEMENTATION_SHA256_AT_IMPORT:
        raise NeuralSentimentV2Error(
            "implementation source changed after this process imported it"
        )
    return binding


def _validate_implementation(value: Any) -> None:
    if not isinstance(value, Mapping) or dict(value) != _implementation_binding():
        raise NeuralSentimentV2Error("v2 implementation binding drift")


@dataclass(frozen=True)
class WindowSpec:
    index: int
    token_start: int
    token_end: int
    content_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    aggregation_weight: float


def _special_token_contract(tokenizer: Any, backend_id: str) -> dict[str, Any]:
    expected = EXPECTED_SPECIAL_TOKEN_IDS.get(backend_id)
    observed = {
        "cls_token_id": getattr(tokenizer, "cls_token_id", None),
        "sep_token_id": getattr(tokenizer, "sep_token_id", None),
    }
    if (
        expected is None
        or observed != expected
        or int(tokenizer.model_max_length) != 512
        or int(tokenizer.num_special_tokens_to_add(pair=False)) != 2
        or getattr(tokenizer, "padding_side", None) != "right"
    ):
        raise NeuralSentimentV2Error("frozen BERT special-token contract drift")
    return {
        **observed,
        "sequence_formula": "[CLS]+content_tokens+[SEP]",
        "content_tokens_per_window": CONTENT_TOKENS,
        "maximum_input_tokens": 512,
        "padding_side": "right",
    }


def _window_specs(tokenizer: Any, text: str, *, backend_id: str) -> list[WindowSpec]:
    special = _special_token_contract(tokenizer, backend_id)
    content_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in content_ids
    ):
        raise NeuralSentimentV2Error("tokenizer returned non-integer content IDs")
    if not content_ids:
        return []
    starts = [0]
    while starts[-1] + CONTENT_TOKENS < len(content_ids):
        starts.append(starts[-1] + WINDOW_STEP)
    spans = [(start, min(start + CONTENT_TOKENS, len(content_ids))) for start in starts]
    coverage = [0] * len(content_ids)
    for start, end in spans:
        for token_index in range(start, end):
            coverage[token_index] += 1
    if any(value <= 0 for value in coverage):
        raise NeuralSentimentV2Error("windowing did not cover each content token")
    result: list[WindowSpec] = []
    for index, (start, end) in enumerate(spans):
        values = content_ids[start:end]
        input_ids = [special["cls_token_id"], *values, special["sep_token_id"]]
        if len(input_ids) > 512:
            raise NeuralSentimentV2Error("v2 neural window exceeds 512 tokens")
        weight = sum(1.0 / coverage[i] for i in range(start, end))
        result.append(
            WindowSpec(
                index=index,
                token_start=start,
                token_end=end,
                content_ids=tuple(values),
                input_ids=tuple(input_ids),
                aggregation_weight=weight,
            )
        )
    if not math.isclose(
        sum(spec.aggregation_weight for spec in result),
        len(content_ids),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise NeuralSentimentV2Error("coverage-corrected window weights drift")
    return result


def _topic_base(
    source: Mapping[str, Any], *, backend_id: str, construct: str
) -> dict[str, Any]:
    return {
        "schema_version": TOPIC_SCHEMA,
        "backend": backend_id,
        "construct": construct,
        **{
            key: copy.deepcopy(source[key])
            for key in (
                "text_id",
                "arm",
                "sample_id",
                "meeting_id",
                "meeting_date",
                "meeting_start_date",
                "era",
                "role",
                "topic",
                "topic_order",
                "replicate_id",
                "replicate_seed",
                "cp318_selection_exposed",
                "text_sha256",
                "empty",
            )
        },
    }


def _aggregate_meeting_rows(
    topic_rows: Sequence[Mapping[str, Any]], *, backend_id: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in topic_rows:
        grouped[
            (str(row["arm"]), str(row["meeting_id"]), row.get("replicate_id"))
        ].append(row)
    result: list[dict[str, Any]] = []
    for arm in sentiment_v1.ARM_ORDER:
        keys = sorted(
            (key for key in grouped if key[0] == arm),
            key=lambda key: (key[1], -1 if key[2] is None else int(key[2])),
        )
        for key in keys:
            topics = sorted(grouped[key], key=lambda row: int(row["topic_order"]))
            if len(topics) != sentiment_v1.TOPIC_COUNT or [
                row["topic"] for row in topics
            ] != list(sentiment_v1.data_contract.CORE_TOPICS):
                raise NeuralSentimentV2Error(f"meeting is not exact Core8: {key}")
            first = topics[0]
            for field in (
                "meeting_date",
                "meeting_start_date",
                "era",
                "role",
                "cp318_selection_exposed",
                "replicate_id",
                "replicate_seed",
            ):
                if any(row.get(field) != first.get(field) for row in topics[1:]):
                    raise NeuralSentimentV2Error(
                        f"meeting metadata drift: {key}:{field}"
                    )
            scores = [row.get("score") for row in topics]
            complete = all(
                isinstance(score, (float, int)) and not isinstance(score, bool)
                for score in scores
            )
            neutral_scores = [
                float(score)
                if isinstance(score, (float, int)) and not isinstance(score, bool)
                else 0.0
                for score in scores
            ]
            result.append(
                {
                    "schema_version": MEETING_SCHEMA,
                    "meeting_id": str(first["meeting_id"]),
                    "meeting_date": str(first["meeting_date"]),
                    "meeting_start_date": str(first["meeting_start_date"]),
                    "era": str(first["era"]),
                    "role": str(first["role"]),
                    "cp318_selection_exposed": bool(first["cp318_selection_exposed"]),
                    "backend": backend_id,
                    "construct": str(first["construct"]),
                    "arm": arm,
                    "replicate_id": first.get("replicate_id"),
                    "replicate_seed": first.get("replicate_seed"),
                    "score": (
                        sum(float(score) for score in scores) / sentiment_v1.TOPIC_COUNT
                        if complete
                        else None
                    ),
                    "complete_core8": complete,
                    "neutral_imputed_score": sum(neutral_scores)
                    / sentiment_v1.TOPIC_COUNT,
                    "missing_topics": [
                        str(row["topic"])
                        for row, score in zip(topics, scores, strict=True)
                        if score is None
                    ],
                    "source_hashes": {
                        "topic_text_sha256s": [
                            str(row["text_sha256"]) for row in topics
                        ],
                        "topic_score_row_sha256s": [
                            _record_sha256(row) for row in topics
                        ],
                    },
                }
            )
    if len(result) != sentiment_v1.EXPECTED_MEETING_ROWS:
        raise NeuralSentimentV2Error("v2 meeting score coverage drift")
    return result


def _backend_manifest(
    *,
    backend_id: str,
    output_dir: Path,
    inventory_binding: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    window_path: Path,
    window_rows: int,
    topic_path: Path,
    topic_rows: Sequence[Mapping[str, Any]],
    meeting_path: Path,
    meeting_rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    special_token_contract: Mapping[str, Any],
) -> dict[str, Any]:
    complete_counts = Counter(
        str(row["arm"]) for row in meeting_rows if bool(row["complete_core8"])
    )
    return seal_manifest(
        {
            "schema_version": BACKEND_MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": EVALUATION_ID,
            "backend": backend_id,
            "construct": str(meeting_rows[0]["construct"]),
            "inventory_manifest": copy.deepcopy(dict(inventory_binding)),
            "model_manifest": copy.deepcopy(dict(model_binding)),
            "artifacts": {
                "window_scores": _binding(window_path, rows=window_rows),
                "topic_scores": _binding(topic_path, rows=len(topic_rows)),
                "meeting_scores": _binding(meeting_path, rows=len(meeting_rows)),
            },
            "coverage": {
                "texts": len(topic_rows),
                "window_rows": window_rows,
                "topic_rows": len(topic_rows),
                "meeting_rows": len(meeting_rows),
                "complete_core8_by_arm": {
                    arm: int(complete_counts.get(arm, 0))
                    for arm in sentiment_v1.ARM_ORDER
                },
            },
            "aggregation": {
                "topic_weighting": "equal_1_over_8",
                "replicate_weighting": "equal_1_over_5_after_topic_aggregation",
                "empty_primary": "missing_incomplete_core8",
                "empty_sensitivity": "signed_zero_equivalent_to_neutral_probability_one",
                "neural_content_tokens": CONTENT_TOKENS,
                "neural_overlap_tokens": OVERLAP_TOKENS,
                "neural_window_step": WINDOW_STEP,
                "neural_overlap_correction": "sum_inverse_token_coverage",
            },
            "special_token_contract": copy.deepcopy(dict(special_token_contract)),
            "runtime": copy.deepcopy(dict(runtime)),
            "implementation": _implementation_binding(),
        }
    )


def score_neural_v2(
    *,
    backend_id: str,
    inventory_manifest: Path,
    model_manifest: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    expected_visible_gpu: str | None,
    torch_num_threads: int | None,
) -> dict[str, Any]:
    if (
        backend_id not in NEURAL_BACKENDS
        or batch_size <= 0
        or (
            torch_num_threads is not None
            and (
                isinstance(torch_num_threads, bool)
                or not isinstance(torch_num_threads, int)
                or torch_num_threads <= 0
            )
        )
    ):
        raise NeuralSentimentV2Error("unsupported backend or invalid batch size")
    try:
        inventory = sentiment_v1.load_and_validate_inventory(inventory_manifest)
        model_loaded = sentiment_v1.load_and_validate_model_manifest(model_manifest)
    except Exception as exc:
        raise NeuralSentimentV2Error(f"v1 input validation failed: {exc}") from exc
    contract = model_loaded["contract"]
    if contract.backend_id != backend_id:
        raise NeuralSentimentV2Error("backend/model binding mismatch")
    try:
        output_dir = sentiment_v1._fresh_directory(output_dir)
    except Exception as exc:
        raise NeuralSentimentV2Error(str(exc)) from exc
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise NeuralSentimentV2Error(f"neural dependencies unavailable: {exc}") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise NeuralSentimentV2Error("CUDA requested but unavailable")
    visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device.startswith("cuda") and (
        expected_visible_gpu is None or visible_cuda_devices != expected_visible_gpu
    ):
        raise NeuralSentimentV2Error(
            "CUDA scoring requires an exact visible physical GPU binding"
        )
    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)
    tokenizer = AutoTokenizer.from_pretrained(
        model_loaded["model_dir"], local_files_only=True, use_fast=True
    )
    special = _special_token_contract(tokenizer, backend_id)
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_loaded["model_dir"], local_files_only=True
    )
    model.eval()
    model.to(device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    pending: list[tuple[Mapping[str, Any], WindowSpec]] = []
    specs_by_text: dict[str, list[WindowSpec]] = {}
    for source in inventory["rows"]:
        specs = _window_specs(tokenizer, str(source["text"]), backend_id=backend_id)
        specs_by_text[str(source["text_id"])] = specs
        pending.extend((source, spec) for spec in specs)

    window_path = output_dir / "window_scores.jsonl"
    topic_path = output_dir / "topic_scores.jsonl"
    meeting_path = output_dir / "meeting_scores.jsonl"
    accumulators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"weight": 0.0, "probability_sums": [0.0, 0.0, 0.0]}
    )
    with sentiment_v1._JsonlWriter(window_path) as writer, torch.inference_mode():
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            padded = tokenizer.pad(
                {"input_ids": [list(spec.input_ids) for _, spec in batch]},
                padding=True,
                return_tensors="pt",
            )
            padded = {key: value.to(device) for key, value in padded.items()}
            logits_tensor = model(**padded).logits.float()
            probabilities_tensor = torch.softmax(logits_tensor, dim=-1)
            for (source, spec), logits, probabilities in zip(
                batch,
                logits_tensor.cpu().tolist(),
                probabilities_tensor.cpu().tolist(),
                strict=True,
            ):
                if len(logits) != 3 or len(probabilities) != 3:
                    raise NeuralSentimentV2Error("classifier output is not 3-label")
                text_id = str(source["text_id"])
                accumulator = accumulators[text_id]
                accumulator["weight"] += spec.aggregation_weight
                for label_index, probability in enumerate(probabilities):
                    accumulator["probability_sums"][label_index] += (
                        spec.aggregation_weight * float(probability)
                    )
                score = float(probabilities[contract.positive_index]) - float(
                    probabilities[contract.negative_index]
                )
                writer.write(
                    {
                        "schema_version": WINDOW_SCHEMA,
                        "backend": backend_id,
                        "construct": contract.construct,
                        "text_id": text_id,
                        "text_sha256": source["text_sha256"],
                        "window_index": spec.index,
                        "token_start": spec.token_start,
                        "token_end": spec.token_end,
                        "content_token_count": len(spec.content_ids),
                        "content_token_ids": list(spec.content_ids),
                        "content_token_ids_sha256": _token_ids_sha256(spec.content_ids),
                        "input_token_ids": list(spec.input_ids),
                        "input_token_ids_sha256": _token_ids_sha256(spec.input_ids),
                        "aggregation_weight": spec.aggregation_weight,
                        "logits": [float(value) for value in logits],
                        "probabilities": [float(value) for value in probabilities],
                        "label_order": list(contract.labels),
                        "score": score,
                    }
                )

    topic_rows: list[dict[str, Any]] = []
    with sentiment_v1._JsonlWriter(topic_path) as writer:
        for source in inventory["rows"]:
            text_id = str(source["text_id"])
            specs = specs_by_text[text_id]
            accumulator = accumulators.get(text_id)
            if not specs:
                probabilities = None
                score = None
                token_count = 0
                total_weight = 0.0
            else:
                assert accumulator is not None
                token_count = specs[-1].token_end
                total_weight = float(accumulator["weight"])
                if not math.isclose(total_weight, token_count, abs_tol=1e-8):
                    raise NeuralSentimentV2Error("topic overlap weight drift")
                probabilities = [
                    float(value) / total_weight
                    for value in accumulator["probability_sums"]
                ]
                if not math.isclose(sum(probabilities), 1.0, abs_tol=2e-6):
                    raise NeuralSentimentV2Error(
                        "topic probabilities do not sum to one"
                    )
                score = (
                    probabilities[contract.positive_index]
                    - probabilities[contract.negative_index]
                )
            topic = {
                **_topic_base(
                    source, backend_id=backend_id, construct=contract.construct
                ),
                "window_count": len(specs),
                "content_token_count": token_count,
                "coverage_corrected_weight": total_weight,
                "probabilities": probabilities,
                "label_order": list(contract.labels),
                "score": score,
                "neutral_imputed_score": 0.0 if score is None else score,
            }
            writer.write(topic)
            topic_rows.append(topic)
    meeting_rows = _aggregate_meeting_rows(topic_rows, backend_id=backend_id)
    with sentiment_v1._JsonlWriter(meeting_path) as writer:
        for row in meeting_rows:
            writer.write(row)
    properties = (
        torch.cuda.get_device_properties(device) if device.startswith("cuda") else None
    )
    manifest = _backend_manifest(
        backend_id=backend_id,
        output_dir=output_dir,
        inventory_binding=inventory["manifest_binding"],
        model_binding=model_loaded["manifest_binding"],
        window_path=window_path,
        window_rows=len(pending),
        topic_path=topic_path,
        topic_rows=topic_rows,
        meeting_path=meeting_path,
        meeting_rows=meeting_rows,
        special_token_contract=special,
        runtime={
            "device": device,
            "batch_size": batch_size,
            "torch_num_threads": torch.get_num_threads(),
            "expected_visible_gpu": expected_visible_gpu,
            "cuda_visible_devices": visible_cuda_devices,
            "cuda_device_name": None if properties is None else properties.name,
            "cuda_total_memory_bytes": (
                None if properties is None else properties.total_memory
            ),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "model_dtype": str(next(model.parameters()).dtype),
            "inference_mode": True,
            "eval_mode": not model.training,
            "tf32": False,
            "deterministic_algorithms": True,
        },
    )
    sentiment_v1._write_readonly_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_topic_rows(rows: Sequence[Mapping[str, Any]], *, backend_id: str) -> None:
    if len(rows) != sentiment_v1.EXPECTED_TEXTS or len(
        {str(row.get("text_id")) for row in rows}
    ) != len(rows):
        raise NeuralSentimentV2Error("v2 topic coverage/identity drift")
    contract = sentiment_v1.MODEL_CONTRACTS[backend_id]
    for row in rows:
        score = row.get("score")
        probabilities = row.get("probabilities")
        if (
            row.get("schema_version") != TOPIC_SCHEMA
            or row.get("backend") != backend_id
            or row.get("construct") != contract.construct
            or row.get("label_order") != list(contract.labels)
            or (
                score is not None
                and (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not -1.0 <= float(score) <= 1.0
                )
            )
            or row.get("neutral_imputed_score") != (0.0 if score is None else score)
            or (
                score is None
                and (
                    probabilities is not None
                    or row.get("window_count") != 0
                    or row.get("content_token_count") != 0
                )
            )
            or (
                score is not None
                and (
                    not isinstance(probabilities, list)
                    or len(probabilities) != 3
                    or not math.isclose(
                        sum(float(value) for value in probabilities),
                        1.0,
                        abs_tol=2e-6,
                    )
                )
            )
        ):
            raise NeuralSentimentV2Error("v2 topic score row drift")


def _validate_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend_id: str,
    topic_rows: Sequence[Mapping[str, Any]],
    special: Mapping[str, Any],
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    contract = sentiment_v1.MODEL_CONTRACTS[backend_id]
    for row in rows:
        content = row.get("content_token_ids")
        inputs = row.get("input_token_ids")
        probabilities = row.get("probabilities")
        if (
            row.get("schema_version") != WINDOW_SCHEMA
            or row.get("backend") != backend_id
            or row.get("construct") != contract.construct
            or row.get("label_order") != list(contract.labels)
            or not isinstance(content, list)
            or not isinstance(inputs, list)
            or len(inputs) != len(content) + 2
            or inputs[:1] != [special["cls_token_id"]]
            or inputs[-1:] != [special["sep_token_id"]]
            or inputs[1:-1] != content
            or len(inputs) > 512
            or row.get("content_token_ids_sha256") != _token_ids_sha256(content)
            or row.get("input_token_ids_sha256") != _token_ids_sha256(inputs)
            or not isinstance(probabilities, list)
            or len(probabilities) != 3
            or not math.isclose(
                sum(float(value) for value in probabilities), 1.0, abs_tol=2e-6
            )
            or not isinstance(row.get("aggregation_weight"), (int, float))
            or float(row["aggregation_weight"]) <= 0.0
        ):
            raise NeuralSentimentV2Error("v2 window row drift")
        grouped[str(row["text_id"])].append(row)
    for topic in topic_rows:
        windows = sorted(
            grouped.get(str(topic["text_id"]), []),
            key=lambda row: int(row["window_index"]),
        )
        if [row["window_index"] for row in windows] != list(range(len(windows))):
            raise NeuralSentimentV2Error("window indexes are not contiguous")
        if len(windows) != topic["window_count"]:
            raise NeuralSentimentV2Error("topic/window count drift")
        if windows:
            if (
                windows[0]["token_start"] != 0
                or windows[-1]["token_end"] != topic["content_token_count"]
                or not math.isclose(
                    sum(float(row["aggregation_weight"]) for row in windows),
                    float(topic["coverage_corrected_weight"]),
                    abs_tol=1e-8,
                )
            ):
                raise NeuralSentimentV2Error("topic/window coverage drift")
        elif topic["score"] is not None:
            raise NeuralSentimentV2Error("scored topic has no windows")


def load_and_validate_neural_backend(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = sentiment_v1._read_json(manifest_path)
    except Exception as exc:
        raise NeuralSentimentV2Error(str(exc)) from exc
    payload_sha = validate_manifest_integrity(manifest)
    backend_id = str(manifest.get("backend") or "")
    if (
        manifest.get("schema_version") != BACKEND_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != EVALUATION_ID
        or backend_id not in NEURAL_BACKENDS
    ):
        raise NeuralSentimentV2Error("v2 backend manifest header drift")
    _validate_implementation(manifest.get("implementation"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise NeuralSentimentV2Error("v2 backend artifact bindings missing")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for key, expected_rows in (
        ("window_scores", int(manifest["coverage"]["window_rows"])),
        ("topic_scores", sentiment_v1.EXPECTED_TEXTS),
        ("meeting_scores", sentiment_v1.EXPECTED_MEETING_ROWS),
    ):
        binding = artifacts.get(key)
        if not isinstance(binding, Mapping):
            raise NeuralSentimentV2Error(f"v2 {key} binding missing")
        path = Path(str(binding.get("path")))
        if dict(binding) != _binding(path, rows=expected_rows):
            raise NeuralSentimentV2Error(f"v2 {key} binding drift")
        rows = sentiment_v1._read_jsonl(path)
        if len(rows) != expected_rows:
            raise NeuralSentimentV2Error(f"v2 {key} row count drift")
        loaded[key] = rows
    topic_rows = loaded["topic_scores"]
    _validate_topic_rows(topic_rows, backend_id=backend_id)
    special = manifest.get("special_token_contract")
    if (
        not isinstance(special, Mapping)
        or {key: special.get(key) for key in ("cls_token_id", "sep_token_id")}
        != EXPECTED_SPECIAL_TOKEN_IDS[backend_id]
    ):
        raise NeuralSentimentV2Error("v2 manifest special-token contract drift")
    _validate_windows(
        loaded["window_scores"],
        backend_id=backend_id,
        topic_rows=topic_rows,
        special=special,
    )
    meetings = _aggregate_meeting_rows(topic_rows, backend_id=backend_id)
    if loaded["meeting_scores"] != meetings:
        raise NeuralSentimentV2Error("v2 meeting scores are not exact Core8 means")
    try:
        inventory = sentiment_v1.load_and_validate_inventory(
            Path(str(manifest["inventory_manifest"]["path"]))
        )
        model = sentiment_v1.load_and_validate_model_manifest(
            Path(str(manifest["model_manifest"]["path"]))
        )
    except Exception as exc:
        raise NeuralSentimentV2Error(
            f"v1 source binding validation failed: {exc}"
        ) from exc
    if (
        manifest["inventory_manifest"] != inventory["manifest_binding"]
        or manifest["model_manifest"] != model["manifest_binding"]
        or model["contract"].backend_id != backend_id
    ):
        raise NeuralSentimentV2Error("v2 backend v1 source bindings drift")
    inventory_by_id = {str(row["text_id"]): row for row in inventory["rows"]}
    source_fields = (
        "arm",
        "sample_id",
        "meeting_id",
        "meeting_date",
        "meeting_start_date",
        "era",
        "role",
        "topic",
        "topic_order",
        "replicate_id",
        "replicate_seed",
        "cp318_selection_exposed",
        "text_sha256",
        "empty",
    )
    if any(
        any(
            row.get(field)
            != inventory_by_id.get(str(row.get("text_id")), {}).get(field)
            for field in source_fields
        )
        for row in topic_rows
    ):
        raise NeuralSentimentV2Error("v2 topic/inventory source binding drift")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "meeting_scores": loaded["meeting_scores"],
        "topic_scores": topic_rows,
        "window_scores": loaded["window_scores"],
        "inventory": inventory,
        "model": model,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--backend", choices=NEURAL_BACKENDS, required=True)
    score.add_argument("--inventory-manifest", type=Path, default=DEFAULT_INVENTORY)
    score.add_argument("--model-manifest", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--batch-size", type=int, default=512)
    score.add_argument("--expected-visible-gpu")
    score.add_argument("--torch-num-threads", type=int)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "score":
            value = score_neural_v2(
                backend_id=args.backend,
                inventory_manifest=args.inventory_manifest,
                model_manifest=args.model_manifest,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                expected_visible_gpu=args.expected_visible_gpu,
                torch_num_threads=args.torch_num_threads,
            )
        else:
            value = load_and_validate_neural_backend(args.manifest)["manifest"]
    except (NeuralSentimentV2Error, sentiment_v1.SentimentScoringError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_canonical(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKEND_MANIFEST_SCHEMA",
    "EVALUATION_ID",
    "EXPECTED_SPECIAL_TOKEN_IDS",
    "NeuralSentimentV2Error",
    "_aggregate_meeting_rows",
    "_special_token_contract",
    "_window_specs",
    "load_and_validate_neural_backend",
    "score_neural_v2",
]
