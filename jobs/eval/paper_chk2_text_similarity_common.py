"""Immutable preparation contract for the paper chk-2 all-391 similarity audit.

The prepared population is intentionally an in-release reconstruction panel,
not a held-out evaluation set.  It joins each strict ``{prompt,response}`` row
to its sidecar by position and hashes, extracts the sole ``</think>`` suffix as
the synthetic Minutes reference, and renders the exact student chat prompt to
token IDs before any vLLM process starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from open_r1.provenance import fingerprint_artifact_path, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_"
    "v6_downstream128_recovery_v1_20260831"
)
RELEASE_MANIFEST = RELEASE_ROOT / "release_manifest.json"
RELEASE_MANIFEST_SHA256 = (
    "cb022dcd379e069e3d5ad54d7c7fdbc9e2ee29f958c85d5ff45d066f7728c097"
)
PROMPT_CONTRACT = RELEASE_ROOT / "prompt_contract.json"
SYSTEM_PROMPT_SHA256 = (
    "4730a4ed585238547447ab850836db5a9fc67e5c3b1328b485c88d701ae78c4e"
)
USER_PROMPT_TEMPLATE_SHA256 = (
    "423e79849cb66d6361c986f705ea4d4b16a03e28fec5826113eb9c8030a976d0"
)

CHK0_MODEL = ROOT / "models/DeepSeek-R1-Distill-Llama-8B"
CHK0_DIGEST = "bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11"
CHK0_RUNTIME_PAYLOAD_DIGEST = (
    "97480b0fa2614940b2a4277a828620585485495c6742ba463634869adab7cabc"
)
CHK0_RUNTIME_FILES: dict[str, tuple[int, str]] = {
    "config.json": (828, "c6162f33d194369772137ecdd4bdcac4cf3fed4f40607d694b94bcbb8e5dc39f"),
    "generation_config.json": (181, "cd5194726d1e8f7361a8c8425fc11d33ade5e69de1fd7615eb23fae5601af68b"),
    "model-00001-of-000002.safetensors": (8_667_826_246, "7e6b24744354ef4ba547547cc758339090f46ba2da917845cfc69f7d4ded9edb"),
    "model-00002-of-000002.safetensors": (7_392_730_108, "19fb83b79bd0d06d49b7cf6f86b83f5183cd292aa2c028d633ca4fceac1ae742"),
    "model.safetensors.index.json": (24_240, "83bdf4be4bb1a054ff315cd804554c48a88036226fdfbc65bee84ff562fea32a"),
    "tokenizer.json": (9_084_480, "b9c9eb63a8e03059914880f918cd28a880dec8b6e15e4461e1ff677e3743dbb8"),
    "tokenizer_config.json": (3_072, "5a773d1f7a8716f53f414040cbbf94ddf5f55f4f00bc3b4cc8fd8d6b64369777"),
}
CHK1_MODEL = ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
CHK1_DIGEST = "0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3"
CHK2_ADAPTER = ROOT / (
    "output/training/retrain_v2/"
    "paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901/"
    "adapters/chk2/checkpoint-50"
)
CHK2_ADAPTER_MODEL_SHA256 = (
    "158f73c538f6969c7d829f869ec79bff501caaa92c40bb61b564c957d92717bf"
)
CHK2_ADAPTER_CONFIG_SHA256 = (
    "1b92cf693cb4f3e2b727dafdd84967c8578438b3380626ec100053f810a398eb"
)
SELECTION_RECEIPT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_minutes_checkpoint_hard_gate_probe_v1_20260901/"
    "relaxed_checkpoint_selection_v1.json"
)
SELECTION_RECEIPT_SHA256 = (
    "9c2fcff135114ff1165c60b881f109c69bb3c0f3b04368157744e634b99e138f"
)
SEMANTIC_MANIFEST = ROOT / "configs/main/checkpoint_eval_semantic_models.json"
SEMANTIC_MANIFEST_SHA256 = (
    "639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
)

EXPECTED_SPLIT_ROWS = {"train": 305, "validation": 42, "test": 44}
EXPECTED_SPLIT_MEETINGS = {"train": 98, "validation": 13, "test": 13}
EXPECTED_ROWS = 391
EXPECTED_MEETINGS = 124
BOUNDARY = "</think>"
REPLICATE_SEEDS = tuple(20_260_901 + index for index in range(10))


class SimilarityPreparationError(RuntimeError):
    """A source, model, prompt, tokenizer, or immutable output binding drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SimilarityPreparationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            _require(raw.endswith("\n"), f"unterminated JSONL row {path}:{number}")
            _require(bool(raw.strip()), f"blank JSONL row {path}:{number}")
            value = json.loads(raw)
            _require(isinstance(value, dict), f"object required {path}:{number}")
            rows.append(value)
    return rows


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _verify_declared_file(
    release_manifest: Mapping[str, Any], split: str, kind: str
) -> Path:
    raw = release_manifest["artifacts"]["minutes_alignment"][split][kind]
    path = RELEASE_ROOT / str(raw["path"])
    _require(path.is_file() and not path.is_symlink(), f"missing release {split}/{kind}")
    _require(sha256_file(path) == raw["sha256"], f"release {split}/{kind} SHA drift")
    _require(path.stat().st_size == raw["bytes"], f"release {split}/{kind} size drift")
    _require(raw["rows"] == EXPECTED_SPLIT_ROWS[split], f"release {split} rows drift")
    return path


def extract_reference(response: str) -> str:
    _require(isinstance(response, str), "response must be text")
    _require(response.count(BOUNDARY) == 1, "response must contain one </think>")
    reasoning, reference = response.split(BOUNDARY, 1)
    _require(bool(reasoning.strip()), "teacher reasoning is empty")
    reference = reference.strip()
    _require(bool(reference), "synthetic Minutes reference is empty")
    _require(BOUNDARY not in reference, "reference contains a control boundary")
    _require("\n\n" not in reference, "reference is not one paragraph")
    return reference


def _prompt_ids(tokenizer: Any, system_prompt: str, prompt: str) -> list[int]:
    values = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(values, "tolist"):
        values = values.tolist()
    _require(isinstance(values, list) and bool(values), "chat template returned no IDs")
    _require(not isinstance(values[0], list), "unexpected batched token IDs")
    return [int(value) for value in values]


def _verify_static_bindings() -> dict[str, Any]:
    _require(sha256_file(RELEASE_MANIFEST) == RELEASE_MANIFEST_SHA256, "release manifest drift")
    _require(sha256_file(SELECTION_RECEIPT) == SELECTION_RECEIPT_SHA256, "cp50 receipt drift")
    receipt = _read_json(SELECTION_RECEIPT)
    _require(receipt.get("status") == "selected", "cp selection is not final")
    _require(receipt.get("selected_checkpoint") == 50, "cp50 is not selected")
    _require(receipt.get("scope", {}).get("selection_only") is True, "selection scope drift")
    _require(sha256_file(SEMANTIC_MANIFEST) == SEMANTIC_MANIFEST_SHA256, "semantic manifest drift")
    semantic = _read_json(SEMANTIC_MANIFEST)
    _require(semantic.get("network_at_scoring_time") is False, "semantic models must be offline")
    _require(
        sha256_file(CHK2_ADAPTER / "adapter_model.safetensors")
        == CHK2_ADAPTER_MODEL_SHA256,
        "cp50 adapter weights drift",
    )
    _require(
        sha256_file(CHK2_ADAPTER / "adapter_config.json")
        == CHK2_ADAPTER_CONFIG_SHA256,
        "cp50 adapter config drift",
    )
    # chk0 is a Hugging Face git clone.  Its historical full-directory digest
    # (bfb0...) is the lineage identifier requested by the experiment, while
    # mutable .git metadata makes recomputing that whole-tree digest unsuitable
    # at runtime.  Verify every runtime-bearing file against the already sealed
    # allowlist and bind both identities explicitly.
    runtime_records: list[dict[str, Any]] = []
    runtime_aggregate = hashlib.sha256()
    runtime_total = 0
    for relative, (expected_bytes, expected_sha) in sorted(CHK0_RUNTIME_FILES.items()):
        candidate = CHK0_MODEL / relative
        _require(candidate.is_file() and not candidate.is_symlink(), f"missing chk0 runtime file: {relative}")
        observed_bytes = candidate.stat().st_size
        observed_sha = sha256_file(candidate)
        _require(observed_bytes == expected_bytes, f"chk0 runtime size drift: {relative}")
        _require(observed_sha == expected_sha, f"chk0 runtime SHA drift: {relative}")
        runtime_aggregate.update(
            f"{relative}\0{observed_bytes}\0{observed_sha}\n".encode("utf-8")
        )
        runtime_total += observed_bytes
        runtime_records.append(
            {"path": relative, "bytes": observed_bytes, "sha256": observed_sha}
        )
    _require(runtime_aggregate.hexdigest() == CHK0_RUNTIME_PAYLOAD_DIGEST, "chk0 runtime aggregate drift")
    chk0 = {
        "path": str(CHK0_MODEL.resolve()),
        "kind": "directory_runtime_payload_allowlist",
        "sha256": CHK0_RUNTIME_PAYLOAD_DIGEST,
        "lineage_full_directory_sha256": CHK0_DIGEST,
        "file_count": len(runtime_records),
        "total_bytes": runtime_total,
        "files": runtime_records,
        "excluded_from_runtime_fingerprint": [".git/**", ".gitattributes", "README.md", "LICENSE", "figures/**"],
    }
    chk1 = fingerprint_artifact_path(CHK1_MODEL)
    chk2 = fingerprint_artifact_path(CHK2_ADAPTER)
    _require(chk1["sha256"] == CHK1_DIGEST, "chk1 directory digest drift")
    return {
        "chk0": chk0,
        "chk1": chk1,
        "chk2": chk2,
        "selection_receipt": _record(SELECTION_RECEIPT),
        "semantic_manifest": _record(SEMANTIC_MANIFEST),
        "semantic_models": semantic["models"],
    }


def build_prepared_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and project all 391 release rows in stable split/release order."""

    release_manifest = _read_json(RELEASE_MANIFEST)
    prompt_contract = _read_json(PROMPT_CONTRACT)
    system_prompt = prompt_contract.get("system_prompt")
    _require(isinstance(system_prompt, str), "system prompt missing")
    _require(sha256_text(system_prompt) == SYSTEM_PROMPT_SHA256, "system prompt drift")
    _require(
        prompt_contract.get("system_prompt_sha256") == SYSTEM_PROMPT_SHA256,
        "declared system prompt SHA drift",
    )
    _require(
        prompt_contract.get("user_prompt_template_sha256")
        == USER_PROMPT_TEMPLATE_SHA256,
        "user template SHA drift",
    )

    from transformers import AutoTokenizer

    tokenizer0 = AutoTokenizer.from_pretrained(
        str(CHK0_MODEL), local_files_only=True, trust_remote_code=True, use_fast=True,
        fix_mistral_regex=False,
    )
    tokenizer1 = AutoTokenizer.from_pretrained(
        str(CHK1_MODEL), local_files_only=True, trust_remote_code=True, use_fast=True,
        fix_mistral_regex=False,
    )
    prepared: list[dict[str, Any]] = []
    meetings: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    max_prompt_tokens = 0
    for split in ("train", "validation", "test"):
        data_path = _verify_declared_file(release_manifest, split, "data")
        sidecar_path = _verify_declared_file(release_manifest, split, "manifest")
        data = _read_jsonl(data_path)
        sidecars = _read_jsonl(sidecar_path)
        _require(
            len(data) == len(sidecars) == EXPECTED_SPLIT_ROWS[split],
            f"{split} data/sidecar row drift",
        )
        for row, sidecar in zip(data, sidecars, strict=True):
            _require(set(row) == {"prompt", "response"}, f"{split} training schema drift")
            _require(sidecar.get("split") == split, f"{split} sidecar split drift")
            sample_id = sidecar.get("sample_id")
            meeting_id = sidecar.get("meeting_date")
            _require(isinstance(sample_id, str) and bool(sample_id), "invalid sample ID")
            _require(sample_id not in seen_ids, f"duplicate sample ID: {sample_id}")
            _require(isinstance(meeting_id, str) and bool(meeting_id), "invalid meeting ID")
            prompt = row["prompt"]
            response = row["response"]
            _require(sidecar.get("prompt_sha256") == sha256_text(prompt), "prompt hash drift")
            _require(sidecar.get("response_sha256") == sha256_text(response), "response hash drift")
            reference = extract_reference(response)
            token_ids0 = _prompt_ids(tokenizer0, system_prompt, prompt)
            token_ids1 = _prompt_ids(tokenizer1, system_prompt, prompt)
            _require(token_ids0 == token_ids1, f"chk0/chk1 tokenizer drift: {sample_id}")
            _require(len(token_ids0) + 2048 <= 4096, f"prompt context overflow: {sample_id}")
            token_ids_sha = sha256_text(canonical_json(token_ids0))
            prepared.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "meeting_id": meeting_id,
                    "release_index": int(sidecar["release_index"]),
                    "source_index": int(sidecar["source_index"]),
                    "prompt": prompt,
                    "reference": reference,
                    "prompt_sha256": sha256_text(prompt),
                    "reference_sha256": sha256_text(reference),
                    "prompt_token_ids": token_ids0,
                    "prompt_token_count": len(token_ids0),
                    "prompt_token_ids_sha256": token_ids_sha,
                }
            )
            max_prompt_tokens = max(max_prompt_tokens, len(token_ids0))
            seen_ids.add(sample_id)
            split_counts[split] += 1
            meetings[split].add(meeting_id)

    _require(len(prepared) == EXPECTED_ROWS, "prepared row count drift")
    _require(dict(split_counts) == EXPECTED_SPLIT_ROWS, "prepared split count drift")
    _require(
        {key: len(value) for key, value in meetings.items()} == EXPECTED_SPLIT_MEETINGS,
        "meeting counts drift",
    )
    meeting_union = set().union(*meetings.values())
    _require(len(meeting_union) == EXPECTED_MEETINGS, "meeting union drift")
    for left in meetings:
        for right in meetings:
            if left < right:
                _require(not meetings[left] & meetings[right], "meeting split leakage")
    return prepared, {
        "release_manifest": _record(RELEASE_MANIFEST),
        "prompt_contract": _record(PROMPT_CONTRACT),
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "user_prompt_template_sha256": USER_PROMPT_TEMPLATE_SHA256,
        "split_rows": dict(split_counts),
        "split_meetings": {key: len(value) for key, value in meetings.items()},
        "meeting_union": len(meeting_union),
        "max_prompt_tokens": max_prompt_tokens,
    }


def prepare(output_root: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "evaluation_manifest.json"
    if output_root.exists():
        return verify_prepared(output_root)
    prepared, release_binding = build_prepared_rows()
    model_binding = _verify_static_bindings()
    staging = output_root.with_name(f".{output_root.name}.prepare-{os.getpid()}")
    _require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    samples_path = staging / "samples.jsonl"
    samples_payload = "".join(canonical_json(row) + "\n" for row in prepared).encode("utf-8")
    _write_bytes_exclusive(samples_path, samples_payload)
    samples_record = _record(samples_path, rows=EXPECTED_ROWS)
    # The worker accepts a dedicated ledger or token IDs embedded in samples.
    # Pointing both bindings to the same immutable projection avoids divergence.
    manifest = {
        "schema_version": "paper-chk2-text-similarity-all391-evaluation-v1",
        "created_at_utc": _utc_now(),
        "status": "prepared",
        "scope": {
            "training_release_reconstruction_diagnostic": True,
            "leakage_safe_evaluation": False,
            "held_out_generalization_claim_allowed": False,
            "reference": "teacher_synthetic_rewritten_minutes_after_unique_think_boundary",
        },
        "population": {
            "rows": EXPECTED_ROWS,
            "meetings": EXPECTED_MEETINGS,
            "split_rows": EXPECTED_SPLIT_ROWS,
            "split_meetings": EXPECTED_SPLIT_MEETINGS,
        },
        "release": release_binding,
        "inputs": {
            "samples": {**samples_record, "path": "samples.jsonl"},
            "prompt_token_ledger": {**samples_record, "path": "samples.jsonl"},
        },
        "models": {
            "chk0": {
                "path": str(CHK0_MODEL.resolve()),
                "sha256": CHK0_DIGEST,
                "runtime_payload_sha256": CHK0_RUNTIME_PAYLOAD_DIGEST,
                "files": {
                    row["path"]: row["sha256"] for row in model_binding["chk0"]["files"]
                },
            },
            "chk1": {"path": str(CHK1_MODEL.resolve()), "sha256": CHK1_DIGEST},
            "chk2": {
                "adapter_path": str(CHK2_ADAPTER.resolve()),
                "base_model_path": str(CHK1_MODEL.resolve()),
                "sha256": model_binding["chk2"]["sha256"],
                "adapter_model_sha256": CHK2_ADAPTER_MODEL_SHA256,
                "config_sha256": CHK2_ADAPTER_CONFIG_SHA256,
            },
        },
        "model_fingerprints": {
            key: model_binding[key] for key in ("chk0", "chk1", "chk2")
        },
        "checkpoint_selection": model_binding["selection_receipt"],
        "semantic_scoring": {
            "manifest": model_binding["semantic_manifest"],
            "models": model_binding["semantic_models"],
            "network_at_scoring_time": False,
        },
        "replicate_seeds": list(REPLICATE_SEEDS),
        "generation": {
            "k": 10,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": 2048,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 4096,
            "tensor_parallel_size_per_engine": 1,
            "data_parallel_size_per_engine": 1,
            "independent_single_gpu_engines": 2,
        },
        "statistics": {
            "bootstrap_draws": 1000,
            "primary_unit": "meeting_equal_weighted_split_stratified_paired",
            "pooled_k10": "meeting_and_replicate_hierarchical_paired",
            "row_bootstrap_is_sensitivity_only": True,
            "holm_primary_family": "raw_best_effort_three_contrasts_times_two_metrics",
        },
    }
    _write_json_exclusive(staging / "evaluation_manifest.json", manifest)
    os.replace(staging, output_root)
    return verify_prepared(output_root)


def verify_prepared(output_root: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema_version") == "paper-chk2-text-similarity-all391-evaluation-v1",
        "evaluation manifest schema drift",
    )
    _require(manifest.get("status") == "prepared", "evaluation is not prepared")
    _require(manifest.get("population", {}).get("split_rows") == EXPECTED_SPLIT_ROWS, "split drift")
    _require(manifest.get("checkpoint_selection", {}).get("sha256") == SELECTION_RECEIPT_SHA256, "selection receipt drift")
    samples_raw = manifest.get("inputs", {}).get("samples", {})
    samples_path = output_root / str(samples_raw.get("path"))
    _require(sha256_file(samples_path) == samples_raw.get("sha256"), "prepared samples SHA drift")
    rows = _read_jsonl(samples_path)
    _require(len(rows) == EXPECTED_ROWS, "prepared samples row drift")
    _require(len({row.get("sample_id") for row in rows}) == EXPECTED_ROWS, "sample ID uniqueness drift")
    _require(Counter(row.get("split") for row in rows) == Counter(EXPECTED_SPLIT_ROWS), "sample split drift")
    for row in rows:
        _require(set(row) >= {"sample_id", "split", "meeting_id", "prompt", "reference", "prompt_token_ids"}, "sample schema drift")
        _require(row["prompt_sha256"] == sha256_text(row["prompt"]), "sample prompt SHA drift")
        _require(row["reference_sha256"] == sha256_text(row["reference"]), "sample reference SHA drift")
        _require(row["prompt_token_ids_sha256"] == sha256_text(canonical_json(row["prompt_token_ids"])), "sample token SHA drift")
    return {
        "schema_version": "paper-chk2-text-similarity-prepare-verification-v1",
        "status": "verified",
        "evaluation_manifest": _record(manifest_path),
        "samples": _record(samples_path, rows=EXPECTED_ROWS),
        "split_rows": EXPECTED_SPLIT_ROWS,
        "meetings": EXPECTED_MEETINGS,
        "generation_rows_expected": 11_730,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare(args.output_root) if args.command == "prepare" else verify_prepared(args.output_root)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
