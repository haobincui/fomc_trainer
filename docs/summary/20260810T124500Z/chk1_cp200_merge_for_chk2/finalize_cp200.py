"""Finalize and seal validation evidence for the cp200 merged chk1 model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from jobs.main.checkpoint_provenance import write_immutable_json
from jobs.retrain_v2.merge_attestation import verify_merge_attestation
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE_REL = Path("models/DeepSeek-R1-Distill-Llama-8B")
BASE = ROOT / BASE_REL
ADAPTER_REL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/"
    "adapters/chk1/checkpoint-200"
)
ADAPTER = ROOT / ADAPTER_REL
MERGED_REL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/"
    "merged/chk1"
)
MERGED = ROOT / MERGED_REL
CONFIG_REL = Path(
    "configs/retrain_v2/"
    "chk1_clean_v2_lr1e6_cp200_merge_for_chk2_20260810.yaml"
)
CONFIG = ROOT / CONFIG_REL
AUTH = HERE / "chk1_cp200_to_chk2_authorization.json"
PROMOTION = HERE / "promotion_manifest.json"
MERGE_RESULT = HERE / "merge_result.json"
EXACT = HERE / "chk1_cp200_exact_merge_lineage.json"
LOAD_SMOKE = HERE / "load_smoke.json"
CHECKPOINT_MANIFEST = HERE / "checkpoint_manifest.json"
README = HERE / "README.md"
SHA256SUMS = HERE / "SHA256SUMS"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _index_inventory(model: Path) -> dict[str, Any]:
    index = _read_json(model / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and weight_map, "Invalid weight index")
    indexed = set(weight_map)
    header_keys: set[str] = set()
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        path = model / shard
        _require(path.is_file(), f"Missing model shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        duplicate = header_keys & keys
        _require(not duplicate, f"Duplicate tensors across shards: {sorted(duplicate)[:3]}")
        header_keys.update(keys)
    missing = sorted(indexed - header_keys)
    unindexed = sorted(header_keys - indexed)
    _require(not missing and not unindexed, "Weight index/header inventory mismatch")
    return {
        "tensor_count": len(indexed),
        "shard_count": len(shards),
        "shards": shards,
        "missing_tensors": missing,
        "unindexed_tensors": unindexed,
        "index_sha256": sha256_file(model / "model.safetensors.index.json"),
    }


def _tokenizer_parity() -> dict[str, Any]:
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, use_fast=True
    )
    merged_tokenizer = AutoTokenizer.from_pretrained(
        MERGED, local_files_only=True, use_fast=True
    )
    _require(base_tokenizer.get_vocab() == merged_tokenizer.get_vocab(), "Vocab drift")
    _require(
        base_tokenizer.get_added_vocab() == merged_tokenizer.get_added_vocab(),
        "Added vocab drift",
    )
    special_ids = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        base_value = getattr(base_tokenizer, name)
        merged_value = getattr(merged_tokenizer, name)
        _require(base_value == merged_value, f"Special token drift: {name}")
        special_ids[name] = merged_value
    texts = [
        "Federal funds rate was unchanged.",
        "Q1 2019 increased 25 basis points.",
        "<think>\nEvidence analysis\n</think>\nFinal answer.",
        "数据与分析",
    ]
    for text in texts:
        _require(
            base_tokenizer.encode(text) == merged_tokenizer.encode(text),
            f"Text tokenization drift: {text!r}",
        )
    messages = [
        {"role": "system", "content": "Analyze supplied evidence."},
        {"role": "user", "content": "Reply with one short word."},
    ]
    base_ids = base_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    merged_ids = merged_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    _require(base_ids == merged_ids, "Chat-template token parity failed")
    _require(
        isinstance(merged_ids, list)
        and merged_ids
        and all(isinstance(item, int) for item in merged_ids),
        "Chat-template IDs are not a flat integer list",
    )
    _require(
        merged_ids.count(merged_tokenizer.bos_token_id) == 1,
        "Chat-template output does not contain exactly one BOS",
    )
    return {
        "tokenizer": merged_tokenizer,
        "prompt_ids": merged_ids,
        "tokenizer_class": type(merged_tokenizer).__name__,
        "vocab_size": len(merged_tokenizer),
        "special_token_ids": special_ids,
        "fixed_text_cases": len(texts),
        "vocabulary_parity": True,
        "fixed_text_token_parity": True,
        "chat_template_token_parity": True,
        "single_bos": True,
        "byte_identity_required": False,
        "byte_identity_note": (
            "Merged tokenizer files were reserialized by save_pretrained; semantic "
            "parity, not JSON byte identity, is the enforced contract."
        ),
    }


def _load_forward_smoke(tokenizer: Any, prompt_ids: list[int]) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(MERGED, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MERGED,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    input_ids = torch.tensor([prompt_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
    _require(torch.isfinite(logits).all().item(), "Forward logits are not finite")
    next_token = int(torch.argmax(logits, dim=-1).item())
    result = {
        "status": "passed",
        "config_class": type(config).__name__,
        "architectures": config.architectures,
        "load_mode": "4bit_nf4_double_quant_bf16",
        "attention_implementation": "sdpa",
        "device": str(model.device),
        "prompt_tokens": len(prompt_ids),
        "logits_finite": True,
        "argmax_next_token_id": next_token,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "known_non_blocking_warnings": [
            "transformers tokenizer regex advisory; actual base/merged token parity passed",
            "rope_scaling original_max_position_embeddings advisory",
        ],
    }
    del model
    torch.cuda.empty_cache()
    return result


def _write_text_immutable(path: Path, text: str) -> None:
    if path.is_file():
        _require(path.read_text(encoding="utf-8") == text, f"Text record drift: {path}")
        return
    _require(not path.exists(), f"Text destination exists and is unsafe: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    for path in (BASE, ADAPTER, MERGED, CONFIG, AUTH, PROMOTION, MERGE_RESULT, EXACT):
        _require(path.exists(), f"Required artifact missing: {path}")

    authorization = _read_json(AUTH)
    promotion = _read_json(PROMOTION)
    promotion_payload_sha = validate_manifest_integrity(promotion)
    exact = _read_json(EXACT)
    exact_payload_sha = validate_manifest_integrity(exact)
    _require(
        exact.get("conclusion") == "exact_base_plus_adapter_merge_verified",
        "Exact merge proof failed",
    )
    tensor_verification = exact.get("tensor_verification", {})
    _require(tensor_verification.get("mismatch_count") == 0, "Tensor mismatch found")

    attestation = _read_json(MERGED / "merge_attestation.json")
    verified_attestation = verify_merge_attestation(
        MERGED, expected_binding=attestation["binding"]
    )
    binding = verified_attestation["binding"]
    _require(
        binding.get("authorization", {}).get("file_sha256") == sha256_file(AUTH),
        "Attestation authorization binding mismatch",
    )
    _require(
        binding.get("promotion_manifest", {}).get("file_sha256")
        == sha256_file(PROMOTION),
        "Attestation promotion binding mismatch",
    )
    _require(
        binding.get("promotion_manifest", {}).get("payload_sha256")
        == promotion_payload_sha,
        "Attestation promotion payload mismatch",
    )
    _require(
        binding.get("semantic_audit", {}).get("status") == "failed",
        "Attestation hid semantic audit failure",
    )

    inventory = _index_inventory(MERGED)
    _require(inventory["tensor_count"] == 291, "Unexpected tensor count")
    tokenizer = _tokenizer_parity()
    smoke = _load_forward_smoke(tokenizer["tokenizer"], tokenizer["prompt_ids"])
    tokenizer_receipt = {key: value for key, value in tokenizer.items() if key not in {"tokenizer", "prompt_ids"}}
    load_payload = seal_manifest(
        {
            "schema_version": "chk1-cp200-merged-load-smoke-v1",
            "status": "passed",
            "model_path": str(MERGED_REL),
            "model_fingerprint": fingerprint_artifact_path(MERGED),
            "weight_inventory": inventory,
            "tokenizer_validation": tokenizer_receipt,
            "runtime_validation": smoke,
        }
    )
    load_file_sha = write_immutable_json(LOAD_SMOKE, load_payload)

    merged_fingerprint = fingerprint_artifact_path(MERGED)
    manifest = seal_manifest(
        {
            "schema_version": "chk1-cp200-merged-checkpoint-manifest-v1",
            "status": "ready_for_chk2_parent_under_explicit_override",
            "artifact_id": "chk1-clean-v2-lr1e6-cp200-merged-for-chk2",
            "model_path": str(MERGED_REL),
            "model_fingerprint": merged_fingerprint,
            "verified_parent": {
                "artifact_id": "chk0-deepseek-r1-distill-llama-8b",
                "path": str(BASE_REL),
                "fingerprint": authorization["bindings"]["base_model"],
            },
            "source_adapter": {
                "path": str(ADAPTER_REL),
                "checkpoint_step": 200,
                "fingerprint": authorization["bindings"]["source_adapter"],
                "adapter_weights_sha256": authorization["bindings"][
                    "adapter_weights_sha256"
                ],
            },
            "authorization": {
                "path": str(AUTH.relative_to(ROOT)),
                "file_sha256": sha256_file(AUTH),
                "authorization_sha256": authorization["authorization_sha256"],
                "scope": authorization["scope"],
            },
            "promotion_manifest": {
                "path": str(PROMOTION.relative_to(ROOT)),
                "file_sha256": sha256_file(PROMOTION),
                "payload_sha256": promotion_payload_sha,
            },
            "merge_attestation": {
                "path": str((MERGED / "merge_attestation.json").relative_to(ROOT)),
                "file_sha256": sha256_file(MERGED / "merge_attestation.json"),
                "canonical_payload_sha256": verified_attestation[
                    "canonical_payload_sha256"
                ],
            },
            "exact_tensor_lineage": {
                "path": str(EXACT.relative_to(ROOT)),
                "file_sha256": sha256_file(EXACT),
                "payload_sha256": exact_payload_sha,
                "conclusion": exact["conclusion"],
                "mismatch_count": 0,
            },
            "load_smoke": {
                "path": str(LOAD_SMOKE.relative_to(ROOT)),
                "file_sha256": load_file_sha,
                "payload_sha256": load_payload["integrity"]["payload_sha256"],
                "status": "passed",
            },
            "semantic_audit": {
                "status": "failed",
                "summary": authorization["bindings"]["semantic_audit_summary"],
                "claim_boundary": (
                    "Explicit user override permits this model only as a chk2 parent; "
                    "no semantic-audit pass is claimed."
                ),
            },
            "canonical_training_receipt": {
                "status": "unavailable_not_applicable",
                "reason": promotion["training_receipt"]["reason"],
            },
            "scope": {
                "allowed_stage": "chk2",
                "further_downstream_stages_allowed": [],
            },
        }
    )
    manifest_file_sha = write_immutable_json(CHECKPOINT_MANIFEST, manifest)

    tracked = [
        CONFIG,
        AUTH,
        PROMOTION,
        MERGE_RESULT,
        EXACT,
        LOAD_SMOKE,
        CHECKPOINT_MANIFEST,
        MERGED / "merge_attestation.json",
        MERGED / "config.json",
        MERGED / "model.safetensors.index.json",
        MERGED / "tokenizer.json",
        MERGED / "tokenizer_config.json",
    ] + sorted(MERGED.glob("model-*.safetensors"))
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT)}\n" for path in tracked
    )
    _write_text_immutable(SHA256SUMS, sums)

    readme = f"""# chk1 checkpoint-200 merge for chk2

Status: `ready_for_chk2_parent_under_explicit_override`

Merged model: `{MERGED_REL}`

The model is an exact BF16 merge of `{BASE_REL}` plus `{ADAPTER_REL}`.  The CPU tensor verifier checked all 291 model tensors: 224 adapted tensors and 67 unchanged tensors, with zero mismatches.  A 4-bit NF4 load and finite-logits forward smoke passed in the `fomc_trainer` environment.

The user explicitly authorized using checkpoint-200 as the chk2 parent.  The source SFT semantic audit remains `failed` (234 failed rows, 2 judge errors, 611 blocking violations); this promotion does not claim otherwise.  The override is limited to chk2 and does not authorize chk3 or chk4.

This historical nested checkpoint cannot honestly receive a retroactive canonical training receipt.  `promotion_manifest.json`, the in-model `merge_attestation.json`, the exact tensor proof, and `checkpoint_manifest.json` form the truthful lineage chain.

Key evidence:

- `chk1_cp200_to_chk2_authorization.json`
- `promotion_manifest.json`
- `merge_result.json`
- `chk1_cp200_exact_merge_lineage.json`
- `load_smoke.json`
- `checkpoint_manifest.json`
- `SHA256SUMS`

Known non-blocking warnings: Transformers reports a tokenizer regex advisory and a rope-scaling advisory.  Base/merged vocabulary, special-token IDs, four fixed text encodings, and chat-template token IDs are exactly equal under the actual training environment.
"""
    _write_text_immutable(README, readme)

    print(
        json.dumps(
            {
                "status": "ready_for_chk2_parent_under_explicit_override",
                "model_path": str(MERGED),
                "model_sha256": merged_fingerprint["sha256"],
                "model_total_bytes": merged_fingerprint["total_bytes"],
                "checkpoint_manifest": str(CHECKPOINT_MANIFEST),
                "checkpoint_manifest_file_sha256": manifest_file_sha,
                "exact_merge_payload_sha256": exact_payload_sha,
                "tensor_mismatch_count": 0,
                "load_smoke": smoke,
                "tokenizer_parity": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
