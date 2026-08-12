"""Prove that a merged model is the exact result of a base plus LoRA adapter.

The verifier is CPU-only and performs no training or model writes.  It checks
every indexed model tensor:

* tensors without a corresponding LoRA pair must be byte-value identical; and
* adapted tensors must exactly equal the float32 PEFT merge expression,
  cast to the stored merged dtype.

The optional evidence file is sealed and written immutably.  It supplements,
but never modifies, an existing checkpoint manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open

from jobs.main.checkpoint_provenance import write_immutable_json
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "lora-merge-lineage-evidence-v1"
ALGORITHM_VERSION = "peft-lora-fp32-exact-v1"


def _directory(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {resolved}")
    return resolved


def _file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _safe_shard_name(raw_name: object, *, index_path: Path) -> str:
    name = str(raw_name or "").strip()
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 1
        or relative.suffix != ".safetensors"
    ):
        raise ValueError(f"Unsafe shard name in {index_path}: {raw_name!r}")
    return name


def _weight_map(model: Path) -> tuple[Path, dict[str, str]]:
    index_path = model / "model.safetensors.index.json"
    index = _json_object(index_path, label="model weight index")
    raw_map = index.get("weight_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError(f"Model weight index has no weight_map: {index_path}")
    weight_map: dict[str, str] = {}
    for raw_tensor_name, raw_shard_name in raw_map.items():
        tensor_name = str(raw_tensor_name or "").strip()
        if not tensor_name:
            raise ValueError(f"Empty tensor name in {index_path}")
        shard_name = _safe_shard_name(raw_shard_name, index_path=index_path)
        if not (model / shard_name).is_file():
            raise FileNotFoundError(
                f"Weight index references a missing shard: {model / shard_name}"
            )
        weight_map[tensor_name] = shard_name
    return index_path, weight_map


def _path_has_suffix(path: Path, declared: object) -> bool:
    declared_path = Path(str(declared or "").strip())
    if not declared_path.parts:
        return False
    if declared_path.is_absolute():
        return path == declared_path.expanduser().resolve()
    return len(declared_path.parts) <= len(path.parts) and (
        path.parts[-len(declared_path.parts) :] == declared_path.parts
    )


def _adapter_target_name(adapter_key: str) -> tuple[str, str] | None:
    prefix = "base_model.model."
    for suffix, side in (
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
    ):
        if adapter_key.startswith(prefix) and adapter_key.endswith(suffix):
            stem = adapter_key[len(prefix) : -len(suffix)]
            return f"{stem}.weight", side
    return None


def _adapter_pairs(
    adapter_keys: set[str],
) -> dict[str, tuple[str, str]]:
    raw_pairs: dict[str, dict[str, str]] = {}
    unsupported: list[str] = []
    for key in sorted(adapter_keys):
        parsed = _adapter_target_name(key)
        if parsed is None:
            unsupported.append(key)
            continue
        target, side = parsed
        if side in raw_pairs.setdefault(target, {}):
            raise ValueError(f"Duplicate LoRA {side} tensor for {target}")
        raw_pairs[target][side] = key
    if unsupported:
        raise ValueError(
            "Only standard LoRA A/B weight tensors are supported; unexpected "
            f"adapter keys: {unsupported[:5]}"
        )
    incomplete = {
        target: sorted(pair)
        for target, pair in raw_pairs.items()
        if set(pair) != {"A", "B"}
    }
    if incomplete:
        raise ValueError(f"Incomplete LoRA tensor pairs: {incomplete}")
    return {
        target: (pair["A"], pair["B"])
        for target, pair in raw_pairs.items()
    }


def _scaling(adapter_config: Mapping[str, Any]) -> tuple[float, str]:
    rank = int(adapter_config.get("r", 0))
    alpha = float(adapter_config.get("lora_alpha", 0))
    if rank <= 0 or not math.isfinite(alpha):
        raise ValueError("adapter_config must contain positive r and finite lora_alpha")
    if bool(adapter_config.get("use_dora", False)):
        raise ValueError("DoRA merge verification is not supported by this verifier")
    if bool(adapter_config.get("fan_in_fan_out", False)):
        raise ValueError("fan_in_fan_out LoRA is not supported by this verifier")
    if str(adapter_config.get("bias", "none")) != "none":
        raise ValueError("Only bias='none' LoRA lineage is supported")
    if bool(adapter_config.get("use_rslora", False)):
        return alpha / math.sqrt(rank), "lora_alpha/sqrt(r)"
    return alpha / rank, "lora_alpha/r"


def _critical_files(
    base: Path,
    adapter: Path,
    merged: Path,
) -> dict[str, dict[str, Any]]:
    candidates = {
        "base_config": base / "config.json",
        "base_weight_index": base / "model.safetensors.index.json",
        "base_tokenizer": base / "tokenizer.json",
        "base_tokenizer_config": base / "tokenizer_config.json",
        "adapter_config": adapter / "adapter_config.json",
        "adapter_weights": adapter / "adapter_model.safetensors",
        "merged_config": merged / "config.json",
        "merged_weight_index": merged / "model.safetensors.index.json",
        "merged_tokenizer": merged / "tokenizer.json",
        "merged_tokenizer_config": merged / "tokenizer_config.json",
    }
    records: dict[str, dict[str, Any]] = {}
    for label, path in candidates.items():
        if path.is_file():
            records[label] = {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def _training_config_checks(
    training_config_path: Path,
    *,
    base: Path,
    adapter: Path,
    merged: Path,
    adapter_config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a mapping: {training_config_path}")
    path_checks = {
        "model_name_or_path_matches_base": _path_has_suffix(
            base, payload.get("model_name_or_path")
        ),
        "output_dir_matches_adapter": _path_has_suffix(
            adapter, payload.get("output_dir")
        ),
        "peft_merged_model_path_matches_merged": _path_has_suffix(
            merged, payload.get("peft_merged_model_path")
        ),
    }
    metadata_checks = {
        "r_matches": int(payload.get("peft_r", -1))
        == int(adapter_config.get("r", -2)),
        "lora_alpha_matches": float(payload.get("peft_lora_alpha", float("nan")))
        == float(adapter_config.get("lora_alpha", float("inf"))),
        "target_modules_match": set(payload.get("peft_target_modules") or [])
        == set(adapter_config.get("target_modules") or []),
        "bias_matches": str(payload.get("peft_bias"))
        == str(adapter_config.get("bias")),
    }
    failed = [
        label
        for label, passed in {**path_checks, **metadata_checks}.items()
        if not passed
    ]
    if failed:
        raise ValueError(f"Archived training config checks failed: {failed}")
    return {
        "path": str(training_config_path),
        "sha256": sha256_file(training_config_path),
        "path_checks": path_checks,
        "lora_metadata_checks": metadata_checks,
    }


def _checkpoint_manifest_binding(
    checkpoint_manifest_path: Path,
    *,
    base: Path,
    merged: Path,
    base_fingerprint: Mapping[str, Any],
    merged_fingerprint: Mapping[str, Any],
    base_artifact_id: str,
    merged_artifact_id: str,
) -> dict[str, Any]:
    manifest = _json_object(
        checkpoint_manifest_path,
        label="checkpoint manifest",
    )
    payload_sha256 = validate_manifest_integrity(manifest)
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("Checkpoint manifest artifacts must be a list")
    artifacts = {
        str(item.get("artifact_id")): item
        for item in raw_artifacts
        if isinstance(item, dict)
    }
    base_record = artifacts.get(base_artifact_id)
    merged_record = artifacts.get(merged_artifact_id)
    if not isinstance(base_record, dict) or not isinstance(merged_record, dict):
        raise ValueError(
            "Checkpoint manifest does not contain both requested artifact IDs"
        )
    assertions = {
        "base_path_matches": Path(str(base_record.get("model_path"))).resolve()
        == base,
        "merged_path_matches": Path(str(merged_record.get("model_path"))).resolve()
        == merged,
        "base_fingerprint_matches": base_record.get("model_sha256")
        == base_fingerprint.get("sha256"),
        "merged_fingerprint_matches": merged_record.get("model_sha256")
        == merged_fingerprint.get("sha256"),
        "merged_verified_parent_matches_base": merged_record.get(
            "verified_parent_artifact_id"
        )
        == base_artifact_id,
    }
    failed = [label for label, passed in assertions.items() if not passed]
    if failed:
        raise ValueError(f"Checkpoint manifest binding checks failed: {failed}")
    return {
        "path": str(checkpoint_manifest_path),
        "sha256": sha256_file(checkpoint_manifest_path),
        "payload_sha256": payload_sha256,
        "base_artifact_id": base_artifact_id,
        "merged_artifact_id": merged_artifact_id,
        "assertions": assertions,
    }


def verify_exact_lora_merge(
    *,
    base_model: str | Path,
    adapter: str | Path,
    merged_model: str | Path,
    training_config: str | Path | None = None,
    checkpoint_manifest: str | Path | None = None,
    base_artifact_id: str = "eval-base",
    merged_artifact_id: str = "eval-analysis-sft",
    fingerprint_sources: bool = True,
) -> dict[str, Any]:
    """Return a sealed, deterministic exact-merge evidence record."""

    base = _directory(base_model, label="base model")
    adapter_path = _directory(adapter, label="LoRA adapter")
    merged = _directory(merged_model, label="merged model")
    adapter_config_path = _file(
        adapter_path / "adapter_config.json",
        label="adapter config",
    )
    adapter_weights_path = _file(
        adapter_path / "adapter_model.safetensors",
        label="adapter weights",
    )
    adapter_config = _json_object(adapter_config_path, label="adapter config")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter_config.peft_type must be LORA")
    if not _path_has_suffix(base, adapter_config.get("base_model_name_or_path")):
        raise ValueError(
            "adapter_config.base_model_name_or_path does not match the supplied base"
        )
    scaling, scaling_formula = _scaling(adapter_config)

    base_index_path, base_map = _weight_map(base)
    merged_index_path, merged_map = _weight_map(merged)
    if set(base_map) != set(merged_map):
        only_base = sorted(set(base_map) - set(merged_map))
        only_merged = sorted(set(merged_map) - set(base_map))
        raise ValueError(
            "Base and merged tensor inventories differ: "
            f"only_base={only_base[:5]}, only_merged={only_merged[:5]}"
        )

    tensor_records: list[dict[str, Any]] = []
    exact_unchanged_count = 0
    exact_adapted_count = 0
    total_unchanged_elements = 0
    total_adapted_elements = 0
    changed_adapted_elements = 0

    with ExitStack() as stack:
        base_handles = {
            shard: stack.enter_context(
                safe_open(base / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(base_map.values()))
        }
        merged_handles = {
            shard: stack.enter_context(
                safe_open(merged / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(merged_map.values()))
        }
        adapter_handle = stack.enter_context(
            safe_open(adapter_weights_path, framework="pt", device="cpu")
        )
        adapter_keys = set(adapter_handle.keys())
        pairs = _adapter_pairs(adapter_keys)
        if not pairs:
            raise ValueError("Adapter contains no standard LoRA tensor pairs")
        missing_targets = sorted(set(pairs) - set(base_map))
        if missing_targets:
            raise ValueError(
                f"Adapter targets missing from base model: {missing_targets[:5]}"
            )

        for tensor_name in sorted(base_map):
            base_tensor = base_handles[base_map[tensor_name]].get_tensor(tensor_name)
            merged_tensor = merged_handles[merged_map[tensor_name]].get_tensor(
                tensor_name
            )
            if base_tensor.shape != merged_tensor.shape:
                raise ValueError(f"Shape differs for tensor {tensor_name}")
            if base_tensor.dtype != merged_tensor.dtype:
                raise ValueError(f"Dtype differs for tensor {tensor_name}")
            elements = base_tensor.numel()
            pair = pairs.get(tensor_name)
            if pair is None:
                if not torch.equal(base_tensor, merged_tensor):
                    changed = int((base_tensor != merged_tensor).sum().item())
                    raise ValueError(
                        "Non-adapted tensor changed in merged model: "
                        f"{tensor_name} ({changed} elements)"
                    )
                exact_unchanged_count += 1
                total_unchanged_elements += elements
                continue

            lora_a = adapter_handle.get_tensor(pair[0])
            lora_b = adapter_handle.get_tensor(pair[1])
            if lora_a.ndim != 2 or lora_b.ndim != 2:
                raise ValueError(f"LoRA pair for {tensor_name} must be rank-2")
            if (
                lora_a.shape[1] != base_tensor.shape[1]
                or lora_b.shape[0] != base_tensor.shape[0]
                or lora_b.shape[1] != lora_a.shape[0]
            ):
                raise ValueError(f"LoRA pair shape is invalid for {tensor_name}")

            expected = (
                base_tensor.float()
                + (lora_b.float() @ lora_a.float()) * scaling
            ).to(merged_tensor.dtype)
            if not torch.equal(expected, merged_tensor):
                mismatch = expected != merged_tensor
                mismatch_count = int(mismatch.sum().item())
                max_abs_error = float(
                    (expected.float() - merged_tensor.float()).abs().max().item()
                )
                raise ValueError(
                    "Adapted tensor does not equal exact PEFT merge expression: "
                    f"{tensor_name}; mismatches={mismatch_count}; "
                    f"max_abs_error={max_abs_error}"
                )
            changed_elements = int((base_tensor != merged_tensor).sum().item())
            exact_adapted_count += 1
            total_adapted_elements += elements
            changed_adapted_elements += changed_elements
            tensor_records.append(
                {
                    "tensor_name": tensor_name,
                    "shape": list(base_tensor.shape),
                    "dtype": str(base_tensor.dtype).removeprefix("torch."),
                    "lora_a_shape": list(lora_a.shape),
                    "lora_b_shape": list(lora_b.shape),
                    "changed_elements_vs_base": changed_elements,
                    "exact_reconstruction": True,
                }
            )

    base_fingerprint = (
        fingerprint_artifact_path(base)
        if fingerprint_sources
        else {"path": str(base), "sha256": None, "algorithm": "not-requested"}
    )
    adapter_fingerprint = (
        fingerprint_artifact_path(adapter_path)
        if fingerprint_sources
        else {
            "path": str(adapter_path),
            "sha256": None,
            "algorithm": "not-requested",
        }
    )
    merged_fingerprint = (
        fingerprint_artifact_path(merged)
        if fingerprint_sources
        else {"path": str(merged), "sha256": None, "algorithm": "not-requested"}
    )

    metadata: dict[str, Any] = {
        "adapter": {
            "base_model_name_or_path": adapter_config.get(
                "base_model_name_or_path"
            ),
            "base_path_matches": True,
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "r": int(adapter_config["r"]),
            "lora_alpha": float(adapter_config["lora_alpha"]),
            "scaling": scaling,
            "scaling_formula": scaling_formula,
            "target_modules": sorted(adapter_config.get("target_modules") or []),
            "bias": adapter_config.get("bias"),
            "use_dora": bool(adapter_config.get("use_dora", False)),
            "use_rslora": bool(adapter_config.get("use_rslora", False)),
        }
    }
    if training_config is not None:
        training_config_path = _file(
            training_config,
            label="archived training config",
        )
        metadata["training_config"] = _training_config_checks(
            training_config_path,
            base=base,
            adapter=adapter_path,
            merged=merged,
            adapter_config=adapter_config,
        )

    bindings: dict[str, Any] = {}
    if checkpoint_manifest is not None:
        if not fingerprint_sources:
            raise ValueError(
                "Checkpoint manifest binding requires source fingerprints"
            )
        checkpoint_manifest_path = _file(
            checkpoint_manifest,
            label="checkpoint manifest",
        )
        bindings["checkpoint_manifest"] = _checkpoint_manifest_binding(
            checkpoint_manifest_path,
            base=base,
            merged=merged,
            base_fingerprint=base_fingerprint,
            merged_fingerprint=merged_fingerprint,
            base_artifact_id=base_artifact_id,
            merged_artifact_id=merged_artifact_id,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "subject_artifact_id": merged_artifact_id,
        "conclusion": "exact_base_plus_adapter_merge_verified",
        "claim_scope": (
            "The supplied merged weight tensors are exactly the supplied base "
            "weight tensors with the supplied LoRA adapter merged using float32 "
            "B@A and addition, cast to each stored merged dtype. This proves the "
            "weight parent; it does not independently prove the training data."
        ),
        "sources": {
            "base_model": base_fingerprint,
            "adapter": adapter_fingerprint,
            "merged_model": merged_fingerprint,
            "critical_files": _critical_files(base, adapter_path, merged),
            "base_weight_index_sha256": sha256_file(base_index_path),
            "merged_weight_index_sha256": sha256_file(merged_index_path),
        },
        "metadata_evidence": metadata,
        "tensor_verification": {
            "device": "cpu",
            "comparison": "torch.equal (exact stored tensor values)",
            "merge_expression": (
                "to_merged_dtype(base.float32 + "
                "(lora_B.float32 @ lora_A.float32) * scaling)"
            ),
            "model_tensor_count": len(base_map),
            "adapter_tensor_count": exact_adapted_count * 2,
            "adapted_model_tensor_count": exact_adapted_count,
            "unchanged_model_tensor_count": exact_unchanged_count,
            "exact_adapted_model_tensor_count": exact_adapted_count,
            "exact_unchanged_model_tensor_count": exact_unchanged_count,
            "mismatch_count": 0,
            "total_adapted_elements": total_adapted_elements,
            "changed_adapted_elements_vs_base": changed_adapted_elements,
            "total_unchanged_elements": total_unchanged_elements,
            "adapted_tensors": tensor_records,
        },
        "bindings": bindings,
    }
    return seal_manifest(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only exact tensor proof that a merged model equals a base plus "
            "a standard LoRA adapter."
        )
    )
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--merged-model", required=True, type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--base-artifact-id", default="eval-base")
    parser.add_argument("--merged-artifact-id", default="eval-analysis-sft")
    parser.add_argument(
        "--output",
        type=Path,
        help="Independent sealed evidence JSON; requires --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow creation of --output. Source artifacts remain read-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and args.output is None:
        raise ValueError("--execute requires --output")
    if args.output is not None and not args.execute:
        raise ValueError("--output requires --execute")
    evidence = verify_exact_lora_merge(
        base_model=args.base_model,
        adapter=args.adapter,
        merged_model=args.merged_model,
        training_config=args.training_config,
        checkpoint_manifest=args.checkpoint_manifest,
        base_artifact_id=args.base_artifact_id,
        merged_artifact_id=args.merged_artifact_id,
    )
    print(f"conclusion={evidence['conclusion']}")
    print(
        "payload_sha256="
        f"{evidence['integrity']['payload_sha256']}"
    )
    if args.output is not None:
        output_sha256 = write_immutable_json(args.output, evidence)
        print(f"output={args.output.expanduser().resolve()}")
        print(f"output_sha256={output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
