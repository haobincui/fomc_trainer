"""Non-destructively merge a retrain_v2 LoRA adapter into its sealed parent."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from jobs.retrain_v2.merge_attestation import create_merge_attestation
from open_r1.provenance import fingerprint_artifact_path


PEFT_COMPATIBILITY_KEYS = {
    "corda_config",
    "eva_config",
    "exclude_modules",
    "lora_bias",
    "trainable_token_indices",
}
PEFT_COMPATIBILITY_DEFAULTS = {
    "corda_config": None,
    "eva_config": None,
    "exclude_modules": None,
    "lora_bias": False,
    "trainable_token_indices": None,
}
METADATA_FILES = {
    "eval_results.json",
    "loss_history.jsonl",
    "reward_history.jsonl",
    "reward.jsonl",
    "resolved_runtime_config.json",
    "train_results.json",
    "trainer_state.json",
    "training_args.bin",
    "training_curve.png",
}


def _load_paths(
    config_path: Path, *, repo_root: str | Path | None = None
) -> tuple[Path, Path, Path]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")
    values = []
    for key in ("model_name_or_path", "output_dir", "peft_merged_model_path"):
        value = payload.get(key)
        if not value:
            raise ValueError(f"Training config is missing {key}: {config_path}")
        candidate = Path(str(value))
        if not candidate.is_absolute() and repo_root is not None:
            candidate = Path(repo_root) / candidate
        values.append(candidate.resolve())
    base, adapter, destination = values
    if len({base, adapter, destination}) != 3:
        raise ValueError("Base, adapter, and merged destination must be distinct paths")
    return base, adapter, destination


def _sanitize_adapter_copy(adapter_copy: Path) -> None:
    config_path = adapter_copy / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Adapter config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter config must contain an object: {config_path}")
    for key, expected in PEFT_COMPATIBILITY_DEFAULTS.items():
        if payload.get(key, expected) != expected:
            raise ValueError(
                f"Adapter {key} uses unsupported non-default merge semantics"
            )
    for key in PEFT_COMPATIBILITY_KEYS:
        payload.pop(key, None)
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_no_meta_tensors(model: Any, *, phase: str) -> None:
    """Fail before a merge can copy or publish an unmaterialized tensor."""

    meta_tensors = [
        *(
            f"parameter:{name}"
            for name, tensor in model.named_parameters()
            if tensor.is_meta
        ),
        *(f"buffer:{name}" for name, tensor in model.named_buffers() if tensor.is_meta),
    ]
    if meta_tensors:
        preview = ", ".join(meta_tensors[:8])
        if len(meta_tensors) > 8:
            preview += f", ... (+{len(meta_tensors) - 8} more)"
        raise RuntimeError(
            f"Merge {phase} contains {len(meta_tensors)} unmaterialized meta "
            f"tensor(s): {preview}"
        )


def _merge_model(base: Path, adapter_copy: Path, output: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    _assert_no_meta_tensors(base_model, phase="base load")
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    # PeftModel otherwise turns the base model's all-CPU hf_device_map into an
    # implicit device_map="auto".  That makes merge topology depend on current
    # GPU free memory and can create CPU-offloaded meta tensors.
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_copy),
        device_map={"": "cpu"},
        local_files_only=True,
    )
    _assert_no_meta_tensors(peft_model, phase="pre-merge adapter load")
    lora_modules_before = sum(
        1 for module in peft_model.modules() if isinstance(module, LoraLayer)
    )
    if lora_modules_before <= 0:
        raise RuntimeError("Loaded adapter exposes no LoRA modules to merge")
    merged_model = peft_model.merge_and_unload()
    _assert_no_meta_tensors(merged_model, phase="post-merge unload")
    residual_lora_modules = sum(
        1 for module in merged_model.modules() if isinstance(module, LoraLayer)
    )
    residual_lora_state_keys = sum(
        1
        for name in merged_model.state_dict()
        if ".lora_A." in name or ".lora_B." in name
    )
    parameter_tensors = 0
    parameter_count = 0
    for parameter in merged_model.parameters():
        parameter_tensors += 1
        parameter_count += parameter.numel()
    output.mkdir(parents=True, exist_ok=False)
    merged_model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output)
    from jobs.retrain_v2.merge_attestation import (
        SEMANTIC_BOUNDARY,
        SEMANTIC_METHOD,
    )

    return {
        "method": SEMANTIC_METHOD,
        "merge_completed": True,
        "lora_modules_before": lora_modules_before,
        "residual_lora_modules_after": residual_lora_modules,
        "residual_lora_state_keys_after": residual_lora_state_keys,
        "merged_parameter_tensors": parameter_tensors,
        "merged_parameter_count": parameter_count,
        "functional_equivalence_boundary": SEMANTIC_BOUNDARY,
    }


def _fsync_tree(root: Path) -> None:
    """Make every staged merge byte durable before the atomic publish."""

    for directory, subdirectories, filenames in os.walk(root, topdown=False):
        current = Path(directory)
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Merge staging tree contains an unsafe entry: {path}")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in subdirectories:
            path = current / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"Merge staging tree contains an unsafe entry: {path}")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a staged merge without clobbering a raced target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "atomic renameat2(RENAME_NOREPLACE) is unavailable; refusing to publish"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(
            f"Merged destination already exists; overwrite is forbidden: {destination}"
        )
    raise RuntimeError(
        "Atomic no-overwrite merge publication failed: "
        f"{destination}: {os.strerror(error)}"
    )


def merge_from_config(
    config_path: str | Path,
    *,
    merge_executor: Callable[[Path, Path, Path], Mapping[str, Any] | None] = _merge_model,
    merge_attestation_binding: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    config = Path(config_path).resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Resolved training config is missing: {config}")
    base, adapter, destination = _load_paths(config, repo_root=repo_root)
    if not base.is_dir():
        raise FileNotFoundError(f"Sealed parent model is missing: {base}")
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter output is missing: {adapter}")
    if destination.exists():
        raise FileExistsError(f"Merged destination already exists; overwrite is forbidden: {destination}")

    adapter_before = fingerprint_artifact_path(adapter)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.merge-", dir=destination.parent)
    )
    adapter_copy = workspace / "adapter-copy"
    output = workspace / "model"
    published = False
    try:
        shutil.copytree(adapter, adapter_copy)
        _sanitize_adapter_copy(adapter_copy)
        semantic_evidence = merge_executor(base, adapter_copy, output)
        if not output.is_dir():
            raise RuntimeError("Merge executor did not create a model directory")
        for filename in sorted(METADATA_FILES):
            source = adapter / filename
            if source.is_file():
                shutil.copy2(source, output / filename)

        attestation = None
        if merge_attestation_binding is not None:
            if not isinstance(semantic_evidence, Mapping):
                raise RuntimeError(
                    "Attested merge executor returned no semantic evidence"
                )
            attestation = create_merge_attestation(
                output,
                binding=merge_attestation_binding,
                semantic_evidence=semantic_evidence,
            )

        adapter_after = fingerprint_artifact_path(adapter)
        if adapter_after["sha256"] != adapter_before["sha256"]:
            raise RuntimeError("Source adapter changed during merge; refusing to publish")
        _fsync_tree(output)
        _rename_noreplace(output, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        published = True
        result = {
            "status": "merged",
            "config": str(config),
            "base_model": str(base),
            "source_adapter": adapter_before,
            "merged_artifact": fingerprint_artifact_path(destination),
            "source_adapter_unchanged": True,
            "merge_attestation": attestation,
        }
    except Exception:
        # Remove only an artifact this invocation actually published.  If a
        # concurrent process created the destination after our initial check,
        # it does not belong to us and must never be deleted here.
        if published and destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("chk1", "chk2", "chk3", "chk4"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from jobs.retrain_v2.execution_receipt import merge_and_record_receipt

        result = merge_and_record_receipt(
            args.run_manifest, args.stage, args.repo_root
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
