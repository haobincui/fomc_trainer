from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


FILES_TO_COPY = [
    "eval_results.json",
    "loss_history.jsonl",
    "train_results.json",
    "trainer_state.json",
    "training_args.bin",
    "training_curve.png",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "README.md",
]


def clean_adapter_config(adapter_path: Path, keys_to_remove: list[str] | None = None) -> None:
    if keys_to_remove is None:
        keys_to_remove = [
            "corda_config",
            "eva_config",
            "exclude_modules",
            "lora_bias",
            "trainable_token_indices",
        ]

    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        print("⚠️ adapter_config.json not found; skipping config cleaning.")
        return

    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    for key in keys_to_remove:
        config.pop(key, None)

    with config_file.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("🧹 Cleaned adapter_config.json")


def merge_model(base_model_path: str, adapter_path: Path, merged_path: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("📦 Loading base model:", base_model_path)
    print("🔗 Loading adapter from:", adapter_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print("✅ Merged model and tokenizer saved to:", merged_path)


def copy_files(adapter_path: Path, merged_path: Path, files_to_copy: list[str]) -> None:
    for filename in files_to_copy:
        src_file = adapter_path / filename
        dst_file = merged_path / filename
        if not src_file.exists():
            print(f"⚠️ Skipped {filename} (not found)")
            continue
        shutil.copy(src_file, dst_file)
        print(f"📄 Copied {filename}")


def load_paths_from_config(config_path: Path) -> tuple[str, Path, Path]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_model = config.get("model_name_or_path")
    adapter_path = config.get("output_dir")
    merged_path = config.get("peft_merged_model_path")

    missing = [
        key
        for key, value in (
            ("model_name_or_path", base_model),
            ("output_dir", adapter_path),
            ("peft_merged_model_path", merged_path),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required keys in config {config_path}: {missing}")

    return str(base_model), Path(str(adapter_path)), Path(str(merged_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter directory into a standalone model directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Training config YAML containing model_name_or_path, output_dir, and peft_merged_model_path.",
    )
    parser.add_argument("--base-model", help="Base model path. Required when --config is not used.")
    parser.add_argument("--adapter-path", type=Path, help="Adapter output directory. Required when --config is not used.")
    parser.add_argument("--merged-path", type=Path, help="Merged model output directory. Required when --config is not used.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    if args.config:
        return load_paths_from_config(args.config)

    if args.base_model and args.adapter_path and args.merged_path:
        return args.base_model, args.adapter_path, args.merged_path

    raise ValueError("Use --config or provide --base-model, --adapter-path, and --merged-path together.")


def main() -> None:
    args = parse_args()
    base_model_name_or_path, adapter_path, merged_path = resolve_paths(args)

    print("🚀 Starting model merge and file export...")
    print("📂 Output directory:", merged_path)

    clean_adapter_config(adapter_path)
    merge_model(base_model_name_or_path, adapter_path, merged_path)
    copy_files(adapter_path, merged_path, FILES_TO_COPY)

    print(f"🎉 All done! Model and resources exported to {merged_path}")


if __name__ == "__main__":
    main()
