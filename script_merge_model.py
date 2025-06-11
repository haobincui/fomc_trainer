import shutil
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel


def clean_adapter_config(adapter_path: Path, keys_to_remove=None):
    if keys_to_remove is None:
        keys_to_remove = [
            'corda_config', 'eva_config', 'exclude_modules', 
            'lora_bias', 'trainable_token_indices'
        ]
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        with config_file.open("r") as f:
            config = json.load(f)
        for key in keys_to_remove:
            config.pop(key, None)
        with config_file.open("w") as f:
            json.dump(config, f)
        print("🧹 Cleaned adapter_config.json")
    else:
        print("⚠️ adapter_config.json not found; skipping config cleaning.")


def merge_model(base_model_path, adapter_path, merged_path):
    print("📦 Loading base model:", base_model_path)
    print("🔗 Loading adapter from:", adapter_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    Path(merged_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)
    print("✅ Merged model saved to:", merged_path)


def copy_files(adapter_path: Path, merged_path: Path, files_to_copy):
    for filename in files_to_copy:
        src_file = adapter_path / filename
        dst_file = merged_path / filename
        try:
            shutil.copy(src_file, dst_file)
            print(f"📄 Overwritten {filename}")
        except FileNotFoundError:
            print(f"⚠️ Skipped {filename} (not found)")


def main(base_model_name_or_path, merged_path, adapter_path):
    adapter_path = Path(adapter_path)
    merged_path = Path(merged_path)

    print("🚀 Starting model merge and file export...")
    print("📂 Output directory:", merged_path)

    clean_adapter_config(adapter_path)

    merge_model(base_model_name_or_path, adapter_path, merged_path)

    files_to_copy = [
        "eval_results.json",
        "loss_history.jsonl",
        "train_results.json",
        "trainer_state.json",
        "training_args.bin",
        "training_curve.png",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "README.md"
    ]

    copy_files(adapter_path, merged_path, files_to_copy)

    print(f"🎉 All done! Model and resources exported to {merged_path}")


if __name__ == '__main__':
    base_model_name_or_path = "output/merged/llama_sft_20250522"
    merged_path = "output/merged/llama_grpo_decision_cp1100_20250530"
    adapter_path = "output/adapters/llama_grpo_decision_20250529/checkpoint-1100"

    main(base_model_name_or_path, merged_path, adapter_path)
