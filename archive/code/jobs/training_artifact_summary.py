import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint3_summary() -> dict:
    train_results = _load_json(Path("output/merged/llama_sft_synthetic_20250526/train_results.json"))
    eval_results = _load_json(Path("output/merged/llama_sft_synthetic_20250526/eval_results.json"))
    trainer_state = _load_json(Path("output/merged/llama_sft_synthetic_20250526/trainer_state.json"))
    log_history = trainer_state["log_history"]
    first_log = log_history[0]
    final_eval = next(item for item in reversed(log_history) if "eval_loss" in item)

    return {
        "checkpoint": "Checkpoint-3",
        "artifact_path": "output/merged/llama_sft_synthetic_20250526",
        "epochs": trainer_state["epoch"],
        "global_step": trainer_state["global_step"],
        "first_logged_train_loss": first_log["loss"],
        "first_logged_step": first_log["step"],
        "first_logged_eval_loss": next(item["eval_loss"] for item in log_history if "eval_loss" in item),
        "final_train_loss": train_results["train_loss"],
        "final_eval_loss": eval_results["eval_loss"],
        "final_eval_step": final_eval["step"],
        "final_logged_eval_loss": final_eval["eval_loss"],
        "train_runtime_seconds": train_results["train_runtime"],
        "eval_runtime_seconds": eval_results["eval_runtime"],
        "figure": "docs/Chapter2/Chapter2Figs/training_curve_synthetic_20250526.png",
    }


def _checkpoint4_summary() -> dict:
    merged_trainer_state = _load_json(Path("output/merged/llama_grpo_decision_cp1100_20250530/trainer_state.json"))
    source_run = _load_json(Path("output/adapters/llama_grpo_decision_20250529/trainer_state.json"))
    last_log = merged_trainer_state["log_history"][-1]

    return {
        "checkpoint": "Checkpoint-4",
        "artifact_path": "output/merged/llama_grpo_decision_cp1100_20250530",
        "source_adapter_checkpoint": "output/adapters/llama_grpo_decision_20250529/checkpoint-1100",
        "merged_checkpoint_step": merged_trainer_state["global_step"],
        "merged_checkpoint_epoch": merged_trainer_state["epoch"],
        "reward_at_merged_checkpoint": last_log["reward"],
        "rate_accuracy_reward_at_merged_checkpoint": last_log["rewards/rate_accuracy_reward/mean"],
        "rate_format_reward_at_merged_checkpoint": last_log["rewards/rate_format_reward/mean"],
        "kl_at_merged_checkpoint": last_log["kl"],
        "mean_completion_length_at_merged_checkpoint": last_log["completions/mean_length"],
        "full_run_epochs": source_run["epoch"],
        "full_run_global_step": source_run["global_step"],
        "full_run_train_loss": source_run["log_history"][-1]["train_loss"],
        "figure_candidates": [
            "docs/Chapter2/Chapter2Figs/stage2_train/trainer_state_grpo_stage2_850_20250530.png",
            "docs/Chapter2/Chapter2Figs/stage2_train/trainer_state_grpo_stage2_1900_20250527.png",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize archived Chapter 2 training artifacts.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/chapter2/training_artifact_summary.json"),
    )
    args = parser.parse_args()

    payload = {
        "checkpoint3_alignment_sft": _checkpoint3_summary(),
        "checkpoint4_decision_grpo": _checkpoint4_summary(),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n## Training Artifact Summary")
    for key, summary in payload.items():
        print(f"- {key}: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
