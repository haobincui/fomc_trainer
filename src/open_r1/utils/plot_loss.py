import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def plot_training_curve(jsonl_path, save_path):
    """
    从 loss_history.jsonl 中读取 loss，合并相同 step，绘制训练曲线。
    """
    step_data = defaultdict(lambda: {"train_loss": None, "eval_loss": None})

    # 读取并合并
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            step = record.get("step")
            if step is None:
                continue
            if record.get("train_loss") is not None:
                step_data[step]["train_loss"] = record["train_loss"]
            if record.get("eval_loss") is not None:
                step_data[step]["eval_loss"] = record["eval_loss"]

    # 排序
    steps = sorted(step_data.keys())
    train_losses = [step_data[s]["train_loss"] for s in steps]
    eval_losses = [step_data[s]["eval_loss"] for s in steps]

    # 去掉 None（只保留有值的点）
    train_steps = [s for s, l in zip(steps, train_losses) if l is not None]
    train_values = [l for l in train_losses if l is not None]

    eval_steps = [s for s, l in zip(steps, eval_losses) if l is not None]
    eval_values = [l for l in eval_losses if l is not None]

    # 绘图
    plt.figure(figsize=(10,6))
    plt.plot(train_steps, train_values, label="Train Loss", marker='o')
    plt.plot(eval_steps, eval_values, label="Eval Loss", marker='x')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Evaluation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_reward_curve(jsonl_path, save_path):
    """
    从 reward_log.jsonl 中读取 reward，合并相同 step，绘制 reward 曲线。
    """
    step_data = defaultdict(lambda: {"reward": None, "eval_reward": None})

    # 读取并合并
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            step = record.get("step")
            if step is None:
                continue
            # 兼容含有"reward"和"eval_reward"字段的日志
            if record.get("reward") is not None:
                step_data[step]["reward"] = record["reward"]
            if record.get("eval_reward") is not None:
                step_data[step]["eval_reward"] = record["eval_reward"]

    # 排序
    steps = sorted(step_data.keys())
    rewards = [step_data[s]["reward"] for s in steps]
    eval_rewards = [step_data[s]["eval_reward"] for s in steps]

    # 去掉 None（只保留有值的点）
    train_steps = [s for s, r in zip(steps, rewards) if r is not None]
    train_values = [r for r in rewards if r is not None]

    eval_steps = [s for s, r in zip(steps, eval_rewards) if r is not None]
    eval_values = [r for r in eval_rewards if r is not None]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_values, label="Reward", marker='o')
    if len(eval_steps) > 0:
        plt.plot(eval_steps, eval_values, label="Eval Reward", marker='x')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training & Evaluation Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
