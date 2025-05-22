import json
from collections import defaultdict
import matplotlib.pyplot as plt


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
            if record.get("reasoning_reward") is not None:
                step_data[step]["reasoning_reward"] = record["reasoning_reward"]
            if record.get("answer_reward") is not None:
                step_data[step]["answer_reward"] = record["answer_reward"]
            if record.get("eval_reward") is not None:
                step_data[step]["eval_reward"] = record["eval_reward"]

    # 排序
    steps = sorted(step_data.keys())

    reasoning_rewards = [step_data[s]["reasoning_reward"] for s in steps]
    answer_rewards = [step_data[s]["answer_reward"] for s in steps]
    eval_rewards = [step_data[s]["eval_reward"] for s in steps]

    # 过滤 None
    reasoning_steps = [s for s, r in zip(steps, reasoning_rewards) if r is not None]
    reasoning_values = [r for r in reasoning_rewards if r is not None]

    answer_steps = [s for s, r in zip(steps, answer_rewards) if r is not None]
    answer_values = [r for r in answer_rewards if r is not None]

    eval_steps = [s for s, r in zip(steps, eval_rewards) if r is not None]
    eval_values = [r for r in eval_rewards if r is not None]

    # 构造 combined_rewards 并过滤 None
    combined_steps = []
    combined_values = []
    for s, r, a in zip(steps, reasoning_rewards, answer_rewards):
        if r is not None and a is not None:
            combined_steps.append(s)
            combined_values.append((r + a)/2)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(combined_steps, combined_values, label="Reward", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("GRPO Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    plot_reward_curve("./input/reward_history_20250517.jsonl", "./output/reward_curve_20250517.png")