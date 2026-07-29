import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_with_rolling_average(json_file_path, output_path, window_size=10):
    """trainer_state.json"""
    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract log_history entries
    log_history = data.get("log_history", [])

    # Extract steps and rewards
    steps = [entry["step"] for entry in log_history if "step" in entry and "reward" in entry]
    rewards = [entry["reward"] for entry in log_history if "step" in entry and "reward" in entry]

    # Build DataFrame
    df = pd.DataFrame({"step": steps, "reward": rewards})
    df = df.sort_values("step")
    df["rolling_reward"] = df["reward"].rolling(window=window_size).mean()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["reward"], color='lightgray', label="Original Reward", alpha=0.6)
    plt.plot(df["step"], df["rolling_reward"], color='blue', label=f"Rolling Avg (window={window_size})", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("Reward vs Training Step with Rolling Average")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_reward_with_std(json_file_path, output_file, window_size=10):
    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract log_history entries
    log_history = data.get("log_history", [])

    # Extract steps and rewards
    steps = [entry["step"] for entry in log_history if "step" in entry and "reward" in entry]
    rewards = [entry["reward"] for entry in log_history if "step" in entry and "reward" in entry]

    # Build DataFrame
    df = pd.DataFrame({"step": steps, "reward": rewards})
    df = df.sort_values("step")
    df["rolling_mean"] = df["reward"].rolling(window=window_size).mean()
    df["rolling_std"] = df["reward"].rolling(window=window_size).std()

    # Plot with shaded standard deviation band
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["rolling_mean"], color='blue', label=f"Rolling Mean Reward (window={window_size})", linewidth=2)
    plt.fill_between(df["step"],
                     df["rolling_mean"] - df["rolling_std"],
                     df["rolling_mean"] + df["rolling_std"],
                     color='blue', alpha=0.2, label="±1 Std Dev")

    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title(f"{window_size}-Step Rolling Average Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def plot_custom_reward_with_std(json_file_path, output_file, reward_key="reward", window_size=10):
    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract log_history entries
    log_history = data.get("log_history", [])

    # Extract steps and the specified reward
    steps = [entry["step"] for entry in log_history if "step" in entry and reward_key in entry]
    rewards = [entry[reward_key] for entry in log_history if "step" in entry and reward_key in entry]

    if not steps or not rewards:
        print(f"❌ No data found for reward key '{reward_key}'")
        return

    # Build DataFrame
    df = pd.DataFrame({"step": steps, reward_key: rewards})
    df = df.sort_values("step")
    df["rolling_mean"] = df[reward_key].rolling(window=window_size).mean()
    df["rolling_std"] = df[reward_key].rolling(window=window_size).std()

    # Plot with shaded standard deviation band
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["rolling_mean"], color='blue', label=f"Rolling Mean ({reward_key})", linewidth=2)
    plt.fill_between(df["step"],
                     df["rolling_mean"] - df["rolling_std"],
                     df["rolling_mean"] + df["rolling_std"],
                     color='blue', alpha=0.2, label="±1 Std Dev")

    plt.xlabel("Training Step")
    plt.ylabel(reward_key.replace("_", " ").title())
    plt.title(f"{window_size}-Step Rolling Average of {reward_key.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def plot_multiple_rewards(json_file_path, output_file, reward_keys, window_size=10):
    import json
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract log_history entries
    log_history = data.get("log_history", [])

    # Initialize DataFrame with step
    df = pd.DataFrame([entry for entry in log_history if "step" in entry])
    df = df.sort_values("step")

    # Compute rolling mean for each reward key
    for key in reward_keys:
        if key in df.columns:
            df[f"rolling_{key}"] = df[key].rolling(window=window_size).mean()
        else:
            print(f"⚠️ Key '{key}' not found in log entries, skipping.")

    # Plot
    plt.figure(figsize=(10, 5))
    for key in reward_keys:
        col = f"rolling_{key}"
        if col in df.columns:
            plt.plot(df["step"], df[col], label=f"{key} (rolling)", linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title(f"Rolling Average of Rewards (window={window_size})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(output_file)
    plt.show()


# Example usage:
# plot_multiple_rewards("trainer_state.json", "multi_reward_plot.png", ["answer_reward", "reasoning_reward"], window_size=10)

rename_map = {
    "rewards/answer_reward/mean": "Accuracy Reward",
    "rewards/reasoning_reward/mean": "Reasoning Reward",
    "reward": "Total Reward",
    "rewards/rate_accuracy_reward/mean": "Vote Accuracy Reward",
    "rewards/rate_format_reward/mean": "Vote Format Reward",
}

def plot_multiple_rewards_with_std(json_file_path, output_file, reward_keys, window_size=10):
    import json
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Extract log_history entries
    log_history = data.get("log_history", [])
    df = pd.DataFrame([entry for entry in log_history if "step" in entry])
    df = df.sort_values("step")
    df.rename(columns=rename_map, inplace=True)
    reward_keys = [rename_map.get(key, key) for key in reward_keys]

    # Plot setup
    plt.figure(figsize=(10, 6))

    for key in reward_keys:
        if key not in df.columns:
            print(f"⚠️ Key '{key}' not found, skipping.")
            continue


        df[f"{key}_mean"] = df[key].rolling(window=window_size).mean()
        df[f"{key}_std"] = df[key].rolling(window=window_size).std()

        # Plot rolling mean and std dev band
        plt.plot(df["step"], df[f"{key}_mean"], linewidth=2, label=f"{key}")
        plt.fill_between(df["step"],
                         df[f"{key}_mean"] - df[f"{key}_std"],
                         df[f"{key}_mean"] + df[f"{key}_std"],
                         alpha=0.2)

    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title(f"{window_size}-Step Rolling Average Reward with 1-standard Deviation Interval")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Example usage:
# plot_multiple_rewards_with_std("trainer_state.json", "rolling_std_rewards.png", ["answer_reward", "reasoning_reward"], window_size=10)



if __name__ == '__main__':
    # input = "input/trainer_state_grpo_stage2_800_20250527.json"
    # output = "./output/trainer_state_grpo_stage2_800_20250527.png"
    input = "input/trainer_state_grpo_stage1_20250517.json"
    output = "./output/trainer_state_grpo_stage1_20250517.png"
    # plot_reward_with_rolling_average(input, output, window_size=10)
    # plot_reward_with_std(input,output,  window_size=20)
    # plot_custom_reward_with_std(input, output, reward_key="rewards/answer_reward/mean", window_size=20)
    # plot_multiple_rewards(input, output, reward_keys=["rewards/answer_reward/mean", "rewards/reasoning_reward/mean", "reward"], window_size=20),
    # plot_multiple_rewards_with_std(input, output,
    #                                reward_keys=["rewards/answer_reward/mean",
    #                                             "rewards/reasoning_reward/mean",
    #                                             "reward"], window_size=20)

    # input = "input/trainer_state_grpo_stage2_1900_20250527.json"
    # output = "./output/trainer_state_grpo_stage2_1900_20250527.png"
    input = "input/trainer_state_grpo_stage2_cp1100_20250530.json"
    output = "./output/trainer_state_grpo_stage2_cp_110020250530.png"
    plot_multiple_rewards_with_std(input, output,
                                   reward_keys=["rewards/rate_accuracy_reward/mean",
                                                "rewards/rate_format_reward/mean",
                                                "reward"], window_size=20)

    # step = 1200


