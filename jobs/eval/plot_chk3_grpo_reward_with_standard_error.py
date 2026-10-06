"""Create the paper-facing Model chk-3 GRPO reward mean-plus-SE figure."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / (
    "output/training/retrain_v2/"
    "chk4_from_pre2009_cp38_direct_grpo_full_no_smoke_v1_20260812/"
    "adapters/chk4_grpo"
)
TRAINER_STATE = RUN_ROOT / "trainer_state.json"
REWARD_ROWS = RUN_ROOT / "reward.jsonl"
OUTPUT_STEM = ROOT / (
    "docs/Chapter2/Chapter2Figs/chk3_grpo_reward_with_standard_error"
)
EXPECTED_TRAINER_STATE_SHA256 = (
    "1ff7297e3ecd84c55c63a505646ac7c8e52412928d82f5b12465ea5998f9f387"
)
EXPECTED_REWARD_ROWS_SHA256 = (
    "123491ea525bef4baf6fc1aeb7d7ff3f216be8a6f8ad0b606aee46ebfdd031f4"
)
EXPECTED_STEPS = 468
COMPLETIONS_PER_STEP = 8
EPOCH_STEPS = 156
EVALUATED_STEP = 450
ROLLING_WINDOW = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"source binding drift: {path}: {observed}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON object: {path}")
    return value


def read_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"source binding drift: {path}: {observed}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"invalid JSONL rows: {path}")
    return rows


def trailing_step_clustered_mean_and_standard_error(
    step_means: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize each window using optimizer-step means as cluster units."""
    if step_means.ndim != 1:
        raise RuntimeError("expected a one-dimensional series of step-level means")
    if window < 2 or window > step_means.shape[0]:
        raise RuntimeError("invalid trailing window")
    means: list[float] = []
    standard_errors: list[float] = []
    for stop in range(window, step_means.shape[0] + 1):
        cluster_means = step_means[stop - window : stop]
        means.append(float(cluster_means.mean()))
        standard_errors.append(
            float(cluster_means.std(ddof=1) / np.sqrt(cluster_means.size))
        )
    steps = np.arange(window, step_means.shape[0] + 1, dtype=np.int64)
    return steps, np.asarray(means), np.asarray(standard_errors)


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    state = read_json(TRAINER_STATE, EXPECTED_TRAINER_STATE_SHA256)
    reward_rows = read_jsonl(REWARD_ROWS, EXPECTED_REWARD_ROWS_SHA256)
    history = state.get("log_history")
    if not isinstance(history, list):
        raise RuntimeError("trainer state has no log_history")
    training_rows = [row for row in history if "reward" in row]

    if int(state.get("global_step", -1)) != EXPECTED_STEPS:
        raise RuntimeError("expected the complete 468-step GRPO trajectory")
    if float(state.get("epoch", -1.0)) != 3.0:
        raise RuntimeError("expected the complete three-epoch GRPO trajectory")
    if len(training_rows) != EXPECTED_STEPS:
        raise RuntimeError("expected one trainer reward record per step")
    if len(reward_rows) != EXPECTED_STEPS * COMPLETIONS_PER_STEP:
        raise RuntimeError("expected eight completion-level reward rows per step")

    steps = np.asarray([int(row["step"]) for row in training_rows], dtype=np.int64)
    if not np.array_equal(steps, np.arange(1, EXPECTED_STEPS + 1)):
        raise RuntimeError("trainer reward steps are not contiguous")

    completion_rewards = np.asarray(
        [float(row["reward"]) for row in reward_rows], dtype=np.float64
    ).reshape(EXPECTED_STEPS, COMPLETIONS_PER_STEP)
    reward_mean = completion_rewards.mean(axis=1)
    reward_std = completion_rewards.std(axis=1, ddof=1)
    trainer_mean = np.asarray([float(row["reward"]) for row in training_rows])
    trainer_std = np.asarray([float(row["reward_std"]) for row in training_rows])
    if not np.allclose(reward_mean, trainer_mean, atol=5e-7, rtol=0.0):
        raise RuntimeError("completion-level means do not reproduce trainer reward")
    if not np.allclose(reward_std, trainer_std, atol=5e-7, rtol=0.0):
        raise RuntimeError("completion-level SDs do not reproduce trainer reward_std")

    smooth_steps, reward_smooth, reward_smooth_se = (
        trailing_step_clustered_mean_and_standard_error(reward_mean, ROLLING_WINDOW)
    )
    band_lower = reward_smooth - reward_smooth_se
    band_upper = reward_smooth + reward_smooth_se

    epoch_summaries: list[dict[str, Any]] = []
    for epoch_index in range(3):
        start = epoch_index * EPOCH_STEPS
        stop = (epoch_index + 1) * EPOCH_STEPS
        values = completion_rewards[start:stop].reshape(-1)
        step_values = reward_mean[start:stop]
        epoch_summaries.append(
            {
                "epoch": epoch_index + 1,
                "step_start": start + 1,
                "step_end": stop,
                "completion_reward_count": int(values.size),
                "reward_mean": float(values.mean()),
                "reward_sample_std": float(values.std(ddof=1)),
                "mean_within_step_sample_std": float(reward_std[start:stop].mean()),
                "step_clustered_standard_error": float(
                    step_values.std(ddof=1) / np.sqrt(step_values.size)
                ),
            }
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#27313d",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
        }
    )
    fig, ax_reward = plt.subplots(
        1,
        1,
        figsize=(7.6, 5.1),
        dpi=200,
        facecolor="white",
    )

    ax_reward.set_facecolor("white")
    ax_reward.grid(axis="y", color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax_reward.grid(axis="x", visible=False)
    ax_reward.spines["top"].set_visible(False)
    ax_reward.spines["right"].set_visible(False)
    ax_reward.spines["left"].set_color("#9ca3af")
    ax_reward.spines["bottom"].set_color("#9ca3af")
    for boundary in (EPOCH_STEPS, 2 * EPOCH_STEPS):
        ax_reward.axvline(
            boundary,
            color="#a3aab3",
            linewidth=0.9,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
    ax_reward.axvline(
        EVALUATED_STEP,
        color="#4b5563",
        linewidth=1.1,
        linestyle=(0, (5, 3)),
        zorder=4,
    )

    ax_reward.fill_between(
        smooth_steps,
        band_lower,
        band_upper,
        color="#6f9ecb",
        alpha=0.27,
        linewidth=0.0,
        label=(
            f"Mean $\\pm$ 1 step-level SE "
            f"({ROLLING_WINDOW}-step window)"
        ),
        zorder=1,
    )
    ax_reward.plot(
        smooth_steps,
        reward_smooth,
        color="#2f5f98",
        linewidth=2.15,
        label=f"Reward mean ({ROLLING_WINDOW}-step trailing window)",
        zorder=2,
    )
    ax_reward.set_ylim(-0.02, 1.03)
    ax_reward.set_ylabel("Reward")
    ax_reward.set_xlabel("Optimiser step")
    ax_reward.legend(
        loc="upper left",
        frameon=True,
        fontsize=7.8,
        handlelength=2.6,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    ax_reward.set_xlim(0, EXPECTED_STEPS + 4)
    # cp450 is identified by its vertical annotation; omitting it from the tick
    # labels prevents a collision with the terminal step 468.
    ax_reward.set_xticks([0, 78, 156, 234, 312, 390, 468])
    ax_reward.annotate(
        "evaluated exploratory cp450",
        xy=(EVALUATED_STEP, 0.935),
        xytext=(EVALUATED_STEP - 8, 0.935),
        ha="right",
        va="center",
        fontsize=8.0,
        color="#4b5563",
    )

    epoch_text = "   |   ".join(
        (
            f"E{row['epoch']} {row['reward_mean']:.3f} $\\pm$ "
            f"{row['step_clustered_standard_error']:.3f}"
        )
        for row in epoch_summaries
    )
    fig.suptitle(
        "Model chk-3 GRPO: Reward Mean with $\\pm$1 SE",
        x=0.085,
        y=0.972,
        ha="left",
        fontsize=14.5,
        fontweight="semibold",
        color="#18212b",
    )
    fig.text(
        0.085,
        0.918,
        (
            "20-step trailing reward summaries; eight completions per optimiser step"
        ),
        ha="left",
        fontsize=8.9,
        color="#59636e",
    )
    fig.text(
        0.085,
        0.884,
        "Epoch reward mean $\\pm$ step-level SE: " + epoch_text,
        ha="left",
        fontsize=8.3,
        color="#59636e",
    )
    fig.text(
        0.085,
        0.023,
        (
            "Note: SE = SD of 20 optimiser-step means / $\\sqrt{20}$; each mean averages 8 completions.\n"
            "Descriptive only: no adjustment for serial, overlapping-window, or between-run dependence; "
            "not a confidence interval."
        ),
        ha="left",
        fontsize=7.3,
        color="#68727d",
    )
    fig.subplots_adjust(left=0.115, right=0.98, top=0.82, bottom=0.18)

    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    manifest_path = OUTPUT_STEM.with_suffix(".manifest.json")
    for path in (png_path, pdf_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"create-only output already exists: {path}")
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    write_new_json(
        manifest_path,
        {
            "schema_version": "chk3-grpo-reward-mean-plus-minus-one-se-figure-v1",
            "paper_model": "Model chk-3",
            "repository_artifact": "chk4",
            "training_stage": "GRPO",
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "source": {
                "trainer_state": str(TRAINER_STATE),
                "trainer_state_sha256": EXPECTED_TRAINER_STATE_SHA256,
                "completion_reward_rows": str(REWARD_ROWS),
                "completion_reward_rows_sha256": EXPECTED_REWARD_ROWS_SHA256,
            },
            "coverage": {
                "epochs": 3,
                "optimizer_steps": EXPECTED_STEPS,
                "completion_rewards_per_step": COMPLETIONS_PER_STEP,
                "completion_reward_rows": len(reward_rows),
                "rolling_window_steps": ROLLING_WINDOW,
            },
            "standard_error_definition": {
                "name": "naive optimizer-step-level standard error",
                "formula": "sample_sd(step_reward_means)/sqrt(number_of_steps)",
                "window_steps": ROLLING_WINDOW,
                "completion_rewards_per_step": COMPLETIONS_PER_STEP,
                "cluster_unit": "optimizer step",
                "clusters_per_window": ROLLING_WINDOW,
                "completion_rewards_per_window": (
                    ROLLING_WINDOW * COMPLETIONS_PER_STEP
                ),
                "sample_standard_deviation_ddof": 1,
                "band": "rolling step-level mean plus or minus one step-level standard error",
                "band_clipped": False,
                "accounts_for_within_step_completion_clustering": True,
                "corrects_between_step_serial_dependence": False,
                "corrects_overlapping_window_dependence": False,
                "accounts_for_between_run_randomness": False,
                "treats_step_means_as_independent_within_window": True,
                "is_standard_error": True,
                "is_confidence_interval": False,
            },
            "epoch_summaries": epoch_summaries,
            "checkpoint_annotations": {
                "evaluated_exploratory_checkpoint": EVALUATED_STEP,
                "terminal_checkpoint": EXPECTED_STEPS,
                "selected_checkpoint_step": None,
            },
            "rendering": {
                "per_step_reward_mean_line": False,
                "reward_mean_window_steps": ROLLING_WINDOW,
                "step_clustered_standard_error_band": True,
                "reward_support": [0.0, 1.0],
                "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
                "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
            },
        },
    )
    print(png_path)
    print(pdf_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
