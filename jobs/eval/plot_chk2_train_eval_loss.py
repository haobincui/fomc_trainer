"""Create the paper-facing Model chk-2 train/evaluation loss figure."""

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
TRAINER_STATE = ROOT / (
    "output/training/retrain_v2/"
    "chk3_direct_chk1_cp200_full3ep_lr1e6_20260810/"
    "adapters/chk3/trainer_state.json"
)
ALL_RESULTS = ROOT / (
    "output/training/retrain_v2/"
    "chk3_direct_chk1_cp200_full3ep_lr1e6_20260810/"
    "adapters/chk3/all_results.json"
)
OUTPUT_STEM = ROOT / "docs/Chapter2/Chapter2Figs/chk2_train_eval_loss_cp318"
EXPECTED_TRAINER_STATE_SHA256 = (
    "f90783f15a2cfe6790139245ddfb1413913246e4de1af83a12f44b2ba3850327"
)
EXPECTED_ALL_RESULTS_SHA256 = (
    "dcc576c39d4e7fd697a5f5d1e6a9e1ddf342f2164932c8e490025796a8d44897"
)
SELECTED_STEP = 318
ROLLING_WINDOW = 10


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


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    state = read_json(TRAINER_STATE, EXPECTED_TRAINER_STATE_SHA256)
    results = read_json(ALL_RESULTS, EXPECTED_ALL_RESULTS_SHA256)
    history = state.get("log_history")
    if not isinstance(history, list):
        raise RuntimeError("trainer state has no log_history")

    train = [row for row in history if "loss" in row and "eval_loss" not in row]
    evaluation = [row for row in history if "eval_loss" in row]
    if int(state.get("global_step", -1)) != SELECTED_STEP or len(train) != 318:
        raise RuntimeError("expected the complete 318-step Model chk-2 trajectory")
    if len(evaluation) != 31:
        raise RuntimeError("expected 31 scheduled evaluation-loss observations")

    train_steps = np.asarray([int(row["step"]) for row in train], dtype=np.int64)
    train_loss = np.asarray([float(row["loss"]) for row in train], dtype=np.float64)
    scheduled_eval_steps = np.asarray(
        [int(row["step"]) for row in evaluation], dtype=np.int64
    )
    scheduled_eval_loss = np.asarray(
        [float(row["eval_loss"]) for row in evaluation], dtype=np.float64
    )
    if SELECTED_STEP in scheduled_eval_steps:
        raise RuntimeError("final evaluation unexpectedly duplicates a scheduled point")

    selected_eval_loss = float(results["eval_loss"])
    eval_steps = np.append(scheduled_eval_steps, SELECTED_STEP)
    eval_loss = np.append(scheduled_eval_loss, selected_eval_loss)
    order = np.argsort(eval_steps)
    eval_steps = eval_steps[order]
    eval_loss = eval_loss[order]

    rolling_loss = np.convolve(
        train_loss,
        np.ones(ROLLING_WINDOW, dtype=np.float64) / ROLLING_WINDOW,
        mode="valid",
    )
    rolling_steps = train_steps[ROLLING_WINDOW - 1 :]
    minimum_index = int(np.argmin(eval_loss))
    minimum_step = int(eval_steps[minimum_index])
    minimum_eval_loss = float(eval_loss[minimum_index])
    if minimum_step != 250:
        raise RuntimeError(f"unexpected minimum evaluation-loss step: {minimum_step}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#27313d",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200, facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        train_steps,
        train_loss,
        color="#9ab8d8",
        linewidth=0.8,
        alpha=0.5,
        label="Train loss (per step)",
        zorder=1,
    )
    ax.plot(
        rolling_steps,
        rolling_loss,
        color="#2f5f98",
        linewidth=2.1,
        label=f"Train loss ({ROLLING_WINDOW}-step trailing mean)",
        zorder=3,
    )
    ax.plot(
        eval_steps,
        eval_loss,
        color="#d97706",
        linewidth=1.8,
        marker="o",
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Evaluation loss",
        zorder=4,
    )

    ax.axvline(
        minimum_step,
        color="#6b7280",
        linewidth=1.05,
        linestyle=(0, (4, 3)),
        label="Minimum validation loss (cp250)",
        zorder=2,
    )
    ax.axvline(
        SELECTED_STEP,
        color="#374151",
        linewidth=1.15,
        linestyle=(0, (1.5, 2.2)),
        label="Retained paper checkpoint (cp318)",
        zorder=2,
    )
    ax.scatter(
        [minimum_step],
        [minimum_eval_loss],
        s=46,
        marker="D",
        color="#d97706",
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    ax.scatter(
        [SELECTED_STEP],
        [selected_eval_loss],
        s=52,
        marker="o",
        color="#d97706",
        edgecolor="#374151",
        linewidth=1.0,
        zorder=6,
    )
    ax.annotate(
        f"minimum: cp250 = {minimum_eval_loss:.4f}",
        xy=(minimum_step, minimum_eval_loss),
        xytext=(218, 1.073),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#27313d",
        arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.8},
    )
    ax.annotate(
        f"retained: cp318 = {selected_eval_loss:.4f}",
        xy=(SELECTED_STEP, selected_eval_loss),
        xytext=(312, 1.205),
        ha="right",
        va="bottom",
        fontsize=8.7,
        color="#27313d",
        arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.8},
    )

    ax.set_xlim(0, 326)
    ax.set_ylim(0.96, 1.40)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Trainer-reported loss")
    ax.grid(axis="y", color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=8.0,
        handlelength=2.5,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    legend.set_zorder(10)

    fig.suptitle(
        "Model chk-2 SFT: Training and Evaluation Loss",
        x=0.09,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color="#18212b",
    )
    fig.text(
        0.09,
        0.91,
        "Repository artifact chk3; full 3-epoch run (318 steps); evaluation every 10 steps and at completion",
        ha="left",
        fontsize=9.0,
        color="#59636e",
    )
    fig.text(
        0.09,
        0.018,
        "Note: train loss is batch-level; evaluation loss uses all 199 validation examples. cp318 was retained by hard-gate replay.",
        ha="left",
        fontsize=7.9,
        color="#68727d",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.17)

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
            "schema_version": "chk2-train-eval-loss-figure-v1",
            "paper_model": "Model chk-2",
            "repository_artifact": "chk3",
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
                "all_results": str(ALL_RESULTS),
                "all_results_sha256": EXPECTED_ALL_RESULTS_SHA256,
            },
            "coverage": {
                "epochs": float(state["epoch"]),
                "full_run_steps": int(state["global_step"]),
                "train_loss_points": len(train),
                "scheduled_eval_loss_points": len(evaluation),
                "final_eval_loss_points": 1,
                "eval_samples": int(results["eval_samples"]),
                "eval_frequency_steps": 10,
            },
            "checkpoint_summary": {
                "minimum_eval_loss_step": minimum_step,
                "minimum_eval_loss": minimum_eval_loss,
                "retained_checkpoint_step": SELECTED_STEP,
                "retained_checkpoint_eval_loss": selected_eval_loss,
                "retained_minus_minimum_eval_loss": selected_eval_loss
                - minimum_eval_loss,
            },
            "rendering": {
                "train_raw": True,
                "train_trailing_mean_window": ROLLING_WINDOW,
                "eval_raw": True,
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
