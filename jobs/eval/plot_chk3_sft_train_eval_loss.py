"""Create the paper-facing Model chk-3 SFT train/evaluation loss figure."""

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
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811/"
    "adapters/chk4_sft"
)
TRAINER_STATE = RUN_ROOT / "trainer_state.json"
ALL_RESULTS = RUN_ROOT / "all_results.json"
OUTPUT_STEM = ROOT / "docs/Chapter2/Chapter2Figs/chk3_sft_train_eval_loss_cp38"
EXPECTED_TRAINER_STATE_SHA256 = (
    "dccb3c1ab9f822b3dc01cca5a88ab51a744393ba267255417e92f62545d73c35"
)
EXPECTED_ALL_RESULTS_SHA256 = (
    "3a963935d31ba1e1dc1012e666f73fd026ec98ae7e02a194ebb67177ae7414ce"
)
SELECTED_STEP = 38
FINAL_STEP = 39
ROLLING_WINDOW = 5


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
    if int(state.get("global_step", -1)) != FINAL_STEP or len(train) != FINAL_STEP:
        raise RuntimeError("expected the complete 39-step Model chk-3 SFT trajectory")
    if [int(row["step"]) for row in evaluation] != [13, 26, 39]:
        raise RuntimeError("expected scheduled evaluation loss at steps 13, 26, and 39")
    if int(results.get("eval_samples", -1)) != 13:
        raise RuntimeError("expected 13 validation samples")
    if abs(float(results["eval_loss"]) - float(evaluation[-1]["eval_loss"])) > 1e-12:
        raise RuntimeError("final all-results evaluation loss does not match trainer state")

    train_steps = np.asarray([int(row["step"]) for row in train], dtype=np.int64)
    train_loss = np.asarray([float(row["loss"]) for row in train], dtype=np.float64)
    eval_steps = np.asarray([int(row["step"]) for row in evaluation], dtype=np.int64)
    eval_loss = np.asarray([float(row["eval_loss"]) for row in evaluation], dtype=np.float64)
    rolling_loss = np.convolve(
        train_loss,
        np.ones(ROLLING_WINDOW, dtype=np.float64) / ROLLING_WINDOW,
        mode="valid",
    )
    rolling_steps = train_steps[ROLLING_WINDOW - 1 :]
    selected_train_loss = float(
        train_loss[np.where(train_steps == SELECTED_STEP)[0][0]]
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
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200, facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        train_steps,
        train_loss,
        color="#9ab8d8",
        linewidth=0.9,
        alpha=0.58,
        label="Train loss (per step)",
        zorder=1,
    )
    ax.plot(
        rolling_steps,
        rolling_loss,
        color="#2f5f98",
        linewidth=2.15,
        label=f"Train loss ({ROLLING_WINDOW}-step trailing mean)",
        zorder=3,
    )
    ax.plot(
        eval_steps,
        eval_loss,
        color="#d97706",
        linewidth=1.85,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Evaluation loss",
        zorder=4,
    )
    ax.axvline(
        SELECTED_STEP,
        color="#4b5563",
        linewidth=1.15,
        linestyle=(0, (4, 3)),
        label="Retained pre-GRPO state (cp38)",
        zorder=2,
    )

    for step, value in zip(eval_steps, eval_loss):
        x_offset = -0.8 if int(step) == FINAL_STEP else 0.0
        horizontal = "right" if int(step) == FINAL_STEP else "center"
        ax.annotate(
            f"{value:.4f}",
            xy=(int(step), float(value)),
            xytext=(int(step) + x_offset, float(value) + 0.045),
            ha=horizontal,
            va="bottom",
            fontsize=8.5,
            color="#7c4a03",
        )

    ax.set_xlim(0, 40.2)
    ax.set_ylim(1.25, 2.26)
    # Keep cp38 explicit while avoiding an unreadable 38/39 tick collision.
    # The final evaluation marker at step 39 remains visible and labelled.
    ax.set_xticks([0, 5, 10, 13, 20, 26, 30, 38])
    ax.set_xlabel("Optimiser step")
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
        fontsize=8.2,
        handlelength=2.5,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    legend.set_zorder(10)

    fig.suptitle(
        "Model chk-3 SFT: Training and Evaluation Loss",
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
        "39 optimiser updates; evaluation at steps 13, 26, and 39",
        ha="left",
        fontsize=9.0,
        color="#59636e",
    )
    fig.text(
        0.09,
        0.018,
        "Note: train loss is batch-level; evaluation loss covers 13 validation meetings. cp38 was retained by behavioural gates.",
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
            "schema_version": "chk3-sft-train-eval-loss-figure-v1",
            "paper_model": "Model chk-3",
            "repository_artifact": "chk4",
            "training_stage": "SFT",
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
                "eval_loss_points": len(evaluation),
                "eval_steps": eval_steps.tolist(),
                "eval_samples": int(results["eval_samples"]),
            },
            "checkpoint_summary": {
                "retained_sft_checkpoint_step": SELECTED_STEP,
                "train_loss_at_retained_step": selected_train_loss,
                "final_training_step": FINAL_STEP,
                "minimum_logged_eval_loss": float(eval_loss.min()),
                "minimum_logged_eval_loss_step": int(eval_steps[np.argmin(eval_loss)]),
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
