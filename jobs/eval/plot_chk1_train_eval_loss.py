"""Create the paper-facing Model chk-1 train/eval loss figure."""

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
FULL_STATE = ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/"
    "adapters/chk1/trainer_state.json"
)
SELECTED_STATE = ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/"
    "merged/chk1/trainer_state.json"
)
OUTPUT_STEM = ROOT / "docs/Chapter2/Chapter2Figs/chk1_train_eval_loss_cp200"
EXPECTED_FULL_SHA256 = "ea7143be36657dfff23e5871ee642f4df166af428a86c433d5665575fded45ca"
EXPECTED_SELECTED_SHA256 = "6199b764f21f24b57eb0b48a48ead880a9112fb33c1b9279f9ad8b648d5c8a87"
SELECTED_STEP = 200
ROLLING_WINDOW = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_state(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"trainer-state binding drift: {path}: {observed}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("log_history"), list):
        raise RuntimeError(f"invalid trainer state: {path}")
    return value


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    full = read_state(FULL_STATE, EXPECTED_FULL_SHA256)
    selected = read_state(SELECTED_STATE, EXPECTED_SELECTED_SHA256)
    history = full["log_history"]
    train = [row for row in history if "loss" in row and "eval_loss" not in row]
    evaluation = [row for row in history if "eval_loss" in row]
    if len(train) != 255 or len(evaluation) != 25:
        raise RuntimeError("expected 255 train-loss and 25 eval-loss observations")
    if [row for row in history if int(row.get("step", -1)) <= SELECTED_STEP] != selected["log_history"]:
        raise RuntimeError("selected cp200 history is not an exact prefix of the full run")

    train_steps = np.asarray([int(row["step"]) for row in train], dtype=np.int64)
    train_loss = np.asarray([float(row["loss"]) for row in train], dtype=np.float64)
    eval_steps = np.asarray([int(row["step"]) for row in evaluation], dtype=np.int64)
    eval_loss = np.asarray([float(row["eval_loss"]) for row in evaluation], dtype=np.float64)
    rolling_loss = np.convolve(
        train_loss, np.ones(ROLLING_WINDOW, dtype=np.float64) / ROLLING_WINDOW, mode="valid"
    )
    rolling_steps = train_steps[ROLLING_WINDOW - 1 :]
    cp_eval = float(eval_loss[np.where(eval_steps == SELECTED_STEP)[0][0]])

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
        linewidth=0.85,
        alpha=0.55,
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
        markersize=3.7,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Evaluation loss (every 10 steps)",
        zorder=4,
    )
    ax.axvline(
        SELECTED_STEP,
        color="#4b5563",
        linewidth=1.15,
        linestyle=(0, (4, 3)),
        label="Paper checkpoint cp200",
        zorder=2,
    )
    ax.scatter(
        [SELECTED_STEP], [cp_eval], s=44, color="#d97706", edgecolor="white", linewidth=1.1, zorder=6
    )
    ax.annotate(
        f"cp200 eval loss = {cp_eval:.4f}",
        xy=(SELECTED_STEP, cp_eval),
        xytext=(192, 1.625),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#27313d",
        arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.9},
    )

    ax.set_xlim(0, 260)
    ax.set_ylim(1.60, 2.05)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Trainer-reported loss")
    ax.grid(axis="y", color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    legend = ax.legend(
        loc="upper right", frameon=True, fontsize=8.4, handlelength=2.5,
        facecolor="white", edgecolor="none", framealpha=0.9,
    )
    legend.set_zorder(10)

    fig.suptitle(
        "Model chk-1 SFT: Training and Evaluation Loss",
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
        "Full 3-epoch run (255 steps); evaluation every 10 steps; paper model retained at cp200",
        ha="left",
        fontsize=9.2,
        color="#59636e",
    )
    fig.text(
        0.09,
        0.018,
        "Note: train loss is a per-step batch statistic; eval loss is aggregated over the validation set.",
        ha="left",
        fontsize=8.1,
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
            "schema_version": "chk1-train-eval-loss-figure-v1",
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
                "full_trainer_state": str(FULL_STATE),
                "full_trainer_state_sha256": EXPECTED_FULL_SHA256,
                "selected_cp200_trainer_state": str(SELECTED_STATE),
                "selected_cp200_trainer_state_sha256": EXPECTED_SELECTED_SHA256,
            },
            "coverage": {
                "full_run_steps": int(full["global_step"]),
                "epochs": float(full["epoch"]),
                "train_loss_points": len(train),
                "eval_loss_points": len(evaluation),
                "eval_frequency_steps": 10,
            },
            "selected_checkpoint": {
                "step": SELECTED_STEP,
                "epoch": float(selected["epoch"]),
                "train_loss_at_step": float(train_loss[np.where(train_steps == SELECTED_STEP)[0][0]]),
                "eval_loss": cp_eval,
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
