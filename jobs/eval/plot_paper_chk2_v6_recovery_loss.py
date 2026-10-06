"""Render a convergence diagnostic for the paper Model chk-2 recovery SFT."""

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
    "paper_chk2_chk1_cp200_minutes_v6_recovery_full3ep_lr1e6_v1_20260901/"
    "adapters/chk2"
)
LOSS_HISTORY = RUN_ROOT / "loss_history.jsonl"
ALL_RESULTS = RUN_ROOT / "all_results.json"
OUTPUT_STEM = RUN_ROOT / "diagnostics/loss_convergence_diagnostic_v1_20260901"
EXPECTED_LOSS_HISTORY_SHA256 = (
    "f035e90827c4c3233ceed22318ed706e5303645dddac9db82620a32b1cb7ca0f"
)
EXPECTED_ALL_RESULTS_SHA256 = (
    "147c587999492f1bf6b018ebb19505427dfed595b22c7680787b049bb717e445"
)
ROLLING_WINDOW = 7
EXPECTED_TRAIN_STEPS = 60
EXPECTED_EVAL_STEPS = (10, 20, 30, 40, 50, 60)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"source binding drift: {path}: {observed}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    require_sha256(LOSS_HISTORY, EXPECTED_LOSS_HISTORY_SHA256)
    require_sha256(ALL_RESULTS, EXPECTED_ALL_RESULTS_SHA256)
    rows = [
        json.loads(line)
        for line in LOSS_HISTORY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_by_step = {
        int(row["step"]): float(row["train_loss"])
        for row in rows
        if row.get("train_loss") is not None
    }
    # The trainer performs a scheduled and a final evaluation at step 60.
    # They are identical; keyed projection keeps one observation per step.
    eval_by_step = {
        int(row["step"]): float(row["eval_loss"])
        for row in rows
        if row.get("eval_loss") is not None
    }
    if tuple(sorted(train_by_step)) != tuple(range(1, EXPECTED_TRAIN_STEPS + 1)):
        raise RuntimeError("expected one train-loss observation for every step 1..60")
    if tuple(sorted(eval_by_step)) != EXPECTED_EVAL_STEPS:
        raise RuntimeError("expected evaluation loss at steps 10, 20, ..., 60")

    results = read_json(ALL_RESULTS)
    if not np.isclose(float(results["eval_loss"]), eval_by_step[60]):
        raise RuntimeError("final evaluation result disagrees with loss history")

    train_steps = np.asarray(sorted(train_by_step), dtype=np.int64)
    train_loss = np.asarray(
        [train_by_step[step] for step in train_steps], dtype=np.float64
    )
    eval_steps = np.asarray(sorted(eval_by_step), dtype=np.int64)
    eval_loss = np.asarray(
        [eval_by_step[step] for step in eval_steps], dtype=np.float64
    )
    rolling_loss = np.convolve(
        train_loss,
        np.ones(ROLLING_WINDOW, dtype=np.float64) / ROLLING_WINDOW,
        mode="valid",
    )
    rolling_steps = train_steps[ROLLING_WINDOW - 1 :]

    epoch_means = [
        float(train_loss[start : start + 20].mean()) for start in (0, 20, 40)
    ]
    eval_deltas = np.diff(eval_loss)
    last_eval_delta = float(eval_loss[-1] - eval_loss[-2])
    last_eval_relative_percent = float(last_eval_delta / eval_loss[-2] * 100)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#27313d",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
        }
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.1], "hspace": 0.13},
        facecolor="white",
    )

    for axis in (ax_top, ax_bottom):
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#d9dee5", linewidth=0.7, alpha=0.8)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9ca3af")
        axis.spines["bottom"].set_color("#9ca3af")
        for boundary in (20, 40):
            axis.axvline(
                boundary,
                color="#9ca3af",
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                zorder=0,
            )

    ax_top.plot(
        train_steps,
        train_loss,
        color="#9ab8d8",
        linewidth=0.85,
        alpha=0.58,
        label="Train loss (per optimizer step)",
        zorder=1,
    )
    ax_top.plot(
        rolling_steps,
        rolling_loss,
        color="#2f5f98",
        linewidth=2.2,
        label=f"Train loss ({ROLLING_WINDOW}-step trailing mean)",
        zorder=3,
    )
    ax_top.plot(
        eval_steps,
        eval_loss,
        color="#d97706",
        linewidth=1.8,
        marker="o",
        markersize=4.3,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Validation loss (42 samples)",
        zorder=4,
    )
    ax_top.set_ylabel("Loss")
    ax_top.set_ylim(1.15, 2.03)
    ax_top.legend(
        loc="upper right",
        frameon=True,
        fontsize=8.5,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
    )
    for x, label in ((10, "Epoch 1"), (30, "Epoch 2"), (50, "Epoch 3")):
        ax_top.text(
            x,
            1.18,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#68727d",
        )

    ax_bottom.plot(
        eval_steps,
        eval_loss,
        color="#d97706",
        linewidth=2.0,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.3,
        zorder=3,
    )
    ax_bottom.scatter(
        [60],
        [eval_loss[-1]],
        s=58,
        marker="D",
        color="#d97706",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    for step, value in zip(eval_steps, eval_loss, strict=True):
        ax_bottom.annotate(
            f"{value:.4f}",
            xy=(step, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#4b5563",
        )
    ax_bottom.annotate(
        f"Step 50→60: {last_eval_delta:+.5f} ({last_eval_relative_percent:+.2f}%)",
        xy=(60, eval_loss[-1]),
        xytext=(59, 1.5515),
        ha="right",
        va="center",
        fontsize=8.6,
        color="#27313d",
        arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.9},
    )
    ax_bottom.set_xlim(0, 62)
    ax_bottom.set_ylim(1.519, 1.571)
    ax_bottom.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax_bottom.set_xlabel("Optimizer step")
    ax_bottom.set_ylabel("Validation loss\n(zoomed scale)")

    fig.suptitle(
        "Model chk-2 SFT: Loss Convergence Diagnostic",
        x=0.10,
        y=0.972,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color="#18212b",
    )
    fig.text(
        0.10,
        0.932,
        "305 training samples · 42 validation samples · 3 epochs · evaluation every 10 optimizer steps",
        ha="left",
        fontsize=9.2,
        color="#59636e",
    )
    fig.text(
        0.10,
        0.018,
        "Epoch boundaries are dashed. Per-step train loss is batch-level and noisy; the lower panel magnifies validation convergence.",
        ha="left",
        fontsize=8.1,
        color="#68727d",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.89, bottom=0.10)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
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
            "schema_version": "paper-chk2-loss-convergence-diagnostic-v1",
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "sources": {
                "loss_history": {
                    "path": str(LOSS_HISTORY),
                    "sha256": EXPECTED_LOSS_HISTORY_SHA256,
                },
                "all_results": {
                    "path": str(ALL_RESULTS),
                    "sha256": EXPECTED_ALL_RESULTS_SHA256,
                },
            },
            "coverage": {
                "train_loss_points": len(train_steps),
                "validation_loss_points": len(eval_steps),
                "train_samples": 305,
                "validation_samples": 42,
                "epochs": 3,
                "optimizer_steps": 60,
            },
            "diagnostics": {
                "epoch_train_loss_means": epoch_means,
                "validation_steps": eval_steps.tolist(),
                "validation_loss": eval_loss.tolist(),
                "validation_deltas": eval_deltas.tolist(),
                "step_50_to_60_delta": last_eval_delta,
                "step_50_to_60_relative_percent": last_eval_relative_percent,
                "minimum_validation_loss": float(eval_loss.min()),
                "minimum_validation_loss_step": int(eval_steps[np.argmin(eval_loss)]),
            },
            "rendering": {
                "train_raw": True,
                "train_trailing_mean_window": ROLLING_WINDOW,
                "validation_zoom_panel": True,
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
