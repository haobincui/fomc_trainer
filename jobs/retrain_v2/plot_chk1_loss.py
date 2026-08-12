#!/usr/bin/env python3
"""Plot train and evaluation loss from the chk1 loss-history JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=170)
    parser.add_argument("--epoch-steps", type=int, default=85)
    return parser.parse_args()


def load_history(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    evaluation: list[tuple[int, float]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            if row.get("train_loss") is not None:
                train.append((int(row["step"]), float(row["train_loss"])))
            if row.get("eval_loss") is not None:
                point = (int(row["step"]), float(row["eval_loss"]))
                if point not in evaluation:
                    evaluation.append(point)
    if not train:
        raise ValueError(f"No train-loss rows found in {path}")
    return train, evaluation


def main() -> None:
    args = parse_args()
    train, evaluation = load_history(args.input)
    train_steps, train_losses = zip(*train)

    fig, ax = plt.subplots(figsize=(11.2, 6.3), facecolor="#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    train_color = "#2563A6"
    eval_color = "#C58B18"
    ax.plot(
        train_steps,
        train_losses,
        color=train_color,
        linewidth=2.4,
        marker="o",
        markersize=5.2,
        markerfacecolor="#FAFBFC",
        markeredgewidth=1.6,
        label="Train loss (10-step windows)",
        zorder=3,
    )

    if evaluation:
        eval_steps, eval_losses = zip(*evaluation)
        ax.scatter(
            eval_steps,
            eval_losses,
            color=eval_color,
            edgecolor="#6F4E0B",
            linewidth=1.1,
            marker="D",
            s=72,
            label="Eval loss (full eval split)",
            zorder=5,
        )
        for step, loss in evaluation:
            ax.annotate(
                f"Epoch 1 eval  {loss:.4f}",
                (step, loss),
                xytext=(10, -23),
                textcoords="offset points",
                color="#6F4E0B",
                fontsize=10,
                fontweight="semibold",
            )

    latest_step, latest_loss = train[-1]
    ax.annotate(
        f"Latest logged  {latest_loss:.4f}\nstep {latest_step}",
        (latest_step, latest_loss),
        xytext=(13, 16),
        textcoords="offset points",
        color=train_color,
        fontsize=10,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": train_color, "linewidth": 1.0},
    )

    ax.axvline(args.epoch_steps, color="#6B7280", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(
        args.epoch_steps + 2,
        max(train_losses) - 0.03,
        "Epoch 2",
        color="#5B6472",
        fontsize=9.5,
        va="top",
    )

    ax.set_xlim(0, args.total_steps)
    lower = min([*train_losses, *(loss for _, loss in evaluation)]) - 0.08
    upper = max(train_losses) + 0.12
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Optimizer step", color="#303846", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", color="#303846", fontsize=11)
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(colors="#4B5563")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB2BF")
    ax.spines["bottom"].set_color("#AAB2BF")
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    fig.suptitle(
        "chk1 compressed-reasoning SFT loss",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#202733",
    )
    fig.text(
        0.075,
        0.915,
        "Current run · 2 epochs / 170 steps · train windows are not directly equivalent to full-split eval",
        ha="left",
        fontsize=10.5,
        color="#5B6472",
    )
    fig.text(
        0.075,
        0.025,
        f"Source: {args.input}  |  Generated from logged checkpoints",
        ha="left",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0.055, 0.065, 0.985, 0.89))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
