"""Create the paper-facing Model chk-2 training/validation loss figure.

The figure is source-bound to the completed 60-step recovery SFT and the
selection-only receipt that retained checkpoint 50. Outputs are create-only
so the historical checkpoint-318 figure remains available for audit.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
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
SELECTION_RECEIPT = ROOT / (
    "output/evaluation/retrain_v2/"
    "paper_chk2_minutes_checkpoint_hard_gate_probe_v1_20260901/"
    "relaxed_checkpoint_selection_v1.json"
)
OUTPUT_STEM = ROOT / (
    "docs/Chapter2/Chapter2Figs/stage2_train/"
    "chk2_train_eval_loss_cp50"
)

EXPECTED_LOSS_HISTORY_SHA256 = (
    "f035e90827c4c3233ceed22318ed706e5303645dddac9db82620a32b1cb7ca0f"
)
EXPECTED_ALL_RESULTS_SHA256 = (
    "147c587999492f1bf6b018ebb19505427dfed595b22c7680787b049bb717e445"
)
EXPECTED_SELECTION_RECEIPT_SHA256 = (
    "9c2fcff135114ff1165c60b881f109c69bb3c0f3b04368157744e634b99e138f"
)

EXPECTED_TRAIN_STEPS = 60
EXPECTED_EVAL_STEPS = (10, 20, 30, 40, 50, 60)
EXPECTED_TRAIN_SAMPLES = 305
EXPECTED_VALIDATION_SAMPLES = 42
ROLLING_WINDOW = 7
SELECTED_STEP = 50
MINIMUM_LOSS_STEP = 60


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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_and_validate() -> dict[str, Any]:
    require_sha256(LOSS_HISTORY, EXPECTED_LOSS_HISTORY_SHA256)
    require_sha256(ALL_RESULTS, EXPECTED_ALL_RESULTS_SHA256)
    require_sha256(SELECTION_RECEIPT, EXPECTED_SELECTION_RECEIPT_SHA256)

    rows = [
        json.loads(raw)
        for raw in LOSS_HISTORY.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("loss history is malformed")
    train_by_step = {
        int(row["step"]): float(row["train_loss"])
        for row in rows
        if row.get("train_loss") is not None
    }
    # Scheduled and final step-60 evaluations are identical. Keyed projection
    # retains one observation per optimizer step.
    eval_by_step = {
        int(row["step"]): float(row["eval_loss"])
        for row in rows
        if row.get("eval_loss") is not None
    }
    if tuple(sorted(train_by_step)) != tuple(range(1, EXPECTED_TRAIN_STEPS + 1)):
        raise RuntimeError("expected one train-loss value for every step 1..60")
    if tuple(sorted(eval_by_step)) != EXPECTED_EVAL_STEPS:
        raise RuntimeError("expected evaluation loss at steps 10, 20, ..., 60")

    results = read_json(ALL_RESULTS)
    if (
        int(results.get("train_samples", -1)) != EXPECTED_TRAIN_SAMPLES
        or int(results.get("eval_samples", -1)) != EXPECTED_VALIDATION_SAMPLES
        or not np.isclose(float(results["eval_loss"]), eval_by_step[60])
    ):
        raise RuntimeError("final training/evaluation summary drift")

    selection = read_json(SELECTION_RECEIPT)
    records = selection.get("records")
    if (
        selection.get("status") != "selected"
        or selection.get("selected_checkpoint") != SELECTED_STEP
        or selection.get("scope", {}).get("panel_rows") != 8
        or not isinstance(records, list)
        or {record.get("checkpoint") for record in records} != {40, 50, 60}
    ):
        raise RuntimeError("checkpoint-selection receipt drift")
    selected = next(
        record for record in records if record.get("checkpoint") == SELECTED_STEP
    )
    if (
        selected.get("relaxed_probe_pass") is not True
        or selected.get("strict_all_rows_pass") is not False
        or selected.get("metrics", {}).get("full_chain_pass_count") != 3
    ):
        raise RuntimeError("selected-checkpoint gate summary drift")

    train_steps = np.asarray(sorted(train_by_step), dtype=np.int64)
    train_loss = np.asarray(
        [train_by_step[step] for step in train_steps], dtype=np.float64
    )
    eval_steps = np.asarray(sorted(eval_by_step), dtype=np.int64)
    eval_loss = np.asarray(
        [eval_by_step[step] for step in eval_steps], dtype=np.float64
    )
    minimum_index = int(np.argmin(eval_loss))
    if int(eval_steps[minimum_index]) != MINIMUM_LOSS_STEP:
        raise RuntimeError("unexpected minimum validation-loss checkpoint")
    if not np.all(np.diff(eval_loss) < 0):
        raise RuntimeError("expected monotonically decreasing logged validation loss")

    rolling_loss = np.convolve(
        train_loss,
        np.ones(ROLLING_WINDOW, dtype=np.float64) / ROLLING_WINDOW,
        mode="valid",
    )
    rolling_steps = train_steps[ROLLING_WINDOW - 1 :]
    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "rolling_steps": rolling_steps,
        "rolling_loss": rolling_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "selection": selection,
    }


def render(data: dict[str, Any]) -> tuple[Path, Path]:
    train_steps = data["train_steps"]
    train_loss = data["train_loss"]
    rolling_steps = data["rolling_steps"]
    rolling_loss = data["rolling_loss"]
    eval_steps = data["eval_steps"]
    eval_loss = data["eval_loss"]
    selected_loss = float(eval_loss[np.where(eval_steps == SELECTED_STEP)[0][0]])
    minimum_loss = float(eval_loss[np.where(eval_steps == MINIMUM_LOSS_STEP)[0][0]])
    final_delta = minimum_loss - selected_loss

    blue = "#2F5F98"
    blue_light = "#9AB8D8"
    orange = "#D97706"
    ink = "#27313D"
    muted = "#68727D"
    grid = "#D9DEE5"
    guide = "#9CA3AF"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": ink,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.7),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1.9, 1.05], "hspace": 0.13},
        facecolor="white",
    )

    for axis in (ax_top, ax_bottom):
        axis.set_facecolor("white")
        axis.grid(axis="y", color=grid, linewidth=0.7, alpha=0.85)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(guide)
        axis.spines["bottom"].set_color(guide)
        for boundary in (20, 40):
            axis.axvline(
                boundary,
                color=guide,
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                zorder=0,
            )
        axis.axvline(
            SELECTED_STEP,
            color=blue,
            linewidth=1.15,
            linestyle=(0, (5, 3)),
            alpha=0.85,
            zorder=0,
        )

    ax_top.plot(
        train_steps,
        train_loss,
        color=blue_light,
        linewidth=0.85,
        alpha=0.58,
        label="Train loss (per optimizer step)",
        zorder=1,
    )
    ax_top.plot(
        rolling_steps,
        rolling_loss,
        color=blue,
        linewidth=2.2,
        label=f"Train loss ({ROLLING_WINDOW}-step trailing mean)",
        zorder=3,
    )
    ax_top.plot(
        eval_steps,
        eval_loss,
        color=orange,
        linewidth=1.9,
        marker="o",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Validation loss (42-sample monitoring split)",
        zorder=4,
    )
    ax_top.axvline(
        SELECTED_STEP,
        color=blue,
        linewidth=1.15,
        linestyle=(0, (5, 3)),
        alpha=0.85,
        label="Selected checkpoint (cp50)",
        zorder=2,
    )
    ax_top.set_ylabel("Completion loss")
    ax_top.set_ylim(1.15, 2.03)
    ax_top.legend(
        loc="upper right",
        frameon=True,
        fontsize=8.2,
        handlelength=2.7,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
    )
    for x, label in ((10, "Epoch 1"), (30, "Epoch 2"), (45, "Epoch 3")):
        ax_top.text(
            x,
            1.18,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=muted,
        )

    ax_bottom.plot(
        eval_steps,
        eval_loss,
        color=orange,
        linewidth=2.1,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.3,
        zorder=3,
    )
    ax_bottom.scatter(
        [SELECTED_STEP],
        [selected_loss],
        s=70,
        marker="s",
        color=blue,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    ax_bottom.scatter(
        [MINIMUM_LOSS_STEP],
        [minimum_loss],
        s=72,
        marker="D",
        color=orange,
        edgecolor=ink,
        linewidth=0.9,
        zorder=6,
    )
    for step, value in zip(eval_steps[:4], eval_loss[:4], strict=True):
        ax_bottom.annotate(
            f"{value:.4f}",
            xy=(step, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.1,
            color="#4B5563",
        )
    ax_bottom.annotate(
        f"Selected cp50\n{selected_loss:.4f}",
        xy=(SELECTED_STEP, selected_loss),
        xytext=(48.2, 1.5398),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=blue,
        arrowprops={"arrowstyle": "-", "color": blue, "linewidth": 0.9},
    )
    ax_bottom.annotate(
        f"Minimum cp60\n{minimum_loss:.4f}",
        xy=(MINIMUM_LOSS_STEP, minimum_loss),
        xytext=(58.6, 1.5370),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=ink,
        arrowprops={"arrowstyle": "-", "color": muted, "linewidth": 0.9},
    )
    ax_bottom.text(
        58.8,
        1.5585,
        f"cp50→cp60: {final_delta:+.6f} ({final_delta / selected_loss * 100:+.3f}%)",
        ha="right",
        va="center",
        fontsize=8.4,
        color=ink,
    )
    ax_bottom.set_xlim(0, 62)
    ax_bottom.set_ylim(1.519, 1.571)
    ax_bottom.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax_bottom.set_xlabel("Optimizer step")
    ax_bottom.set_ylabel("Validation loss\n(focused scale)")

    fig.suptitle(
        "Model chk-2 SFT: Training and validation loss",
        x=0.10,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color="#18212B",
    )
    fig.text(
        0.10,
        0.936,
        "305 training samples · 42-sample monitoring split · 3 epochs · evaluation every 10 optimizer steps",
        ha="left",
        fontsize=9.1,
        color="#59636E",
    )
    fig.text(
        0.10,
        0.018,
        "Epoch boundaries are grey dashed; cp50 is blue dashed. Selection used generation-quality screening, not minimum loss alone.",
        ha="left",
        fontsize=8.0,
        color=muted,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.895, bottom=0.105)

    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    for path in (png_path, pdf_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"create-only output already exists: {path}")
    fig.savefig(
        png_path,
        dpi=300,
        facecolor="white",
        metadata={
            "Title": "Model chk-2 SFT: Training and validation loss",
            "Source": "paper-chk2 60-step recovery SFT",
        },
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Model chk-2 SFT: Training and validation loss",
            "Author": "",
            "CreationDate": datetime(2026, 9, 1, tzinfo=timezone.utc),
            "ModDate": datetime(2026, 9, 1, tzinfo=timezone.utc),
        },
    )
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    data = load_and_validate()
    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    manifest_path = OUTPUT_STEM.with_suffix(".manifest.json")
    for path in (png_path, pdf_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"create-only output already exists: {path}")
    png_path, pdf_path = render(data)

    eval_steps = data["eval_steps"]
    eval_loss = data["eval_loss"]
    selected_loss = float(eval_loss[np.where(eval_steps == SELECTED_STEP)[0][0]])
    minimum_loss = float(eval_loss[np.where(eval_steps == MINIMUM_LOSS_STEP)[0][0]])
    manifest = {
        "schema_version": "paper-chk2-train-eval-loss-figure-v2",
        "paper_model": "Model chk-2",
        "status": "complete",
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
            "checkpoint_selection": {
                "path": str(SELECTION_RECEIPT),
                "sha256": EXPECTED_SELECTION_RECEIPT_SHA256,
                "receipt_sha256": data["selection"]["receipt_sha256"],
            },
        },
        "coverage": {
            "epochs": 3,
            "optimizer_steps": EXPECTED_TRAIN_STEPS,
            "train_loss_points": len(data["train_steps"]),
            "validation_loss_points": len(eval_steps),
            "train_samples": EXPECTED_TRAIN_SAMPLES,
            "validation_samples": EXPECTED_VALIDATION_SAMPLES,
        },
        "checkpoint_summary": {
            "selected_checkpoint": SELECTED_STEP,
            "selected_checkpoint_eval_loss": selected_loss,
            "minimum_eval_loss_checkpoint": MINIMUM_LOSS_STEP,
            "minimum_eval_loss": minimum_loss,
            "selected_minus_minimum_eval_loss": selected_loss - minimum_loss,
            "selected_checkpoint_full_chain_pass_count": 3,
            "selection_panel_rows": 8,
            "selection_panel_evaluation_eligible": False,
        },
        "rendering": {
            "train_raw": True,
            "train_trailing_mean_window": ROLLING_WINDOW,
            "validation_focused_panel": True,
            "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
            "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        },
    }
    manifest["payload_sha256"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    write_new_json(manifest_path, manifest)
    print(png_path)
    print(pdf_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
