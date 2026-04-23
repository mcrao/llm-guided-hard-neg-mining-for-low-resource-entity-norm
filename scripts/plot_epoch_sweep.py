#!/usr/bin/env python3
"""
plot_epoch_sweep.py — Generate Figure 1 for the NeurIPS paper.

Creates a dual-axis line plot of Acc@1 and Acc@5 vs. training epochs
from results/epoch_sweep.csv, saved to manuscript/NeurIPS/myPaper/figures/.

Usage
-----
  uv run python scripts/plot_epoch_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

EPOCH_SWEEP_CSV = PROJECT_ROOT / "results" / "epoch_sweep.csv"
OUT_DIR         = PROJECT_ROOT / "manuscript" / "NeurIPS" / "myPaper" / "figures"
OUT_PATH        = OUT_DIR / "epoch_sweep.pdf"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EPOCH_SWEEP_CSV)
    df = df.sort_values("epochs")

    epochs  = df["epochs"].tolist()
    acc1    = df["acc_at_1"].tolist()
    acc5    = df["acc_at_5"].tolist()
    mrr     = df["mrr"].tolist()

    # 3-epoch reference values (train_100pct, in-batch — the first data point)
    # These match the dashed reference lines in the paper's pgfplots figure.
    base_acc1 = 0.0459
    base_acc5 = 0.5991

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(5.5, 3.2))

    color_acc1 = "#d62728"   # red  — Acc@1
    color_acc5 = "#1f77b4"   # blue — Acc@5

    # ── Acc@1 (left axis) ────────────────────────────────────────────────────
    ax1.plot(epochs, acc1, color=color_acc1, marker="o", linewidth=2,
             markersize=6, label="Acc@1 (left axis)", zorder=3)
    ax1.axhline(base_acc1, color=color_acc1, linestyle="--", linewidth=1,
                alpha=0.6, label=f"Acc@1 @ 3 ep. ({base_acc1:.4f})")
    ax1.set_xlabel("Training epochs", fontsize=11)
    ax1.set_ylabel("Acc@1", color=color_acc1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_acc1)
    ax1.set_ylim(0.035, 0.055)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax1.set_xticks(epochs)

    # ── Acc@5 (right axis) ───────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc5, color=color_acc5, marker="s", linewidth=2,
             markersize=6, label="Acc@5 (right axis)", zorder=3)
    ax2.axhline(base_acc5, color=color_acc5, linestyle="--", linewidth=1,
                alpha=0.6, label=f"Acc@5 @ 3 ep. ({base_acc5:.4f})")
    ax2.set_ylabel("Acc@5", color=color_acc5, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_acc5)
    ax2.set_ylim(0.50, 0.68)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── Annotations ──────────────────────────────────────────────────────────
    # Annotate the Acc@1 divergence point at epoch 15
    # "Acc@1 drops" in red with arrow pointing to Acc@1 at epoch 15
    ax1.annotate(
        "Acc@1 drops",
        xy=(15, acc1[-1]),
        xytext=(12.8, 0.0405),
        fontsize=7.5,
        color=color_acc1,
        arrowprops=dict(arrowstyle="->", color=color_acc1, lw=1.2),
        ha="right",
    )
    # "Acc@5 still rises" in blue, placed just below the red label
    ax1.annotate(
        "Acc@5 still rises",
        xy=(12.8, 0.0405),
        xytext=(12.8, 0.0388),
        fontsize=7.5,
        color=color_acc5,
        ha="right",
        annotation_clip=False,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=7.5, framealpha=0.9)

    # ── Title and layout ──────────────────────────────────────────────────────
    ax1.set_title(
        "Epoch sweep on train_100pct (6,977 match pairs, distilbert)",
        fontsize=9, pad=6,
    )
    fig.tight_layout()

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
