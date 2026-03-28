#!/usr/bin/env python3
"""
sweep_epochs.py — Investigate the Acc@1 plateau by sweeping training epochs.

The bi-encoder Acc@1 is flat (~0.045) across all data splits at 3 epochs.
This script runs the bi-encoder at 3, 5, 10, 15 epochs on a single split
to check whether more training helps or whether the plateau is structural
(i.e. caused by easy in-batch negatives rather than insufficient training).

Expected finding: Acc@1 will not significantly improve even with more epochs,
because MultipleNegativesRankingLoss only sees random in-batch negatives
(typically easy non-matches), never the near-duplicate hard cases that M3
will inject. This will motivate the LLM hard-negative generation in M3.

Usage
-----
  uv run python scripts/sweep_epochs.py                      # all epoch counts
  uv run python scripts/sweep_epochs.py --epochs 3,5,10      # custom list
  uv run python scripts/sweep_epochs.py --split train_25pct  # different split
  uv run python scripts/sweep_epochs.py --no-cache           # re-train always

Output
------
  results/epoch_sweep.csv
    Columns: epoch, train_split, acc_at_1, acc_at_5, mrr,
             n_train_match_pairs, n_eval_queries, training_time_s,
             eval_time_s, timestamp
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import typer
from rich.table import Table

from wdc_hn.baselines.bi_encoder_in_batch import BiEncoderBaseline
from wdc_hn.data.splits import load_split
from wdc_hn.utils import console, get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SPLITS_DIR  = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR  = PROJECT_ROOT / "models"
CSV_PATH    = RESULTS_DIR / "epoch_sweep.csv"

CSV_COLUMNS = [
    "epochs", "train_split", "eval_split",
    "acc_at_1", "acc_at_5", "mrr",
    "n_train_match_pairs", "n_eval_queries",
    "training_time_s", "eval_time_s", "timestamp",
]

DEFAULT_EPOCH_COUNTS = [3, 5, 10, 15]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _n_match_pairs(df) -> int:
    return int((df["label"] == 1).sum())


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _append_csv(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@app.command()
def main(
    split: str = typer.Option(
        "train_100pct",
        help="Training split to use. One of: train_10pct, train_25pct, train_100pct.",
    ),
    epochs: str = typer.Option(
        None,
        help="Comma-separated list of epoch counts to sweep. Default: 3,5,10,15.",
    ),
    batch_size: int = typer.Option(32, help="Per-device training batch size."),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore cached models and always re-train.",
    ),
) -> None:
    """
    Sweep training epochs to investigate the Acc@1 plateau.

    The flat Acc@1 (~0.045) across 10%/25%/100% splits at 3 epochs is
    expected with in-batch negatives: the model never sees near-duplicate
    hard cases during training. This sweep confirms whether more epochs
    help or whether the ceiling is structural.
    """
    epoch_counts: List[int] = (
        [int(e.strip()) for e in epochs.split(",")]
        if epochs
        else DEFAULT_EPOCH_COUNTS
    )

    console.rule("[bold cyan]Epoch Sweep — Acc@1 Plateau Investigation[/bold cyan]")
    console.print(
        f"[bold]Split:[/bold]        {split}\n"
        f"[bold]Epoch counts:[/bold] {epoch_counts}\n"
        f"[bold]Batch size:[/bold]   {batch_size}\n"
        f"[bold]Output:[/bold]       {CSV_PATH}\n"
    )
    console.print(
        "[dim]Why this plateau exists:[/dim]\n"
        "  MultipleNegativesRankingLoss treats every other item in the\n"
        "  batch as a negative. With random batches from the training set,\n"
        "  negatives are semantically distant (e.g. 'Apple MacBook' vs\n"
        "  'HP Keyboard'), which the model learns to separate quickly.\n"
        "  Near-duplicate hard negatives ('Apple iPhone 14 Pro 256GB' vs\n"
        "  'Apple iPhone 14 Pro Max 256GB') never appear — so Acc@1\n"
        "  plateaus even as Acc@5 and MRR continue to improve.\n"
        "  M3 fixes this by injecting LLM-generated hard negatives.\n"
    )

    log.info("Loading val set …")
    val_df = load_split(SPLITS_DIR, "val")
    log.info("Loading training split …")
    train_df = load_split(SPLITS_DIR, split)

    rows = []

    for n_epochs in epoch_counts:
        console.rule(f"[bold]Epochs = {n_epochs}[/bold]")
        model_dir = MODELS_DIR / f"bi_encoder_sweep_{split.replace('train_', '')}_{n_epochs}ep"

        baseline = BiEncoderBaseline()

        if not no_cache and model_dir.exists():
            log.info(f"[yellow]Cache hit[/yellow] — loading {model_dir}")
            baseline.load(model_dir)
            train_time_s = 0.0
        else:
            log.info(f"Training for {n_epochs} epoch(s) …")
            t0 = time.time()
            baseline.train(
                train_df,
                output_dir=model_dir,
                epochs=n_epochs,
                batch_size=batch_size,
            )
            train_time_s = round(time.time() - t0, 1)

        metrics = baseline.evaluate(val_df)

        row = {
            "epochs":               n_epochs,
            "train_split":          split,
            "eval_split":           "val",
            "acc_at_1":             _fmt(metrics.get("acc_at_1", 0.0)),
            "acc_at_5":             _fmt(metrics.get("acc_at_5", 0.0)),
            "mrr":                  _fmt(metrics.get("mrr",      0.0)),
            "n_train_match_pairs":  _n_match_pairs(train_df),
            "n_eval_queries":       int(metrics.get("n_queries", 0)),
            "training_time_s":      str(train_time_s),
            "eval_time_s":          str(metrics.get("eval_time_s", 0.0)),
            "timestamp":            _now(),
        }
        _append_csv(row)
        rows.append(row)

        console.print(
            f"  [bold green]✓[/bold green] "
            f"epochs=[cyan]{n_epochs}[/cyan] | "
            f"Acc@1=[yellow]{row['acc_at_1']}[/yellow]  "
            f"Acc@5=[yellow]{row['acc_at_5']}[/yellow]  "
            f"MRR=[yellow]{row['mrr']}[/yellow]  "
            f"(train {train_time_s}s)"
        )

    # Summary table
    console.rule("[bold green]Epoch Sweep Summary[/bold green]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Epochs",     justify="right", style="cyan")
    table.add_column("Acc@1",      justify="right", style="yellow")
    table.add_column("Acc@5",      justify="right", style="yellow")
    table.add_column("MRR",        justify="right", style="green")
    table.add_column("Train (s)",  justify="right")

    for r in rows:
        table.add_row(
            str(r["epochs"]),
            str(r["acc_at_1"]),
            str(r["acc_at_5"]),
            str(r["mrr"]),
            str(r["training_time_s"]),
        )
    console.print(table)
    console.print(
        f"\n[bold green]✓[/bold green] Results saved to [cyan]{CSV_PATH}[/cyan]\n"
        "\n[bold]Interpretation:[/bold]\n"
        "  • If Acc@1 stays flat → plateau is structural (hard-negative deficit)\n"
        "  • If Acc@1 rises → more epochs help; try 15–20 in run_baselines.py\n"
        "  • Either way, M3 LLM hard negatives should push Acc@1 past 0.10\n"
    )


if __name__ == "__main__":
    app()
