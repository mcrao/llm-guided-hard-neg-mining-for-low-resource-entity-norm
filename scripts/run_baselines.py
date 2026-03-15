#!/usr/bin/env python3
"""
run_baselines.py — Milestone 2 baseline evaluation entry point.

Runs both baselines across all data splits and writes results/baselines.csv.

Usage
-----
  uv run python scripts/run_baselines.py                  # full run (all splits)
  uv run python scripts/run_baselines.py --splits 10pct   # single split
  uv run python scripts/run_baselines.py --skip-bi-encoder # BM25 only
  uv run python scripts/run_baselines.py --skip-bm25       # bi-encoder only
  uv run python scripts/run_baselines.py --no-cache        # ignore saved models

What gets evaluated
-------------------
  Baseline 1 — TF-IDF (BM25-style):
    Unsupervised; one result on the val set regardless of training split.

  Baseline 2 — Bi-encoder (in-batch negatives):
    Trained separately on train_10pct, train_25pct, train_100pct.
    Each trained model is evaluated on the full val set.
    This shows the low-resource degradation curve (RQ3 in the roadmap).

Output
------
  results/baselines.csv
    Columns: method, train_split, eval_split, acc_at_1, acc_at_5, mrr,
             n_train_match_pairs, n_eval_queries, training_time_s,
             eval_time_s, timestamp
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import typer
from rich.panel import Panel
from rich.table import Table

from wdc_hn.baselines.bm25 import TFIDFBaseline
from wdc_hn.baselines.bi_encoder_in_batch import BiEncoderBaseline
from wdc_hn.data.splits import load_split
from wdc_hn.utils import console, get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# ── Paths ──────────────────────────────────────────────────────────────────────

SPLITS_DIR  = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR  = PROJECT_ROOT / "models"
CSV_PATH    = RESULTS_DIR / "baselines.csv"

CSV_COLUMNS = [
    "method",
    "train_split",
    "eval_split",
    "acc_at_1",
    "acc_at_5",
    "mrr",
    "n_train_match_pairs",
    "n_eval_queries",
    "training_time_s",
    "eval_time_s",
    "timestamp",
]

TRAIN_SPLITS = ["train_10pct", "train_25pct", "train_100pct"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _n_match_pairs(df) -> int:
    return int((df["label"] == 1).sum())


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _append_csv(row: Dict) -> None:
    """Append one result row to baselines.csv (creates file + header if needed)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_result(row: Dict) -> None:
    console.print(
        f"  [bold green]✓[/bold green] "
        f"[bold]{row['method']}[/bold] | "
        f"train=[cyan]{row['train_split']}[/cyan] | "
        f"Acc@1=[yellow]{row['acc_at_1']}[/yellow]  "
        f"Acc@5=[yellow]{row['acc_at_5']}[/yellow]  "
        f"MRR=[yellow]{row['mrr']}[/yellow]"
    )


def _summary_table(rows: List[Dict]) -> None:
    """Print a Rich summary table of all results."""
    table = Table(
        title="Baseline Results Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Method",       style="bold")
    table.add_column("Train Split",  style="cyan")
    table.add_column("Eval Split",   style="cyan")
    table.add_column("Acc@1",        justify="right", style="yellow")
    table.add_column("Acc@5",        justify="right", style="yellow")
    table.add_column("MRR",          justify="right", style="green")
    table.add_column("Train Pairs",  justify="right")
    table.add_column("Eval Queries", justify="right")
    table.add_column("Train (s)",    justify="right")
    table.add_column("Eval (s)",     justify="right")

    for r in rows:
        table.add_row(
            r["method"],
            str(r["train_split"]),
            r["eval_split"],
            str(r["acc_at_1"]),
            str(r["acc_at_5"]),
            str(r["mrr"]),
            str(r["n_train_match_pairs"]),
            str(r["n_eval_queries"]),
            str(r["training_time_s"]),
            str(r["eval_time_s"]),
        )
    console.print(table)


# ── Baseline runners ──────────────────────────────────────────────────────────

def run_bm25(val_df) -> Dict:
    """Run the TF-IDF baseline on the val set (no training required)."""
    console.rule("[bold]Baseline 1: TF-IDF (BM25-style)[/bold]")
    baseline = TFIDFBaseline()
    metrics  = baseline.evaluate(val_df)

    row = {
        "method":               "bm25_tfidf",
        "train_split":          "N/A",
        "eval_split":           "val",
        "acc_at_1":             _fmt(metrics.get("acc_at_1", 0.0)),
        "acc_at_5":             _fmt(metrics.get("acc_at_5", 0.0)),
        "mrr":                  _fmt(metrics.get("mrr",      0.0)),
        "n_train_match_pairs":  "N/A",
        "n_eval_queries":       int(metrics.get("n_queries", 0)),
        "training_time_s":      "0.0",
        "eval_time_s":          str(metrics.get("eval_time_s", 0.0)),
        "timestamp":            _now(),
    }
    _append_csv(row)
    _print_result(row)
    return row


def run_bi_encoder_split(
    train_split: str,
    train_df,
    val_df,
    use_cache: bool,
    epochs: int,
    batch_size: int,
) -> Dict:
    """Train + evaluate the bi-encoder on one training split."""
    model_dir = MODELS_DIR / f"bi_encoder_in_batch_{train_split.replace('train_', '')}"

    baseline = BiEncoderBaseline()

    # ── Training (skip if cached model exists and --no-cache not set) ─────────
    train_time_s = 0.0

    if use_cache and model_dir.exists():
        log.info(
            f"[yellow]Cache hit[/yellow] — loading model from {model_dir}. "
            "Pass --no-cache to re-train."
        )
        baseline.load(model_dir)
    else:
        log.info(f"Training bi-encoder on [bold]{train_split}[/bold] …")
        t0 = time.time()
        baseline.train(
            train_df,
            output_dir=model_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
        train_time_s = round(time.time() - t0, 1)

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = baseline.evaluate(val_df)

    row = {
        "method":               "bi_encoder_in_batch",
        "train_split":          train_split,
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
    _print_result(row)
    return row


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    splits: Optional[List[str]] = typer.Option(
        None,
        "--splits",
        help="Comma-separated training splits to use for bi-encoder. "
             "Choices: 10pct, 25pct, 100pct. Default: all three.",
    ),
    skip_bm25: bool = typer.Option(
        False,
        "--skip-bm25",
        help="Skip the TF-IDF baseline.",
    ),
    skip_bi_encoder: bool = typer.Option(
        False,
        "--skip-bi-encoder",
        help="Skip the bi-encoder baseline.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Ignore cached trained models and re-train from scratch.",
    ),
    epochs: int = typer.Option(
        3,
        help="Number of training epochs for the bi-encoder.",
    ),
    batch_size: int = typer.Option(
        32,
        help="Per-device training batch size for the bi-encoder.",
    ),
) -> None:
    """
    Run all Milestone 2 baselines and write results to results/baselines.csv.
    """
    console.rule("[bold cyan]Milestone 2 — Baseline Evaluation[/bold cyan]")
    console.print(
        Panel(
            f"[bold]Splits dir:[/bold]   {SPLITS_DIR}\n"
            f"[bold]Results dir:[/bold]  {RESULTS_DIR}\n"
            f"[bold]Models dir:[/bold]   {MODELS_DIR}\n"
            f"[bold]Skip BM25:[/bold]    {skip_bm25}\n"
            f"[bold]Skip Bi-enc:[/bold]  {skip_bi_encoder}\n"
            f"[bold]No-cache:[/bold]     {no_cache}\n"
            f"[bold]Epochs:[/bold]       {epochs}\n"
            f"[bold]Batch size:[/bold]   {batch_size}",
            title="Configuration",
            border_style="blue",
        )
    )

    # ── Resolve which training splits to run ──────────────────────────────────
    if splits:
        # normalise: accept both "10pct" and "train_10pct"
        resolved_splits = [
            s if s.startswith("train_") else f"train_{s}" for s in splits
        ]
    else:
        resolved_splits = TRAIN_SPLITS

    log.info(f"Training splits: {resolved_splits}")

    # ── Load val set (always needed) ──────────────────────────────────────────
    log.info("Loading val set …")
    val_df = load_split(SPLITS_DIR, "val")
    log.info(
        f"Val set: {len(val_df):,} pairs | "
        f"matches={_n_match_pairs(val_df):,}"
    )

    all_rows: List[Dict] = []

    # ── Baseline 1: TF-IDF ───────────────────────────────────────────────────
    if not skip_bm25:
        row = run_bm25(val_df)
        all_rows.append(row)

    # ── Baseline 2: Bi-encoder (one run per training split) ──────────────────
    if not skip_bi_encoder:
        for split_name in resolved_splits:
            console.rule(
                f"[bold]Baseline 2: Bi-encoder | split=[cyan]{split_name}[/cyan][/bold]"
            )
            log.info(f"Loading training split [{split_name}] …")
            train_df = load_split(SPLITS_DIR, split_name)
            log.info(
                f"  {len(train_df):,} pairs | "
                f"{_n_match_pairs(train_df):,} match pairs"
            )
            row = run_bi_encoder_split(
                train_split=split_name,
                train_df=train_df,
                val_df=val_df,
                use_cache=not no_cache,
                epochs=epochs,
                batch_size=batch_size,
            )
            all_rows.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.rule("[bold green]Results Summary[/bold green]")
    if all_rows:
        _summary_table(all_rows)
    console.print(
        f"\n[bold green]✓[/bold green] Results saved to [cyan]{CSV_PATH}[/cyan]\n"
        "\n[bold]Next steps:[/bold]\n"
        "  - Review results/baselines.csv\n"
        "  - Confirm bi-encoder outperforms TF-IDF baseline\n"
        "  - Proceed to Milestone 3: LLM hard negative generation\n"
        "    [dim]src/wdc_hn/generation/generate_negatives.py[/dim]\n"
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
