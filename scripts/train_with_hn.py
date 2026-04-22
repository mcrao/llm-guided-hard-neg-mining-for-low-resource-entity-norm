#!/usr/bin/env python3
"""
train_with_hn.py — Train bi-encoder with LLM-generated hard negatives (M3).

Loads a cached negatives parquet produced by generate_negatives.py,
builds explicit triplets (anchor, positive, hard_negative), fine-tunes the
bi-encoder using MNRL, evaluates on the val set, and appends results to
results/m3_results.csv.

Usage
-----
  # Single run: component_swap × zero_shot on train_10pct
  uv run python scripts/train_with_hn.py \\
      --split train_10pct --type component_swap --strategy zero_shot \\
      --model gpt-4o-mini

  # Sweep all 3 strategies for component_swap
  for strategy in zero_shot few_shot chain_of_thought; do
    uv run python scripts/train_with_hn.py \\
        --split train_10pct --type component_swap --strategy $strategy \\
        --model gpt-4o-mini
  done

  # Specify ratio (A5 ablation: 1:1 vs 4:1 vs 1:0)
  uv run python scripts/train_with_hn.py \\
      --split train_10pct --type component_swap --strategy zero_shot \\
      --model gpt-4o-mini --ratio 4

Output
------
  results/m3_results.csv
    Columns: split, neg_type, strategy, llm_model, backbone, ratio,
             acc_at_1, acc_at_5, mrr, n_train_match, n_hard_neg,
             training_time_s, eval_time_s, timestamp
  models/bi_encoder_hn_{split}__{neg_type}__{strategy}/
    Saved model checkpoint.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import typer
from rich.panel import Panel
from rich.table import Table

from wdc_hn.baselines.bi_encoder_in_batch import BiEncoderBaseline
from wdc_hn.data.splits import load_split
from wdc_hn.generation.generate_negatives import build_augmented_df
from wdc_hn.utils import console, get_logger

import pandas as pd

log = get_logger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SPLITS_DIR  = PROJECT_ROOT / "data" / "splits"
NEG_DIR     = PROJECT_ROOT / "data" / "negatives"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR  = PROJECT_ROOT / "models"

RESULTS_CSV = RESULTS_DIR / "m3_results.csv"
RESULTS_COLS = [
    "split", "neg_type", "strategy", "llm_model", "backbone", "ratio",
    "acc_at_1", "acc_at_5", "mrr", "n_train_match", "n_hard_neg",
    "training_time_s", "eval_time_s", "timestamp",
]

VALID_SPLITS     = ["train_10pct", "train_25pct", "train_100pct"]
VALID_TYPES      = ["phonetic", "component_swap", "abbreviation", "semantic_distractor", "mixed"]
VALID_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]


def _append_result(row: dict) -> None:
    """Append one result row to m3_results.csv (creates file with header if needed)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@app.command()
def main(
    split: str = typer.Option("train_10pct", help="Training split name."),
    neg_type: str = typer.Option(
        "component_swap", "--type", help="Hard negative type."
    ),
    strategy: str = typer.Option("zero_shot", help="Prompting strategy."),
    model: str = typer.Option("gpt-4o-mini", help="Generator LLM model ID."),
    backbone: str = typer.Option(
        "distilbert-base-uncased", help="Bi-encoder backbone (A1 ablation)."
    ),
    ratio: int = typer.Option(
        1, help="Hard negatives per match pair (1=1:1, 4=4:1, 0=in-batch only)."
    ),
    epochs: int = typer.Option(3, help="Training epochs."),
    batch_size: int = typer.Option(32, help="Training batch size."),
    seed: int = typer.Option(42, help="Random seed."),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Re-train even if a saved model exists."
    ),
) -> None:
    # ── Resolve negatives parquet path ─────────────────────────────────────────
    safe_model = model.replace("/", "_")
    neg_path = NEG_DIR / f"{split}__{neg_type}__{strategy}__{safe_model}.parquet"

    if ratio > 0 and not neg_path.exists():
        log.error(
            f"Negatives file not found: {neg_path}\n"
            f"Run generate_negatives.py first:\n"
            f"  uv run python scripts/generate_negatives.py "
            f"--split {split} --type {neg_type} --strategy {strategy} --model {model}"
        )
        raise typer.Exit(1)

    # ── Header ─────────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Train with Hard Negatives[/bold]\n"
        f"Split: [cyan]{split}[/cyan] | "
        f"Type: [cyan]{neg_type}[/cyan] | "
        f"Strategy: [cyan]{strategy}[/cyan] | "
        f"Model: [cyan]{model}[/cyan] | "
        f"Ratio: [cyan]{ratio}:1[/cyan] | "
        f"Backbone: [cyan]{backbone}[/cyan]",
        expand=False,
    ))

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info(f"Loading split [{split}] …")
    train_df = load_split(SPLITS_DIR, split)
    val_df   = load_split(SPLITS_DIR, "val")
    n_match  = (train_df["label"] == 1).sum()

    hard_neg_df: Optional[pd.DataFrame] = None
    n_hard_neg = 0

    if ratio > 0:
        log.info(f"Loading negatives from {neg_path.name} …")
        neg_records_df = pd.read_parquet(neg_path)
        # neg_path stores triplets already built at ratio=1 by generate_negatives.py.
        # Re-build with the requested ratio to ensure correct sampling.
        negatives_raw = pd.read_parquet(neg_path).rename(
            columns={"anchor_text": "anchor_text",
                     "positive_text": "positive_text",
                     "negative_text": "negative_text"}
        )
        # negatives_raw is already a triplets DataFrame — just re-sample for ratio
        hard_neg_df = build_augmented_df(train_df, _triplets_to_negatives_df(negatives_raw), ratio=ratio)
        n_hard_neg = len(hard_neg_df)
        log.info(f"Hard negatives loaded: {n_hard_neg:,} triplets (ratio {ratio}:1)")

    # ── Model path ─────────────────────────────────────────────────────────────
    safe_backbone = backbone.replace("/", "_")
    run_tag  = f"{split}__{neg_type}__{strategy}__{safe_model}__{safe_backbone}__r{ratio}"
    model_dir = MODELS_DIR / f"bi_encoder_hn_{run_tag}"

    # ── Train ──────────────────────────────────────────────────────────────────
    baseline = BiEncoderBaseline(base_model=backbone)

    if model_dir.exists() and not no_cache:
        log.info(f"Loading cached model from {model_dir} …")
        baseline.load(model_dir)
        train_time = 0.0
    else:
        t0 = time.time()
        baseline.train(
            train_df=train_df,
            hard_neg_df=hard_neg_df,
            output_dir=model_dir,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        train_time = round(time.time() - t0, 1)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    log.info("Evaluating on val set …")
    metrics = baseline.evaluate(val_df)

    # ── Print results ──────────────────────────────────────────────────────────
    table = Table(title="M3 Results", show_header=True)
    table.add_column("Metric"); table.add_column("Value")
    table.add_row("Acc@1",     f"{metrics['acc_at_1']:.4f}")
    table.add_row("Acc@5",     f"{metrics['acc_at_5']:.4f}")
    table.add_row("MRR",       f"{metrics['mrr']:.4f}")
    table.add_row("N queries", str(int(metrics['n_queries'])))
    table.add_row("Train time", f"{train_time:.1f}s")
    console.print(table)

    # Baseline reference
    log.info(
        "Baseline (in-batch only): Acc@1=0.0433 | Acc@5=0.5108 | MRR=0.2486 "
        "(train_10pct, distilbert)"
    )

    # ── Append to CSV ──────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _append_result({
        "split":           split,
        "neg_type":        neg_type,
        "strategy":        strategy,
        "llm_model":       model,
        "backbone":        backbone,
        "ratio":           ratio,
        "acc_at_1":        round(metrics["acc_at_1"], 4),
        "acc_at_5":        round(metrics["acc_at_5"], 4),
        "mrr":             round(metrics["mrr"], 4),
        "n_train_match":   int(n_match),
        "n_hard_neg":      n_hard_neg,
        "training_time_s": train_time,
        "eval_time_s":     round(metrics.get("eval_time_s", 0), 2),
        "timestamp":       ts,
    })
    log.info(f"Results appended to [cyan]{RESULTS_CSV.relative_to(PROJECT_ROOT)}[/cyan]")


def _triplets_to_negatives_df(triplets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a triplets DataFrame (anchor_text, positive_text, negative_text, pair_id)
    back to the negatives_df format expected by build_augmented_df
    (pair_id, negative_text).
    """
    return triplets_df[["pair_id", "negative_text"]].copy()


if __name__ == "__main__":
    app()
