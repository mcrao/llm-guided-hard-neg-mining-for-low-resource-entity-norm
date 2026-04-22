#!/usr/bin/env python3
"""
generate_negatives.py — LLM hard negative generation CLI.

Generates hard negatives for a training split and caches them to disk.
Re-running on the same split + type + strategy costs nothing for cached items.

Usage
-----
  # Dry run — inspect prompts without calling the API
  uv run python scripts/generate_negatives.py \\
      --split train_10pct --type component_swap --strategy zero_shot --dry-run

  # Real run — generates and caches negatives
  uv run python scripts/generate_negatives.py \\
      --split train_10pct --type component_swap --strategy zero_shot \\
      --model gpt-4o-mini --n 5

  # Run all 3 strategies for one type (runs sequentially, cache warm after first)
  for strategy in zero_shot few_shot chain_of_thought; do
    uv run python scripts/generate_negatives.py \\
        --split train_10pct --type component_swap --strategy $strategy \\
        --model gpt-4o-mini
  done

  # Local model via vLLM on RunPod
  uv run python scripts/generate_negatives.py \\
      --split train_10pct --type component_swap --strategy zero_shot \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --base-url http://<runpod-ip>:8000/v1 --api-key EMPTY

Output
------
  data/negatives_cache/{model}__{type}__{strategy}.jsonl  — cache file
  data/negatives_cache/failures.jsonl                     — failed items
  data/negatives/{split}__{type}__{strategy}__{model}.parquet  — training-ready
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import typer
from rich.panel import Panel

from wdc_hn.data.splits import load_split
from wdc_hn.generation.generate_negatives import (
    build_augmented_df,
    generate_for_split,
)
from wdc_hn.utils import console, get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SPLITS_DIR  = PROJECT_ROOT / "data" / "splits"
CACHE_DIR   = PROJECT_ROOT / "data" / "negatives_cache"
NEG_DIR     = PROJECT_ROOT / "data" / "negatives"

VALID_TYPES      = ["phonetic", "component_swap", "abbreviation", "semantic_distractor", "mixed"]
VALID_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]
VALID_SPLITS     = ["train_10pct", "train_25pct", "train_100pct"]


@app.command()
def main(
    split: str = typer.Option(
        "train_10pct",
        help=f"Training split to generate negatives for. One of: {VALID_SPLITS}",
    ),
    neg_type: str = typer.Option(
        "component_swap",
        "--type",
        help=f"Hard negative type. One of: {VALID_TYPES}",
    ),
    strategy: str = typer.Option(
        "zero_shot",
        help=f"Prompting strategy. One of: {VALID_STRATEGIES}",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        help="LLM model ID (OpenAI or vLLM-served local model).",
    ),
    n: int = typer.Option(
        5,
        help="Number of hard negatives to request per product.",
    ),
    ratio: int = typer.Option(
        1,
        help="Hard negatives per match pair to include in saved parquet (1=1:1, 4=4:1).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print prompts without calling the API.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-generate even if cached.",
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        help="OpenAI API key (defaults to OPENAI_API_KEY env var).",
    ),
    base_url: str = typer.Option(
        None,
        "--base-url",
        help="Base URL for OpenAI-compatible local API (vLLM / Ollama).",
    ),
) -> None:
    # ── Validate ───────────────────────────────────────────────────────────────
    if neg_type not in VALID_TYPES:
        log.error(f"Invalid --type '{neg_type}'. Choose from: {VALID_TYPES}")
        raise typer.Exit(1)
    if strategy not in VALID_STRATEGIES:
        log.error(f"Invalid --strategy '{strategy}'. Choose from: {VALID_STRATEGIES}")
        raise typer.Exit(1)
    if split not in VALID_SPLITS:
        log.error(f"Invalid --split '{split}'. Choose from: {VALID_SPLITS}")
        raise typer.Exit(1)

    # Resolve API key
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not dry_run and not resolved_key and not base_url:
        log.error("No API key found. Set OPENAI_API_KEY or pass --api-key.")
        raise typer.Exit(1)

    # ── Header ─────────────────────────────────────────────────────────────────
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE[/green]"
    console.print(Panel(
        f"[bold]Hard Negative Generation[/bold] — {mode}\n"
        f"Split: [cyan]{split}[/cyan] | "
        f"Type: [cyan]{neg_type}[/cyan] | "
        f"Strategy: [cyan]{strategy}[/cyan] | "
        f"Model: [cyan]{model}[/cyan] | "
        f"n={n}",
        expand=False,
    ))

    # ── Load split ─────────────────────────────────────────────────────────────
    log.info(f"Loading split [{split}] …")
    split_df = load_split(SPLITS_DIR, split)
    n_match  = (split_df["label"] == 1).sum()
    log.info(f"Split loaded: {len(split_df):,} pairs | {n_match:,} match pairs")

    # ── Generate ───────────────────────────────────────────────────────────────
    negatives_df = generate_for_split(
        split_df=split_df,
        neg_type=neg_type,
        strategy=strategy,
        model_id=model,
        n_per_product=n,
        cache_dir=CACHE_DIR,
        dry_run=dry_run,
        force=force,
        api_key=resolved_key,
        base_url=base_url,
    )

    if dry_run or negatives_df.empty:
        log.info("Dry run complete — no files written.")
        return

    # ── Save training-ready parquet ────────────────────────────────────────────
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    out_path = NEG_DIR / f"{split}__{neg_type}__{strategy}__{safe_model}.parquet"

    triplet_df = build_augmented_df(split_df, negatives_df, ratio=ratio)
    triplet_df.to_parquet(out_path, index=False)

    log.info(
        f"Saved {len(triplet_df):,} triplets → [cyan]{out_path.relative_to(PROJECT_ROOT)}[/cyan]"
    )
    console.print(Panel(
        f"[green]Done.[/green]\n"
        f"Negatives: {len(negatives_df):,} rows | "
        f"Triplets (ratio {ratio}:1): {len(triplet_df):,} rows\n"
        f"Cache: [cyan]{CACHE_DIR.relative_to(PROJECT_ROOT)}[/cyan]\n"
        f"Triplets: [cyan]{out_path.relative_to(PROJECT_ROOT)}[/cyan]",
        expand=False,
    ))


if __name__ == "__main__":
    app()
