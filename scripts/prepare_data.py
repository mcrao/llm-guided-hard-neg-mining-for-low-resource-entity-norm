#!/usr/bin/env python3
"""
prepare_data.py — Milestone 2 data pipeline entry point.

Usage:
  python scripts/prepare_data.py                      # defaults
  python scripts/prepare_data.py --train-size large   # use 'large' training set
  python scripts/prepare_data.py --force              # re-download even if cached
  python scripts/prepare_data.py --smoke-test         # quick sanity check (no download)

Run from the project root:
  uv run python scripts/prepare_data.py
  OR
  python scripts/prepare_data.py   (with venv activated)

Full pipeline:
  1. Download WDC LSPC Computers (HuggingFace → WDC direct → manual fallback).
  2. Create low-resource training splits (10%, 25%, 100%).
  3. Run a smoke-test: load each split, instantiate ProductPairDataset,
     and print example items for visual inspection.
  4. Print a final summary of all data artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Make src importable when running from project root ─────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import typer
from rich.panel import Panel
from rich.table import Table

from wdc_hn.data import (
    ProductPairDataset,
    create_low_resource_splits,
    download_wdc_computers,
    load_split,
)
from wdc_hn.utils import console, get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# ── CLI ────────────────────────────────────────────────────

@app.command()
def main(
    train_size: str = typer.Option(
        "xlarge",
        help="WDC training set size: small | medium | large | xlarge",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download and re-split even if cached files exist.",
    ),
    smoke_test: bool = typer.Option(
        False,
        "--smoke-test",
        help="Skip download; run dataset sanity check on existing files only.",
    ),
    seed: int = typer.Option(42, help="Random seed for all split operations."),
    no_hf: bool = typer.Option(
        False,
        "--no-hf",
        help="Disable HuggingFace download; go directly to WDC HTTP.",
    ),
) -> None:
    """
    End-to-end data preparation pipeline for Milestone 2.
    """
    data_dir   = PROJECT_ROOT / "data"
    raw_dir    = data_dir / "raw"
    splits_dir = data_dir / "splits"

    console.rule("[bold cyan]Milestone 2 — Data Preparation Pipeline[/bold cyan]")
    console.print(
        Panel(
            f"[bold]Project root:[/bold] {PROJECT_ROOT}\n"
            f"[bold]Train size:[/bold]   computers_{train_size}\n"
            f"[bold]Seed:[/bold]         {seed}\n"
            f"[bold]Force:[/bold]        {force}",
            title="Configuration",
            border_style="blue",
        )
    )

    # ── Step 1: Download ───────────────────────────────────
    if not smoke_test:
        console.rule("[bold]Step 1 / 3 — Download WDC LSPC Computers[/bold]")
        train_path, val_path, test_path = download_wdc_computers(
            raw_dir=raw_dir,
            train_size=train_size,
            force_redownload=force,
            prefer_huggingface=not no_hf,
        )
    else:
        log.info("[yellow]Smoke-test mode — skipping download step.[/yellow]")
        train_path = raw_dir / f"computers_train_{train_size}.parquet"
        val_path   = raw_dir / "computers_val.parquet"
        test_path  = raw_dir / "computers_test.parquet"
        for p in [train_path, val_path, test_path]:
            if not p.exists():
                log.error(f"File not found for smoke-test: {p}")
                raise typer.Exit(1)

    # ── Step 2: Create splits ──────────────────────────────
    console.rule("[bold]Step 2 / 3 — Create Low-Resource Splits[/bold]")
    split_paths = create_low_resource_splits(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        splits_dir=splits_dir,
        seed=seed,
        force=force,
    )

    # ── Step 3: Dataset smoke test ─────────────────────────
    console.rule("[bold]Step 3 / 3 — Dataset Sanity Check[/bold]")

    for split_name in ["train_10pct", "train_25pct", "train_100pct", "val", "test"]:
        _smoke_test_split(splits_dir, split_name)

    # ── Final summary ──────────────────────────────────────
    console.rule("[bold green]✓ Data preparation complete[/bold green]")
    _print_artifact_table(split_paths)


# ── Helpers ────────────────────────────────────────────────

def _smoke_test_split(splits_dir: Path, split_name: str) -> None:
    """Load a split, instantiate a dataset, and print sample items."""
    df = load_split(splits_dir, split_name)

    if len(df) == 0:
        log.warning(
            f"  [{split_name}] split is empty — smoke test skipped. "
            "(This is expected for the test split when no gold-standard file "
            "is available from HuggingFace; see README for manual download.)"
        )
        return

    # Pairwise mode
    ds_pair = ProductPairDataset(df, mode="pairwise")
    sample_pair = ds_pair[0]
    log.info(f"  [pairwise / {split_name}] {ds_pair.summary()}")

    # Only run contrastive mode for training splits (need both match + non-match)
    if split_name.startswith("train"):
        try:
            ds_cont = ProductPairDataset(df, mode="contrastive")
            sample_cont = ds_cont[0]
            log.info(f"  [contrastive / {split_name}] {ds_cont.summary()}")
            console.print(
                Panel(
                    f"[bold]anchor:[/bold]   {sample_cont['anchor'][:120]}…\n"
                    f"[bold]positive:[/bold] {sample_cont['positive'][:120]}…\n"
                    f"[bold]negative:[/bold] {sample_cont['negative'][:120]}…",
                    title=f"[cyan]Triplet sample — {split_name}[/cyan]",
                    border_style="dim",
                )
            )
        except Exception as exc:
            log.warning(f"  Contrastive smoke-test skipped for {split_name}: {exc}")

    # Show one pairwise sample
    console.print(
        Panel(
            f"[bold]text_left:[/bold]  {sample_pair['text_left'][:120]}…\n"
            f"[bold]text_right:[/bold] {sample_pair['text_right'][:120]}…\n"
            f"[bold]label:[/bold]      {sample_pair['label']}",
            title=f"[cyan]Pairwise sample — {split_name}[/cyan]",
            border_style="dim",
        )
    )


def _print_artifact_table(split_paths: dict) -> None:
    table = Table(title="Data Artifacts", show_header=True, header_style="bold green")
    table.add_column("Split",     style="bold")
    table.add_column("File",      style="cyan")
    table.add_column("Exists",    justify="center")

    for name, path in sorted(split_paths.items()):
        exists = "✓" if Path(path).exists() else "✗"
        table.add_row(name, str(path), exists)

    console.print(table)
    console.print(
        "\n[bold green]Next steps:[/bold green]\n"
        "  1. Review split_stats.json in data/splits/\n"
        "  2. Proceed to Milestone 3: Baseline bi-encoder training\n"
        "     [dim]python scripts/train_baseline.py --split train_100pct[/dim]\n"
        "  3. Compare 10%/25%/100% to see the low-resource degradation curve\n"
    )


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    app()
