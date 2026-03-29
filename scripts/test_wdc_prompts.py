#!/usr/bin/env python3
"""
test_wdc_prompts.py — Preview LLM prompts on real WDC product pairs.

Loads product texts from the val split and prints all three prompt
strategies × all four negative types for a sample of products.
No API calls — purely for visual inspection before sending to mentor.

Always writes a full plain-text copy to manuscript/milestone3/wdc_prompts_preview.md
so the complete output can be read without terminal truncation.

Usage
-----
  uv run python scripts/test_wdc_prompts.py               # 5 random products
  uv run python scripts/test_wdc_prompts.py --n 10        # 10 products
  uv run python scripts/test_wdc_prompts.py --type component_swap  # one type
  uv run python scripts/test_wdc_prompts.py --strategy few_shot    # one strategy
  uv run python scripts/test_wdc_prompts.py --seed 0      # reproducible sample
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import typer
from rich.panel import Panel
from rich.rule import Rule

from wdc_hn.data.splits import load_split
from wdc_hn.data.dataset import build_text
from wdc_hn.generation.prompts import build_prompt, _TYPE_INSTRUCTIONS
from wdc_hn.utils import console, get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
MANUSCRIPT3  = PROJECT_ROOT / "manuscript" / "milestone3"
OUTPUT_FILE  = MANUSCRIPT3 / "wdc_prompts_preview.md"

ALL_TYPES      = list(_TYPE_INSTRUCTIONS.keys())
ALL_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]


@app.command()
def main(
    n: int = typer.Option(5, help="Number of products to sample."),
    negative_type: str = typer.Option(
        None,
        "--type",
        help=f"Negative type to show. Choices: {ALL_TYPES}. Default: all.",
    ),
    strategy: str = typer.Option(
        None,
        "--strategy",
        help=f"Prompt strategy. Choices: {ALL_STRATEGIES}. Default: all.",
    ),
    seed: int = typer.Option(42, help="Random seed for product sampling."),
    split: str = typer.Option(
        "val",
        help="Which split to sample products from.",
    ),
) -> None:
    """
    Preview hard-negative prompts on real WDC product descriptions.
    """
    # ── Resolve what to show ──────────────────────────────────────────────────
    types_to_show = [negative_type] if negative_type else ALL_TYPES
    strategies_to_show = [strategy] if strategy else ALL_STRATEGIES

    for t in types_to_show:
        if t not in ALL_TYPES:
            console.print(f"[red]Unknown type '{t}'. Choices: {ALL_TYPES}[/red]")
            raise typer.Exit(1)
    for s in strategies_to_show:
        if s not in ALL_STRATEGIES:
            console.print(f"[red]Unknown strategy '{s}'. Choices: {ALL_STRATEGIES}[/red]")
            raise typer.Exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading split '{split}' …")
    df = load_split(SPLITS_DIR, split)

    # Sample from label==1 pairs (real product matches → left side is a query)
    match_df = df[df["label"] == 1].drop_duplicates(subset=["title_left"])
    sample = match_df.sample(n=min(n, len(match_df)), random_state=seed)

    console.rule(
        f"[bold cyan]WDC Hard-Negative Prompt Preview[/bold cyan] "
        f"— {n} products × {len(types_to_show)} type(s) × {len(strategies_to_show)} strategy(ies)"
    )

    # Accumulate markdown lines for the output file in parallel with terminal output
    md_lines: list[str] = [
        "# WDC Hard-Negative Prompt Preview",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Split:** `{split}` | **Products:** {n} | "
        f"**Types:** {', '.join(types_to_show)} | "
        f"**Strategies:** {', '.join(strategies_to_show)}",
        "",
        "---",
        "",
    ]

    for idx, (_, row) in enumerate(sample.iterrows(), start=1):
        product_text = build_text(
            title=row.get("title_left"),
            brand=row.get("brand_left"),
            description=row.get("description_left"),
        )

        console.print(Rule(f"[bold]Product {idx}/{n}[/bold]"))
        console.print(
            Panel(
                product_text,
                title="[cyan]Product text[/cyan]",
                border_style="dim",
            )
        )

        md_lines += [
            f"## Product {idx} / {n}",
            "",
            f"**Product text:**",
            "",
            f"```",
            product_text,
            f"```",
            "",
        ]

        for neg_type in types_to_show:
            for strat in strategies_to_show:
                prompt = build_prompt(
                    product_text=product_text,
                    negative_type=neg_type,
                    strategy=strat,
                    n_negatives=5,
                )
                console.print(
                    Panel(
                        prompt,
                        title=(
                            f"[yellow]{neg_type}[/yellow] / "
                            f"[green]{strat}[/green]"
                        ),
                        border_style="blue",
                    )
                )

                md_lines += [
                    f"### `{neg_type}` / `{strat}`",
                    "",
                    "```",
                    prompt,
                    "```",
                    "",
                ]

        md_lines += ["---", ""]

    # ── Write markdown file ───────────────────────────────────────────────────
    MANUSCRIPT3.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(md_lines), encoding="utf-8")
    log.info(f"[green]✓[/green] Full prompts saved to [cyan]{OUTPUT_FILE}[/cyan]")

    console.print(
        "\n[bold green]Done.[/bold green] "
        "Review the prompts above and share with mentor for approval "
        "before running the full generation pipeline.\n"
        f"\n[bold]Full output saved to:[/bold] [cyan]{OUTPUT_FILE}[/cyan]\n"
        "\n[bold]Next step:[/bold]\n"
        "  1. Select the best strategy + type combination\n"
        "  2. Implement generate_negatives.py to call the LLM API\n"
        "  3. Cache results to data/negatives_cache/\n"
        "  4. Retrain bi-encoder with hard negatives injected\n"
    )


if __name__ == "__main__":
    app()
