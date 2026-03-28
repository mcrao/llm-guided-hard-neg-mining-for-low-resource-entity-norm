"""
Low-resource training split creator.

Creates stratified subsets of the training data at three sizes:
  - 10%  → simulates very low-resource setting (hardest for bi-encoder)
  - 25%  → moderate low-resource
  - 100% → full training data (upper-bound baseline)

Key design decisions aligned with the research hypothesis:
  - Stratified sampling preserves the match/non-match ratio.
  - Hard negatives are preserved proportionally in each split.
  - Splits are deterministic (fixed seed) for reproducibility.
  - Val and test sets are NEVER subsampled — always use 100%.

Output files (under data/splits/):
  computers_train_10pct.parquet
  computers_train_25pct.parquet
  computers_train_100pct.parquet   (= full training set, symlink/copy)
  computers_val.parquet
  computers_test.parquet
  split_stats.json                 (record-level counts for each split)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from wdc_hn.utils import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────

SPLIT_FRACTIONS: Dict[str, float] = {
    "10pct":  0.10,
    "25pct":  0.25,
    "100pct": 1.00,
}

DEFAULT_SEED = 42


# ── Stratified sampling ────────────────────────────────────

def _stratified_sample(
    df: pd.DataFrame,
    fraction: float,
    seed: int,
    stratify_col: str = "label",
) -> pd.DataFrame:
    """
    Return a stratified random sample of `fraction` of `df`.

    Uses sklearn's train_test_split with stratify to preserve
    the label distribution exactly.
    """
    if fraction >= 1.0:
        return df.copy()

    # train_test_split: we use the *test* portion as our sample
    _, sample = train_test_split(
        df,
        test_size=fraction,
        random_state=seed,
        stratify=df[stratify_col],
    )
    return sample.reset_index(drop=True)


# ── Split creation ─────────────────────────────────────────

def create_low_resource_splits(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    splits_dir: Path,
    seed: int = DEFAULT_SEED,
    fractions: Optional[Dict[str, float]] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Create and save low-resource training splits.

    Args:
        train_path:  Path to the full training Parquet file.
        val_path:    Path to the validation Parquet file.
        test_path:   Path to the test Parquet file.
        splits_dir:  Directory to write split Parquet files into.
        seed:        Random seed for reproducibility.
        fractions:   Dict mapping split name → fraction.
                     Defaults to {10pct: 0.10, 25pct: 0.25, 100pct: 1.00}.
        force:       If True, regenerate all split files even if they exist.

    Returns:
        Dict mapping split name → Path of saved Parquet file.
    """
    fractions = fractions or SPLIT_FRACTIONS
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading full training data from [cyan]{train_path}[/cyan] …")
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    _check_label_balance(train_df, "full training set")

    # ── Derive test holdout when gold-standard file unavailable ───────────
    test_was_empty = len(test_df) == 0
    if test_was_empty:
        log.warning(
            "Test split is empty (gold-standard file unavailable). "
            "Carving a stratified 10% holdout from the full training set. "
            "Training splits will be created from the remaining 90%. "
            "Pass --force to re-run with this behaviour if splits already exist."
        )
        train_df, test_df = train_test_split(
            train_df,
            test_size=0.10,
            random_state=seed,
            stratify=train_df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        log.info(
            f"  Test holdout: {len(test_df):,} pairs | "
            f"match={int((test_df['label'] == 1).sum()):,} | "
            f"non-match={int((test_df['label'] == 0).sum()):,}"
        )
        log.info(
            f"  Remaining train: {len(train_df):,} pairs | "
            f"match={int((train_df['label'] == 1).sum()):,}"
        )
        _check_label_balance(train_df, "train after holdout")

    output_paths: Dict[str, Path] = {}
    stats: Dict[str, Dict] = {}

    # ── Training splits ────────────────────────────────────
    for name, frac in fractions.items():
        out_path = splits_dir / f"computers_train_{name}.parquet"

        if out_path.exists() and not force and not test_was_empty:
            log.info(f"  [{name}] Already exists — skipping (delete to regenerate).")
            output_paths[f"train_{name}"] = out_path
            continue

        log.info(f"  Creating split [bold]{name}[/bold] ({frac*100:.0f}% of training) …")
        split_df = _stratified_sample(train_df, frac, seed)
        split_df.to_parquet(out_path, index=False)
        output_paths[f"train_{name}"] = out_path

        stats[name] = _compute_stats(split_df, f"train_{name}")
        log.info(
            f"    → {len(split_df):,} pairs | "
            f"match={stats[name]['n_match']:,} | "
            f"non-match={stats[name]['n_non_match']:,} | "
            f"hard-neg={stats[name]['n_hard_neg']:,}"
        )

    # ── Val / test (always 100%) ───────────────────────────
    val_out  = splits_dir / "computers_val.parquet"
    test_out = splits_dir / "computers_test.parquet"

    if not val_out.exists() or force:
        val_df.to_parquet(val_out, index=False)
    # Always (re-)write test when it was previously empty or force is set
    if not test_out.exists() or force or test_was_empty:
        test_df.to_parquet(test_out, index=False)
        if test_was_empty:
            log.info(
                f"[green]✓[/green] Test split written: "
                f"{len(test_df):,} pairs → {test_out.name}"
            )

    output_paths["val"]  = val_out
    output_paths["test"] = test_out

    stats["val"]  = _compute_stats(val_df,  "val")
    stats["test"] = _compute_stats(test_df, "test")

    # ── Save split statistics JSON ─────────────────────────
    stats_path = splits_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Split statistics saved to [cyan]{stats_path}[/cyan]")

    _print_summary_table(stats)
    return output_paths


def _check_label_balance(df: pd.DataFrame, name: str) -> None:
    n_total   = len(df)
    n_match   = (df["label"] == 1).sum()
    n_nomatch = (df["label"] == 0).sum()
    ratio     = n_match / n_total if n_total else 0
    log.info(
        f"  [{name}] Total={n_total:,} | "
        f"match={n_match:,} ({ratio:.1%}) | non-match={n_nomatch:,}"
    )


def _compute_stats(df: pd.DataFrame, name: str) -> Dict:
    n_hard = int((df.get("is_hard_negative", pd.Series(dtype=bool)) == True).sum())
    return {
        "name":           name,
        "n_pairs":        len(df),
        "n_match":        int((df["label"] == 1).sum()),
        "n_non_match":    int((df["label"] == 0).sum()),
        "n_hard_neg":     n_hard,
        "match_fraction": round(float((df["label"] == 1).mean()), 4),
    }


def _print_summary_table(stats: Dict[str, Dict]) -> None:
    """Print a summary table of all splits."""
    from rich.table import Table
    from wdc_hn.utils import console

    table = Table(title="Split Summary", show_header=True, header_style="bold cyan")
    table.add_column("Split",       style="bold")
    table.add_column("Total Pairs", justify="right")
    table.add_column("Matches",     justify="right")
    table.add_column("Non-matches", justify="right")
    table.add_column("Hard-neg",    justify="right")
    table.add_column("Match %",     justify="right")

    for name, s in stats.items():
        table.add_row(
            s["name"],
            f"{s['n_pairs']:,}",
            f"{s['n_match']:,}",
            f"{s['n_non_match']:,}",
            f"{s['n_hard_neg']:,}",
            f"{s['match_fraction']*100:.1f}%",
        )
    console.print(table)


# ── Loading helpers ────────────────────────────────────────

def load_split(
    splits_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    """
    Load a named split as a DataFrame.

    Args:
        splits_dir:  Directory containing split Parquet files.
        split_name:  One of: 'train_10pct', 'train_25pct', 'train_100pct',
                     'val', 'test'.

    Returns:
        Loaded DataFrame.
    """
    splits_dir = Path(splits_dir)
    path_map = {
        "train_10pct":  splits_dir / "computers_train_10pct.parquet",
        "train_25pct":  splits_dir / "computers_train_25pct.parquet",
        "train_100pct": splits_dir / "computers_train_100pct.parquet",
        "val":          splits_dir / "computers_val.parquet",
        "test":         splits_dir / "computers_test.parquet",
    }
    if split_name not in path_map:
        raise ValueError(
            f"Unknown split '{split_name}'. "
            f"Choose from: {list(path_map.keys())}"
        )
    path = path_map[split_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Run prepare_data.py first to generate splits."
        )
    df = pd.read_parquet(path)
    log.info(f"Loaded split [bold]{split_name}[/bold]: {len(df):,} pairs from {path.name}")
    return df
