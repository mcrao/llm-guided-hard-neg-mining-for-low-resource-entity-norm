"""
Unit tests for src/wdc_hn/data/splits.py — holdout carving logic.

Uses a small synthetic DataFrame so no real data files are needed.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wdc_hn.data.splits import create_low_resource_splits, SPLIT_FRACTIONS

# ── Synthetic data helpers ──────────────────────────────────────────────────

_UNIFIED_COLS = [
    "id_left", "title_left", "description_left", "brand_left",
    "price_left", "priceCurrency_left",
    "id_right", "title_right", "description_right", "brand_right",
    "price_right", "priceCurrency_right",
    "label", "pair_id", "cluster_id_left", "cluster_id_right",
    "is_hard_negative",
]


def _make_df(n_match: int, n_nonmatch: int, seed: int = 0) -> pd.DataFrame:
    """Build a minimal synthetic DataFrame with the unified schema."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = n_match + n_nonmatch
    labels = [1] * n_match + [0] * n_nonmatch
    rows = []
    for i, lbl in enumerate(labels):
        rows.append({
            "id_left":            f"L{i}",
            "title_left":         f"Product {i} Left",
            "description_left":   "",
            "brand_left":         "BrandA",
            "price_left":         pd.NA,
            "priceCurrency_left": pd.NA,
            "id_right":           f"R{i}",
            "title_right":        f"Product {i} Right",
            "description_right":  "",
            "brand_right":        "BrandA",
            "price_right":        pd.NA,
            "priceCurrency_right": pd.NA,
            "label":              lbl,
            "pair_id":            f"L{i}#R{i}",
            "cluster_id_left":    pd.NA,
            "cluster_id_right":   pd.NA,
            "is_hard_negative":   False,
        })
    return pd.DataFrame(rows, columns=_UNIFIED_COLS)


def _write_parquets(tmp_path: Path, train_df, val_df, test_df):
    raw = tmp_path / "raw"
    raw.mkdir()
    train_path = raw / "computers_train_xlarge.parquet"
    val_path   = raw / "computers_val.parquet"
    test_path  = raw / "computers_test.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_path, val_path, test_path


# ── Tests ───────────────────────────────────────────────────────────────────

def test_holdout_carved_when_test_empty(tmp_path):
    """When raw test is empty, a ~10% holdout is carved from train."""
    train_df = _make_df(n_match=100, n_nonmatch=500)
    val_df   = _make_df(n_match=20,  n_nonmatch=100)
    test_df  = pd.DataFrame(columns=train_df.columns)  # empty

    train_path, val_path, test_path = _write_parquets(
        tmp_path, train_df, val_df, test_df
    )
    splits_dir = tmp_path / "splits"

    create_low_resource_splits(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        splits_dir=splits_dir,
        seed=42,
    )

    test_out = pd.read_parquet(splits_dir / "computers_test.parquet")
    train_100 = pd.read_parquet(splits_dir / "computers_train_100pct.parquet")

    # Test holdout should be ~10% of original training data
    original_n = len(train_df)
    assert len(test_out) > 0, "Test split should not be empty after carving"
    assert abs(len(test_out) - round(original_n * 0.10)) <= 2, (
        f"Expected ~{round(original_n*0.10)} test pairs, got {len(test_out)}"
    )

    # Training should be trimmed to ~90%
    assert len(train_100) + len(test_out) == original_n

    # Match ratio should be preserved in test (within rounding)
    original_ratio = train_df["label"].mean()
    test_ratio = test_out["label"].mean()
    assert abs(test_ratio - original_ratio) < 0.02, (
        f"Match ratio not preserved: original={original_ratio:.3f}, test={test_ratio:.3f}"
    )


def test_holdout_idempotent(tmp_path):
    """Second call reuses existing test split; training splits are not re-carved."""
    train_df = _make_df(n_match=100, n_nonmatch=500)
    val_df   = _make_df(n_match=20,  n_nonmatch=100)
    test_df  = pd.DataFrame(columns=train_df.columns)

    train_path, val_path, test_path = _write_parquets(
        tmp_path, train_df, val_df, test_df
    )
    splits_dir = tmp_path / "splits"

    # First call — carves holdout
    create_low_resource_splits(
        train_path=train_path, val_path=val_path, test_path=test_path,
        splits_dir=splits_dir, seed=42,
    )
    test_after_first = pd.read_parquet(splits_dir / "computers_test.parquet")
    train_mtime_first = (splits_dir / "computers_train_100pct.parquet").stat().st_mtime

    # Second call — should reuse, not re-carve
    create_low_resource_splits(
        train_path=train_path, val_path=val_path, test_path=test_path,
        splits_dir=splits_dir, seed=42,
    )
    test_after_second = pd.read_parquet(splits_dir / "computers_test.parquet")
    train_mtime_second = (splits_dir / "computers_train_100pct.parquet").stat().st_mtime

    # Test split unchanged
    assert list(test_after_first["pair_id"]) == list(test_after_second["pair_id"])
    # Training split file not re-written (mtime unchanged)
    assert train_mtime_first == train_mtime_second


def test_no_leakage(tmp_path):
    """No pair_id should appear in both train_100pct and test after carving."""
    train_df = _make_df(n_match=100, n_nonmatch=500)
    val_df   = _make_df(n_match=20,  n_nonmatch=100)
    test_df  = pd.DataFrame(columns=train_df.columns)

    train_path, val_path, test_path = _write_parquets(
        tmp_path, train_df, val_df, test_df
    )
    splits_dir = tmp_path / "splits"

    create_low_resource_splits(
        train_path=train_path, val_path=val_path, test_path=test_path,
        splits_dir=splits_dir, seed=42,
    )

    train_ids = set(pd.read_parquet(splits_dir / "computers_train_100pct.parquet")["pair_id"])
    test_ids  = set(pd.read_parquet(splits_dir / "computers_test.parquet")["pair_id"])

    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} pair_ids in both train and test"
