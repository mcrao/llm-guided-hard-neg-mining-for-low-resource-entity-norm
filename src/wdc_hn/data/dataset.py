"""
ProductPairDataset and related utilities.

Two operating modes:
  - 'pairwise'  : Returns (left_text, right_text, label) for binary
                  classification (cross-encoder baseline).
  - 'contrastive': Returns (anchor_text, positive_text, negative_text)
                  triplets for InfoNCE / triplet loss training
                  (bi-encoder baseline and LLM-HN method).

Text representation:
  For each product side we concatenate the available text fields:
    "[TITLE] <title> [BRAND] <brand> [DESC] <description (truncated)>"
  Missing fields are omitted. This mirrors the approach used in
  BLINK (Wu et al. 2020) and RocketQA (Qu et al. 2021).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from wdc_hn.utils import get_logger

log = get_logger(__name__)

# ── Text building ──────────────────────────────────────────

_FIELD_PREFIXES = {
    "title":       "[TITLE]",
    "brand":       "[BRAND]",
    "description": "[DESC]",
    "price":       "[PRICE]",
}

MAX_DESC_CHARS = 300  # truncate long descriptions to keep within token budget


def build_text(
    title: Optional[str] = None,
    brand: Optional[str] = None,
    description: Optional[str] = None,
    price: Optional[str] = None,
    price_currency: Optional[str] = None,
    include_price: bool = False,
) -> str:
    """
    Build a single text string from product attributes.

    Example output:
      "[TITLE] Dell XPS 13 Laptop [BRAND] Dell [DESC] 13.4-inch FHD+ ..."
    """
    parts: List[str] = []

    def _clean(s: object) -> str:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return ""
        return str(s).strip()

    t = _clean(title)
    if t:
        parts.append(f"{_FIELD_PREFIXES['title']} {t}")

    b = _clean(brand)
    if b:
        parts.append(f"{_FIELD_PREFIXES['brand']} {b}")

    d = _clean(description)
    if d:
        d = d[:MAX_DESC_CHARS]  # hard truncate; tokeniser will handle the rest
        parts.append(f"{_FIELD_PREFIXES['description']} {d}")

    if include_price:
        p  = _clean(price)
        pc = _clean(price_currency)
        if p:
            price_str = f"{pc} {p}".strip() if pc else p
            parts.append(f"{_FIELD_PREFIXES['price']} {price_str}")

    return " ".join(parts) if parts else "[EMPTY]"


def _row_to_text(row: pd.Series, side: str, include_price: bool = False) -> str:
    return build_text(
        title=row.get(f"title_{side}"),
        brand=row.get(f"brand_{side}"),
        description=row.get(f"description_{side}"),
        price=row.get(f"price_{side}"),
        price_currency=row.get(f"priceCurrency_{side}"),
        include_price=include_price,
    )


# ── Pairwise Dataset ───────────────────────────────────────

class ProductPairDataset(Dataset):
    """
    Dataset for WDC LSPC product pair matching.

    Args:
        df:            DataFrame in the unified WDC schema.
        mode:          'pairwise' → (text_left, text_right, label)
                       'contrastive' → (anchor, positive, negative)
        tokenizer:     Optional HuggingFace tokenizer. If provided,
                       __getitem__ returns tokenised tensors; otherwise
                       returns raw strings for inspection / custom collation.
        max_length:    Maximum token length (only used if tokenizer given).
        include_price: Whether to include price in the product text.
        seed:          Random seed for triplet sampling in contrastive mode.

    Contrastive mode note:
        For each anchor (a product offer), we need:
          - positive: a different offer for the *same* product (label=1 pair).
          - negative: an offer for a *different* product (label=0 pair),
                      preferably a hard negative (is_hard_negative=True).
        This class builds index structures for efficient sampling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: Literal["pairwise", "contrastive"] = "pairwise",
        tokenizer=None,
        max_length: int = 128,
        include_price: bool = False,
        seed: int = 42,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_price = include_price
        self._rng = random.Random(seed)

        if mode == "contrastive":
            self._build_triplet_index()

    # ── Triplet index for contrastive mode ────────────────

    def _build_triplet_index(self) -> None:
        """
        Precompute lookup structures for fast triplet sampling.

        We treat each *row* in the DataFrame as an (anchor, candidate) pair.
        In contrastive mode each item returned is:
          anchor   = left side of a match pair (label == 1)
          positive = right side of the same match pair
          negative = right side of a non-match pair for the same left offer

        This mirrors the RocketQA / ANCE training setup.
        """
        matches    = self.df[self.df["label"] == 1].copy()
        non_matches = self.df[self.df["label"] == 0].copy()

        # Prefer hard negatives; fall back to random non-matches
        hard_neg  = non_matches[non_matches["is_hard_negative"] == True]
        soft_neg  = non_matches[non_matches["is_hard_negative"] != True]

        self._match_indices = matches.index.tolist()
        self._hard_neg_pool = hard_neg.index.tolist()
        self._soft_neg_pool = soft_neg.index.tolist()

        log.info(
            f"Triplet index built: "
            f"{len(self._match_indices):,} anchors | "
            f"{len(self._hard_neg_pool):,} hard-neg pool | "
            f"{len(self._soft_neg_pool):,} soft-neg pool"
        )

    def _sample_negative(self) -> int:
        """Sample a negative index, preferring hard negatives."""
        if self._hard_neg_pool:
            return self._rng.choice(self._hard_neg_pool)
        return self._rng.choice(self._soft_neg_pool)

    # ── Core dataset interface ─────────────────────────────

    def __len__(self) -> int:
        if self.mode == "contrastive":
            return len(self._match_indices)
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        if self.mode == "pairwise":
            return self._get_pairwise(idx)
        return self._get_contrastive(idx)

    def _get_pairwise(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        text_left  = _row_to_text(row, "left",  self.include_price)
        text_right = _row_to_text(row, "right", self.include_price)
        label      = int(row["label"])

        if self.tokenizer is None:
            return {
                "text_left":  text_left,
                "text_right": text_right,
                "label":      label,
                "pair_id":    str(row.get("pair_id", idx)),
            }

        encoding = self.tokenizer(
            text_left,
            text_right,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }

    def _get_contrastive(self, idx: int) -> Dict:
        match_idx = self._match_indices[idx]
        match_row = self.df.iloc[match_idx]

        anchor_text   = _row_to_text(match_row, "left",  self.include_price)
        positive_text = _row_to_text(match_row, "right", self.include_price)

        neg_idx     = self._sample_negative()
        neg_row     = self.df.iloc[neg_idx]
        negative_text = _row_to_text(neg_row, "right", self.include_price)

        if self.tokenizer is None:
            return {
                "anchor":   anchor_text,
                "positive": positive_text,
                "negative": negative_text,
            }

        def _encode(text: str) -> Dict[str, torch.Tensor]:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in enc.items()}

        return {
            "anchor":   _encode(anchor_text),
            "positive": _encode(positive_text),
            "negative": _encode(negative_text),
        }

    # ── Convenience factory ────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        mode: Literal["pairwise", "contrastive"] = "pairwise",
        **kwargs,
    ) -> "ProductPairDataset":
        """Load directly from a Parquet file."""
        df = pd.read_parquet(path)
        return cls(df, mode=mode, **kwargs)

    # ── Diagnostics ────────────────────────────────────────

    def summary(self) -> str:
        n = len(self.df)
        n_match  = (self.df["label"] == 1).sum()
        n_nomatch = (self.df["label"] == 0).sum()
        n_hard   = (self.df.get("is_hard_negative", pd.Series(dtype=bool)) == True).sum()
        return (
            f"ProductPairDataset | mode={self.mode} | n={n:,} | "
            f"match={n_match:,} | non-match={n_nomatch:,} | hard-neg={n_hard:,}"
        )
