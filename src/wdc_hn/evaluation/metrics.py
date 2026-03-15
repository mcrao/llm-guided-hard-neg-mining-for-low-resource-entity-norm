"""
Retrieval evaluation metrics for Milestone 2 baselines.

Implements:
  - build_eval_corpus()  : prepare (queries, corpus, positive_indices) from a DataFrame
  - compute_ranks()      : find the 1-indexed rank of the positive for each query
  - acc_at_k()           : Acc@k — fraction of queries with positive in top-k
  - mrr()                : Mean Reciprocal Rank
  - compute_retrieval_metrics() : aggregate dict of all metrics

Design notes
------------
- Ranks are 1-indexed (rank 1 = top result), matching the standard MRR formula.
- All metric functions accept plain numpy arrays so they work identically for the
  TF-IDF baseline (scipy sparse scores) and the bi-encoder baseline (dense cosine).
- build_eval_corpus() de-duplicates the right-side corpus so the same physical
  product doesn't inflate the candidate pool.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from wdc_hn.data.dataset import build_text
from wdc_hn.utils import get_logger

log = get_logger(__name__)


# ── Corpus builder ─────────────────────────────────────────────────────────────

def _row_to_text(row: pd.Series, side: str) -> str:
    """Build a product text string from a DataFrame row for one side."""
    return build_text(
        title=row.get(f"title_{side}"),
        brand=row.get(f"brand_{side}"),
        description=row.get(f"description_{side}"),
        price=row.get(f"price_{side}"),
        price_currency=row.get(f"priceCurrency_{side}"),
    )


def build_eval_corpus(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Prepare evaluation data from a labelled product-pair DataFrame.

    Evaluation protocol (mirrors RocketQA / DPR):
      - Corpus  : all unique right-side product texts in the DataFrame.
      - Queries : left-side text of every matching pair (label == 1).
      - Positive: the right-side text of that matching pair.

    Args:
        df: DataFrame in unified WDC schema with columns
            title_left, brand_left, …, title_right, brand_right, …, label.

    Returns:
        queries          : list[str]  — one entry per match pair.
        corpus           : list[str]  — de-duplicated right-side texts.
        positive_indices : list[int]  — index into *corpus* for each query's
                           ground-truth positive.  Queries whose positive text
                           is not found in the corpus are silently dropped.
    """
    # Build de-duplicated corpus from all right-side texts
    all_right_texts: List[str] = [
        _row_to_text(df.iloc[i], "right") for i in range(len(df))
    ]
    # dict.fromkeys preserves insertion order (Python 3.7+) and de-duplicates
    corpus: List[str] = list(dict.fromkeys(all_right_texts))
    text_to_idx: Dict[str, int] = {text: i for i, text in enumerate(corpus)}

    # Build queries from match pairs
    match_df = df[df["label"] == 1].reset_index(drop=True)

    queries: List[str] = []
    positive_indices: List[int] = []
    n_dropped = 0

    for _, row in match_df.iterrows():
        query_text    = _row_to_text(row, "left")
        positive_text = _row_to_text(row, "right")
        pos_idx = text_to_idx.get(positive_text)

        if pos_idx is None:
            n_dropped += 1
            continue

        queries.append(query_text)
        positive_indices.append(pos_idx)

    if n_dropped:
        log.warning(
            f"build_eval_corpus: dropped {n_dropped} queries whose positive "
            "text was not found in the corpus (possible dedup collision)."
        )

    log.info(
        f"Eval corpus: {len(corpus):,} unique candidates | "
        f"{len(queries):,} evaluation queries"
    )
    return queries, corpus, positive_indices


# ── Rank computation ───────────────────────────────────────────────────────────

def compute_ranks(
    scores_matrix: np.ndarray,
    positive_indices: List[int],
) -> np.ndarray:
    """
    Compute the 1-indexed rank of the positive for every query.

    Args:
        scores_matrix    : float array of shape (n_queries, n_corpus).
                           Higher score = more similar.
        positive_indices : list[int] of length n_queries.

    Returns:
        ranks : int64 array of shape (n_queries,), 1-indexed.
    """
    n_queries = scores_matrix.shape[0]
    ranks = np.zeros(n_queries, dtype=np.int64)

    for i, pos_idx in enumerate(positive_indices):
        # argsort descending — position 0 is the top-ranked item
        sorted_idx = np.argsort(-scores_matrix[i])
        # np.where returns a tuple; [0][0] gets the scalar position
        position = int(np.where(sorted_idx == pos_idx)[0][0])
        ranks[i] = position + 1  # convert 0-indexed → 1-indexed

    return ranks


# ── Metric functions ──────────────────────────────────────────────────────────

def acc_at_k(ranks: np.ndarray, k: int) -> float:
    """Fraction of queries where the positive is ranked in the top-k."""
    return float(np.mean(ranks <= k))


def mrr(ranks: np.ndarray) -> float:
    """Mean Reciprocal Rank: (1/|Q|) * Σ (1 / rank_i)."""
    return float(np.mean(1.0 / ranks.astype(float)))


def compute_retrieval_metrics(
    ranks: np.ndarray,
    ks: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Aggregate all retrieval metrics into a single dict.

    Args:
        ranks : 1-indexed rank array of shape (n_queries,).
        ks    : k values for Acc@k (default: 1 and 5).

    Returns:
        Dict with keys: mrr, acc_at_1, acc_at_5, n_queries.
    """
    results: Dict[str, float] = {"mrr": mrr(ranks)}
    for k in ks:
        results[f"acc_at_{k}"] = acc_at_k(ranks, k)
    results["n_queries"] = float(len(ranks))

    log.info(
        f"  MRR={results['mrr']:.4f} | "
        + "  ".join(f"Acc@{k}={results[f'acc_at_{k}']:.4f}" for k in ks)
        + f"  ({int(results['n_queries'])} queries)"
    )
    return results
