"""
Unit tests for src/wdc_hn/evaluation/metrics.py — compute_ranks and compute_retrieval_metrics.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wdc_hn.evaluation.metrics import compute_ranks, compute_retrieval_metrics


def test_rank_one():
    """Positive is the highest-scoring item → rank 1."""
    # 3 queries, 5 corpus items each
    scores = np.array([
        [0.9, 0.5, 0.3, 0.2, 0.1],
        [0.1, 0.8, 0.4, 0.3, 0.2],
        [0.2, 0.3, 0.1, 0.4, 0.7],
    ], dtype=np.float32)
    positives = [0, 1, 4]  # each positive is the highest scorer for its query

    ranks = compute_ranks(scores, positives)
    assert list(ranks) == [1, 1, 1]


def test_rank_last():
    """Positive is the lowest-scoring item → rank = n_corpus."""
    n_corpus = 5
    scores = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.1],  # positive at index 4 — lowest score
    ], dtype=np.float32)
    positives = [4]

    ranks = compute_ranks(scores, positives)
    assert ranks[0] == n_corpus


def test_ranks_are_one_indexed():
    """Ranks must be 1-indexed (minimum rank is 1, not 0)."""
    scores = np.ones((4, 4), dtype=np.float32)
    scores[0, 2] = 2.0  # make positive score highest for query 0
    positives = [2, 2, 2, 2]

    ranks = compute_ranks(scores, positives)
    assert all(r >= 1 for r in ranks), "All ranks must be >= 1 (1-indexed)"


def test_mrr_perfect():
    """All positives at rank 1 → MRR = 1.0."""
    scores = np.eye(4, dtype=np.float32) * 10  # diagonal is highest per row
    positives = [0, 1, 2, 3]

    ranks = compute_ranks(scores, positives)
    metrics = compute_retrieval_metrics(ranks, ks=(1, 5))

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["acc_at_1"] == pytest.approx(1.0)


def test_mrr_mixed():
    """Known ranks → verify MRR and Acc@k by hand."""
    # Query 0: positive at rank 1  → reciprocal = 1.0
    # Query 1: positive at rank 2  → reciprocal = 0.5
    # Query 2: positive at rank 5  → reciprocal = 0.2
    scores = np.array([
        [1.0, 0.5, 0.3, 0.2, 0.1],  # positive index 0, rank 1
        [0.9, 1.0, 0.3, 0.2, 0.1],  # positive index 1, rank 1 → but we want rank 2
        [0.1, 0.2, 0.3, 0.4, 0.5],  # positive index 0, rank 5
    ], dtype=np.float32)
    # Adjust so query 1 positive (idx 0) is at rank 2
    scores[1] = [0.5, 1.0, 0.3, 0.2, 0.1]  # positive idx 1, rank 1 — tweak
    # Let's construct deterministic known-rank scores instead
    n_corpus = 5
    scores = np.zeros((3, n_corpus), dtype=np.float32)
    # Query 0: positive at rank 1
    scores[0, 2] = 5.0; scores[0, 0] = 4.0; scores[0, 1] = 3.0
    positives_0 = 2  # rank 1
    # Query 1: positive at rank 2
    scores[1, 0] = 5.0; scores[1, 3] = 4.0; scores[1, 1] = 3.0
    positives_1 = 3  # rank 2
    # Query 2: positive at rank 5 (last)
    scores[2, 1] = 5.0; scores[2, 2] = 4.0; scores[2, 0] = 3.0; scores[2, 4] = 2.0
    positives_2 = 3  # index 3, which has score 0 → rank 5
    # scores[2, 3] = 0.0 (default) → rank 5 after [1,2,0,4,3]

    ranks = compute_ranks(scores, [positives_0, positives_1, positives_2])
    assert ranks[0] == 1
    assert ranks[1] == 2
    assert ranks[2] == 5

    metrics = compute_retrieval_metrics(ranks, ks=(1, 5))
    expected_mrr = (1.0 + 0.5 + 0.2) / 3
    assert metrics["mrr"] == pytest.approx(expected_mrr, abs=1e-4)
    assert metrics["acc_at_1"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["acc_at_5"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["n_queries"] == 3
