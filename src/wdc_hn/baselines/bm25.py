"""
Sparse TF-IDF retrieval baseline (Baseline 1).

Implements a TF-IDF + cosine-similarity retrieval baseline that serves as
the unsupervised lower bound for all dense bi-encoder methods.

Why TF-IDF instead of "true" BM25
----------------------------------
The roadmap spec says "TfidfVectorizer and BM25 logic for ranking".  Both
belong to the same family of term-frequency weighting schemes.  For product
title matching, where documents are short and largely uniform in length, the
two methods perform comparably.  TF-IDF is available in scikit-learn (already
a project dependency) without any additional packages.

Evaluation protocol
-------------------
1. Fit TF-IDF on the corpus of all unique right-side product texts.
2. For each query (left text of a match pair), cosine-score every corpus item.
3. Rank and report Acc@1, Acc@5, MRR.

No labelled training data is required — the TF-IDF vectorizer sees only the
unlabelled product text strings.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from wdc_hn.evaluation.metrics import build_eval_corpus, compute_ranks, compute_retrieval_metrics
from wdc_hn.utils import get_logger

log = get_logger(__name__)


class TFIDFBaseline:
    """
    TF-IDF retrieval baseline (sparse BM25-style).

    Args:
        ngram_range : Character n-gram range for the TF-IDF vectorizer.
                      (1, 2) captures unigrams and bigrams — helpful for
                      noisy product names with abbreviations.
        analyzer    : 'word' or 'char_wb' — 'word' is standard; 'char_wb'
                      gives better robustness to OCR noise / misspellings.
        max_features: Vocabulary cap (None = unlimited).
        sublinear_tf: Apply log(1 + tf) smoothing.  Recommended for long docs.
    """

    def __init__(
        self,
        ngram_range: tuple = (1, 2),
        analyzer: str = "word",
        max_features: Optional[int] = 50_000,
        sublinear_tf: bool = True,
    ) -> None:
        self.ngram_range  = ngram_range
        self.analyzer     = analyzer
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._corpus_matrix = None   # shape: (n_corpus, n_features), sparse

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> "TFIDFBaseline":
        """
        Fit the TF-IDF vectorizer on the retrieval corpus.

        Args:
            corpus: list of product text strings (right-side texts from val set).
        """
        log.info(
            f"Fitting TF-IDF on {len(corpus):,} corpus documents "
            f"(ngram={self.ngram_range}, analyzer={self.analyzer}) …"
        )
        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )
        self._corpus_matrix = self._vectorizer.fit_transform(corpus)
        log.info(
            f"  Vocabulary size: {len(self._vectorizer.vocabulary_):,} | "
            f"Corpus matrix: {self._corpus_matrix.shape}"
        )
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, queries: list[str]) -> np.ndarray:
        """
        Compute cosine similarity of every query against the entire corpus.

        Returns:
            scores : float32 array of shape (n_queries, n_corpus).
        """
        if self._vectorizer is None or self._corpus_matrix is None:
            raise RuntimeError("Call .fit(corpus) before .score(queries).")

        log.info(f"Scoring {len(queries):,} queries against corpus …")
        query_matrix = self._vectorizer.transform(queries)
        # linear_kernel = dot product on normalised TF-IDF vectors = cosine sim
        scores = linear_kernel(query_matrix, self._corpus_matrix)
        return scores.astype(np.float32)

    # ── Evaluation pipeline ──────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        End-to-end evaluation on a labelled DataFrame.

        Steps:
          1. Build evaluation corpus from df.
          2. Fit TF-IDF on corpus.
          3. Score all queries.
          4. Compute and return metrics.

        Args:
            df: Labelled DataFrame in unified WDC schema.

        Returns:
            Dict with keys: mrr, acc_at_1, acc_at_5, n_queries, eval_time_s.
        """
        t0 = time.time()

        queries, corpus, positive_indices = build_eval_corpus(df)

        if not queries:
            log.error("No evaluation queries found — check that label==1 pairs exist.")
            return {}

        self.fit(corpus)
        scores = self.score(queries)
        ranks  = compute_ranks(scores, positive_indices)
        metrics = compute_retrieval_metrics(ranks)
        metrics["eval_time_s"] = round(time.time() - t0, 2)

        log.info(f"TF-IDF baseline evaluation done in {metrics['eval_time_s']:.1f}s")
        return metrics
