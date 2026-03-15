"""
Bi-encoder baseline with in-batch negatives (Baseline 2).

Architecture
------------
distilbert-base-uncased fine-tuned as a Siamese sentence encoder using the
sentence-transformers library (>= 3.0).

Training objective
------------------
MultipleNegativesRankingLoss (MNRL) — treats every other positive in the
same batch as a hard negative.  This is the standard in-batch negative
paradigm from Henderson et al. (2017), popularised by DPR (Karpukhin et al.,
2020) and SimCSE (Gao et al., 2021).

Only label==1 pairs (anchor, positive) are used for training.  No external
negative mining is performed at this stage — that is the contribution of
Milestone 3.

Evaluation protocol (mirrors TFIDFBaseline for fair comparison)
----------------------------------------------------------------
1. Encode all unique right-side texts in the val set as a corpus.
2. For each query (left text of a match pair), cosine-rank the corpus.
3. Report Acc@1, Acc@5, MRR.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from wdc_hn.data.dataset import build_text
from wdc_hn.evaluation.metrics import build_eval_corpus, compute_ranks, compute_retrieval_metrics
from wdc_hn.utils import get_logger

log = get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_EPOCHS     = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_MAX_LENGTH   = 128


def _detect_device() -> str:
    """Return the best available device string for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Baseline class ─────────────────────────────────────────────────────────────

class BiEncoderBaseline:
    """
    Sentence-transformer bi-encoder trained with in-batch negatives.

    Args:
        base_model : HuggingFace model name for the encoder backbone.
        device     : 'cuda', 'mps', or 'cpu'.  Auto-detected if None.
        max_length : Maximum token length for the encoder.
    """

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        device: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.base_model = base_model
        self.device     = device or _detect_device()
        self.max_length = max_length
        self._model     = None  # lazy-initialised on first train() or load()

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_model(self) -> None:
        """Load the sentence-transformer model onto the target device."""
        from sentence_transformers import SentenceTransformer

        log.info(
            f"Loading sentence-transformer [{self.base_model}] "
            f"on device=[cyan]{self.device}[/cyan] …"
        )
        self._model = SentenceTransformer(
            self.base_model,
            device=self.device,
        )
        # Enforce max sequence length
        self._model.max_seq_length = self.max_length

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        warmup_ratio: float = DEFAULT_WARMUP_RATIO,
        seed: int = 42,
    ) -> "BiEncoderBaseline":
        """
        Fine-tune the bi-encoder on positive pairs using MNRL.

        Args:
            train_df   : Labelled training DataFrame (uses label==1 rows).
            output_dir : If given, save the trained model here.
            epochs     : Number of training epochs.
            batch_size : Per-device training batch size.
            warmup_ratio: Fraction of steps used for linear warmup.
            seed       : Random seed for reproducibility.

        Returns:
            self (for chaining).
        """
        from datasets import Dataset as HFDataset
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import SentenceTransformerTrainingArguments
        from sentence_transformers.trainer import SentenceTransformerTrainer

        # ── Build training pairs ───────────────────────────────────────────
        match_df = train_df[train_df["label"] == 1].reset_index(drop=True)
        n_pairs  = len(match_df)

        if n_pairs == 0:
            raise ValueError("train_df contains no positive pairs (label==1).")

        log.info(
            f"Building training pairs from {n_pairs:,} match rows "
            f"(epochs={epochs}, batch_size={batch_size}) …"
        )

        anchors   = [
            build_text(
                title=row.get("title_left"),
                brand=row.get("brand_left"),
                description=row.get("description_left"),
            )
            for _, row in match_df.iterrows()
        ]
        positives = [
            build_text(
                title=row.get("title_right"),
                brand=row.get("brand_right"),
                description=row.get("description_right"),
            )
            for _, row in match_df.iterrows()
        ]

        train_dataset = HFDataset.from_dict({
            "anchor":   anchors,
            "positive": positives,
        })

        # ── Initialise model ───────────────────────────────────────────────
        self._init_model()
        loss = MultipleNegativesRankingLoss(self._model)

        # ── Training arguments ─────────────────────────────────────────────
        total_steps  = (n_pairs // batch_size) * epochs
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        # fp16 / bf16 — use bf16 on CUDA if available; skip on MPS/CPU
        use_fp16 = False
        use_bf16 = False
        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                use_bf16 = True
            else:
                use_fp16 = True

        _output_dir = str(output_dir) if output_dir else "models/bi_encoder_tmp"

        args = SentenceTransformerTrainingArguments(
            output_dir=_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=warmup_steps,
            seed=seed,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_drop_last=True,   # ensures consistent batch sizes for MNRL
            logging_steps=max(1, total_steps // 20),
            save_strategy="no",          # no mid-run checkpoints to save disk space
            report_to="none",            # disable wandb/tensorboard for this baseline
        )

        # ── Train ──────────────────────────────────────────────────────────
        log.info(
            f"Training for {epochs} epoch(s) | "
            f"{total_steps} steps | {warmup_steps} warmup steps …"
        )
        t0 = time.time()

        trainer = SentenceTransformerTrainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()

        self._train_time_s = round(time.time() - t0, 1)
        log.info(f"Training complete in {self._train_time_s:.1f}s")

        # ── Save ──────────────────────────────────────────────────────────
        if output_dir is not None:
            self.save(output_dir)

        return self

    # ── Evaluation ───────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to L2-normalised embeddings.

        Returns:
            float32 array of shape (n_texts, hidden_dim).
        """
        if self._model is None:
            raise RuntimeError("Model not initialised.  Call train() or load() first.")

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → dot product = cosine sim
        )

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """
        End-to-end evaluation on a labelled DataFrame.

        Steps:
          1. Build evaluation corpus from val_df.
          2. Encode queries and corpus with the bi-encoder.
          3. Rank by cosine similarity (dot product of normalised embeddings).
          4. Compute and return metrics.

        Args:
            val_df : Labelled DataFrame in unified WDC schema.

        Returns:
            Dict with keys: mrr, acc_at_1, acc_at_5, n_queries, eval_time_s.
        """
        t0 = time.time()

        queries, corpus, positive_indices = build_eval_corpus(val_df)

        if not queries:
            log.error("No evaluation queries found — check that label==1 pairs exist.")
            return {}

        log.info(f"Encoding {len(corpus):,} corpus texts …")
        corpus_emb = self.encode(corpus, show_progress=True)

        log.info(f"Encoding {len(queries):,} query texts …")
        query_emb  = self.encode(queries, show_progress=True)

        # Cosine similarity: dot product of L2-normalised embeddings
        scores = (query_emb @ corpus_emb.T).astype(np.float32)

        ranks   = compute_ranks(scores, positive_indices)
        metrics = compute_retrieval_metrics(ranks)
        metrics["eval_time_s"] = round(time.time() - t0, 2)

        log.info(f"Bi-encoder evaluation done in {metrics['eval_time_s']:.1f}s")
        return metrics

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save the sentence-transformer model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        log.info(f"Model saved to [cyan]{path}[/cyan]")

    def load(self, path: Path) -> "BiEncoderBaseline":
        """Load a previously saved sentence-transformer model."""
        from sentence_transformers import SentenceTransformer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        log.info(f"Loading model from [cyan]{path}[/cyan] …")
        self._model = SentenceTransformer(str(path), device=self.device)
        self._model.max_seq_length = self.max_length
        return self
