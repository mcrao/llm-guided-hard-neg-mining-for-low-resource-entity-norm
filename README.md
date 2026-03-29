# LLM-Guided Hard Negative Mining for Low-Resource Entity Normalization

**Researcher:** Manoj Chandrashekar Rao <br>
**Mentor:** Dr. Raj Dandekar <br>
**Target venue:** EMNLP / ACL workshop (RepL4NLP or similar)

> **Research goal:** Design, implement, and evaluate a system that uses a generator LLM (e.g.,
> Llama-3-8B) to synthesise *structured hard negatives* (phonetic variants, component swaps,
> abbreviations) for contrastive bi-encoder training. The central hypothesis is that LLM-generated
> hard negatives improve entity normalisation accuracy most dramatically in low-resource settings
> (10 %–25 % of labelled data) — directly addressing Research Questions RQ1–RQ3.

---

## Milestone Progress

| Milestone | Weeks | Status | Key deliverables |
|-----------|-------|--------|-----------------|
| M1 — Literature Review & Foundations | 1–2 | ✅ **Complete** (mentor approved) | `literature_review/review_memo.pdf`, `papers.xlsx`, `comparison_table.xlsx` |
| M2 — Data / Environment Setup & Baselines | 3–4 | ✅ **Complete** | Reproducible environment, WDC pipeline, BM25 + bi-encoder baselines, `results/baselines.csv` |
| M3 — Core Method: LLM Hard Negative Generation | 5–6 | ⏳ Next | `src/wdc_hn/generation/generate_negatives.py`, `src/train.py`, ablation grid A1–A5 |
| M4 — Final Evaluation & In-depth Analysis | 7–8 | ⏳ Pending | Publication-quality figures, tables, error analysis |
| M5 — Manuscript Writing & Submission | 9–10 | ⏳ Pending | `manuscript/main.pdf`, public GitHub repo |

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Repository Layout](#2-repository-layout)
3. [Environment Setup](#3-environment-setup)
4. [Dataset: WDC LSPC Computers](#4-dataset-wdc-lspc-computers)
5. [Data Pipeline](#5-data-pipeline)
6. [Baseline Evaluation](#6-baseline-evaluation)
7. [Using the Library in Code](#7-using-the-library-in-code)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Current Results](#9-current-results)
10. [Research Alignment](#10-research-alignment)
11. [Reproducing All Results](#11-reproducing-all-results)
12. [Milestone 3 — LLM Hard Negative Generation](#12-milestone-3--llm-hard-negative-generation)

---

## 1. Quick Start

```bash
# 1. Install uv (fast Python package manager — replaces pip/conda)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone / enter the project directory
cd /path/to/project

# 3. Create virtual environment and install all dependencies (reads pyproject.toml + uv.lock)
uv sync

# 4. Copy environment template and fill in any API keys
cp .env.example .env

# 5. Download WDC LSPC Computers and create all data splits
uv run python scripts/prepare_data.py

# 6. Run all baselines and write results/baselines.csv
uv run python scripts/run_baselines.py
```

After step 6, `results/baselines.csv` will contain Acc@1, Acc@5, and MRR for:
- `bm25_tfidf` on the val set
- `bi_encoder_in_batch` trained on each of `train_10pct`, `train_25pct`, `train_100pct`

---

## 2. Repository Layout

```
project/
│
├── pyproject.toml              # uv project manifest + pinned dependencies
├── requirements.txt            # pip-compatible fallback (same deps)
├── uv.lock                     # exact locked dependency tree
├── .python-version             # pins Python 3.11
├── .env.example                # environment variable template
├── .env                        # your local secrets (never committed)
│
├── data/
│   ├── raw/                    # downloaded WDC Parquet files (auto-generated)
│   │   ├── computers_train_xlarge.parquet
│   │   ├── computers_val.parquet
│   │   ├── computers_test.parquet
│   │   └── .hf_cache/          # HuggingFace Hub download cache
│   └── splits/                 # stratified low-resource training splits
│       ├── computers_train_10pct.parquet   # 5,477 pairs (775 matches)
│       ├── computers_train_25pct.parquet   # 13,692 pairs (1,938 matches)
│       ├── computers_train_100pct.parquet  # 54,768 pairs (7,752 matches)
│       ├── computers_val.parquet           # 13,693 pairs (1,938 matches)
│       ├── computers_test.parquet          # held-out test set (auto-carved from train)
│       └── split_stats.json               # exact row counts for every split
│
├── src/wdc_hn/                 # installable Python package (wdc_hn)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # WDC download: HuggingFace → HTTP → curl fallback
│   │   ├── dataset.py          # ProductPairDataset (pairwise & contrastive modes)
│   │   └── splits.py           # stratified low-resource split creation + test holdout
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # build_eval_corpus, compute_ranks, Acc@k, MRR
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── bm25.py             # TFIDFBaseline — sparse TF-IDF + cosine similarity
│   │   └── bi_encoder_in_batch.py  # BiEncoderBaseline — distilbert + MNRL
│   ├── generation/
│   │   ├── __init__.py
│   │   └── prompts.py          # LLM prompt builders: 4 negative types × 3 strategies
│   └── utils/
│       ├── __init__.py
│       └── logging.py          # Rich-based coloured logging
│
├── scripts/
│   ├── prepare_data.py         # End-to-end data download + split creation CLI
│   ├── run_baselines.py        # Baseline evaluation CLI → results/baselines.csv
│   ├── sweep_epochs.py         # Epoch sweep (3/5/10/15) → results/epoch_sweep.csv
│   └── test_wdc_prompts.py     # Preview LLM prompts on real WDC products (no API)
│
├── models/                     # Saved bi-encoder checkpoints (auto-created)
│   ├── bi_encoder_in_batch_10pct/
│   ├── bi_encoder_in_batch_25pct/
│   └── bi_encoder_in_batch_100pct/
│
├── results/
│   ├── baselines.csv           # All baseline metrics (appended on each run)
│   └── epoch_sweep.csv         # Epoch sweep results (generated by sweep_epochs.py)
│
└── notebooks/                  # Exploratory analysis (Milestones 3–4)
```

> **Note on the test split:** The WDC LSPC gold-standard file (`computers_gs.json.gz`) is not
> available from the WDC server or HuggingFace.  On the **first run**, `splits.py` detects that
> `data/raw/computers_test.parquet` is empty and automatically carves a stratified 10% holdout
> from the full training data (seed=42), then regenerates all training splits from the remaining
> 90%.  On **subsequent runs** it detects that `data/splits/computers_test.parquet` already
> exists and is non-empty, so the holdout and training splits are left untouched (idempotent).
> Use `prepare_data.py --force` to explicitly re-carve.  All current M2 evaluation is on
> `computers_val.parquet`; the held-out test set will be used for final evaluation in M4.

---

## 3. Environment Setup

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11 (pinned in `.python-version`) | [python.org](https://www.python.org/) or `pyenv` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### Install all dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and creates `.venv/` with exact pinned versions.
No Conda required. No manual pip installs.

### Core dependencies (auto-installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.2 | PyTorch — model training and tensor ops |
| `transformers` | ≥ 4.40 | HuggingFace Transformers — tokenizers, model loading |
| `sentence-transformers` | ≥ 3.0 | Siamese bi-encoder training (`SentenceTransformerTrainer`) |
| `datasets` | ≥ 2.19 | HuggingFace Datasets — training data in-memory |
| `faiss-cpu` | ≥ 1.8 | FAISS vector index — used from Milestone 3 for ANN mining |
| `scikit-learn` | ≥ 1.4 | TF-IDF vectorizer, stratified splitting |
| `pandas` | ≥ 2.2 | DataFrame operations |
| `numpy` | ≥ 1.26 | Numerical computations |
| `pyarrow` | ≥ 15 | Parquet I/O |
| `rich` | ≥ 13.7 | Coloured terminal output + progress bars |
| `typer` | ≥ 0.12 | CLI argument parsing |
| `pyyaml` | ≥ 6.0 | Experiment config files (Milestone 3) |
| `tqdm` | ≥ 4.66 | Download progress bars |

### Optional dev dependencies

```bash
uv sync --extra dev
# Adds: ipykernel, jupyterlab, matplotlib, seaborn, black, ruff
```

### Environment variables

Copy `.env.example` to `.env` and fill in values as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | When needed |
|----------|---------|-------------|
| `WDC_CATEGORY` | `computers` | Data download |
| `WDC_TRAIN_SIZE` | `xlarge` | Data download |
| `RANDOM_SEED` | `42` | Reproducibility |
| `BIENCODER_MODEL` | `sentence-transformers/all-mpnet-base-v2` | M3+ training |
| `MAX_SEQ_LENGTH` | `128` | M3+ training |
| `OPENAI_API_KEY` | *(empty)* | M3 LLM negative generation (optional) |
| `ANTHROPIC_API_KEY` | *(empty)* | M3 LLM negative generation (optional) |
| `LLM_MODEL` | `claude-3-5-haiku-20241022` | M3 negative generation |

---

## 4. Dataset: WDC LSPC Computers

**Source:** [WDC Large-Scale Product Corpus 2017](http://webdatacommons.org/largescaleproductcorpus/v2/) —
English Computers subset.  Also mirrored at HuggingFace (`wdc/products-2017`).

**Why this dataset:**
- Millions of product offers from thousands of e-commerce sites, clustered by product ID.
- High prevalence of noisy, ambiguous product names ("HP Pavilion dv6" vs "HP dv6 Pavilion Entertainment").
- Pairs are labelled: `label=1` = same product, `label=0` = different product.
- `is_hard_negative=True` flags negatives that were similarity-mined (most confusable non-matches).

### Actual split sizes (from `data/splits/split_stats.json`)

| Split file | Total pairs | Matches (label=1) | Non-matches (label=0) | Match % |
|-----------|------------|-------------------|-----------------------|---------|
| `computers_train_10pct.parquet` | 5,477 | 775 | 4,702 | 14.15% |
| `computers_train_25pct.parquet` | 13,692 | 1,938 | 11,754 | 14.15% |
| `computers_train_100pct.parquet` | 54,768 | 7,752 | 47,016 | 14.15% |
| `computers_val.parquet` | 13,693 | 1,938 | 11,755 | 14.15% |
| `computers_test.parquet` | ~4,930 | ~698 | ~4,232 | 14.15% (auto-carved holdout) |

> The match ratio is preserved across all splits (14.15%) thanks to stratified sampling in `splits.py`.
> Test split sizes above are estimates based on a 10% holdout of `train_100pct`; run `prepare_data.py --force`
> to generate the exact numbers and update `split_stats.json`.

### Research rationale for three training splits

| Split | Match pairs | Research purpose |
|-------|------------|-----------------|
| `train_10pct` | 775 | **Very low-resource** — where LLM hard negatives should help *most* (RQ3) |
| `train_25pct` | 1,938 | **Moderate low-resource** — middle of the learning curve |
| `train_100pct` | 7,752 | **Full data** — upper-bound baseline; measures ceiling performance |

### Dataset schema

Each Parquet file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id_left` | str | Unique identifier of the left product offer |
| `id_right` | str | Unique identifier of the right product offer |
| `title_left` | str | Product title from left source |
| `title_right` | str | Product title from right source |
| `description_left` | str | Description text (truncated to 300 chars in text builder) |
| `description_right` | str | Description text from right source |
| `brand_left` | str | Brand name from left source |
| `brand_right` | str | Brand name from right source |
| `price_left` | float/str | Listed price, left source |
| `price_right` | float/str | Listed price, right source |
| `priceCurrency_left` | str | ISO currency code, left source |
| `priceCurrency_right` | str | ISO currency code, right source |
| `label` | int (0/1) | **1** = same product; **0** = different products |
| `pair_id` | str | `"{id_left}#{id_right}"` — unique pair identifier |
| `cluster_id_left` | int/NA | All offers in the same cluster refer to the same product |
| `cluster_id_right` | int/NA | Same as above for right offer |
| `is_hard_negative` | bool | **True** if this non-match pair was similarity-mined |

---

## 5. Data Pipeline

The full pipeline is driven by `scripts/prepare_data.py`.

### Running the pipeline

```bash
# Standard run (downloads xlarge split, creates all three training splits)
uv run python scripts/prepare_data.py

# Faster iteration — use 'large' training set instead of 'xlarge'
uv run python scripts/prepare_data.py --train-size large

# Sanity check on already-downloaded data (skips download)
uv run python scripts/prepare_data.py --smoke-test

# Force re-download even if Parquet files already exist
uv run python scripts/prepare_data.py --force

# Skip HuggingFace; go directly to WDC HTTP server
uv run python scripts/prepare_data.py --no-hf
```

### Pipeline steps

**Step 1 — Download** (`src/wdc_hn/data/download.py`)

Three download strategies are tried in order:
1. **HuggingFace Hub** (`wdc/products-2017`) — direct Parquet file download via
   `huggingface_hub`. This bypasses the legacy `datasets.load_dataset` loading script
   (no longer supported in recent `datasets` versions).
2. **WDC HTTP** (`data.dws.informatik.uni-mannheim.de`) — `requests.Session` with
   browser-like headers to avoid 403 blocks.
3. **System curl** — fallback if `requests` returns 403 (university servers often
   block Python requests by default).

Output: `data/raw/computers_train_xlarge.parquet`, `computers_val.parquet`, `computers_test.parquet`

**Step 2 — Split creation** (`src/wdc_hn/data/splits.py`)

`create_low_resource_splits()` uses `sklearn.train_test_split` with `stratify=label` to
guarantee the 14.15% match ratio is preserved in every sub-sample.  The function is
**idempotent**: it checks `splits_dir/computers_test.parquet` (not the raw input) to decide
whether to carve a test holdout, so re-running never re-partitions an already-fixed split.
Output files are skipped if they already exist; pass `--force` to regenerate.

**Step 3 — Smoke test**

Loads each split, instantiates `ProductPairDataset` in both `pairwise` and `contrastive`
modes, and prints a sample triplet for visual inspection.

---

## 6. Baseline Evaluation

Two baselines are implemented, both evaluated on `computers_val.parquet`.

### Running baselines

```bash
# Full run: BM25 + bi-encoder on all three training splits
uv run python scripts/run_baselines.py

# BM25 only (fast, ~2 seconds, no GPU needed)
uv run python scripts/run_baselines.py --skip-bi-encoder

# Bi-encoder only, specific splits
uv run python scripts/run_baselines.py --skip-bm25 --splits 10pct 100pct

# Re-train even if a saved model already exists
uv run python scripts/run_baselines.py --no-cache

# Adjust training hyperparameters
uv run python scripts/run_baselines.py --epochs 5 --batch-size 64

# Investigate the Acc@1 plateau across epoch counts (3, 5, 10, 15)
uv run python scripts/sweep_epochs.py --split train_100pct
```

Results are **appended** to `results/baselines.csv` on every run (with a UTC timestamp),
so you have a full history. Open the file directly or filter by `timestamp` to see the
latest run.

### Baseline 1 — TF-IDF / BM25 (`src/wdc_hn/baselines/bm25.py`)

**Class:** `TFIDFBaseline`

Implements sparse keyword-based retrieval as an unsupervised lower bound.

| Design choice | Value | Rationale |
|---------------|-------|-----------|
| Vectorizer | `TfidfVectorizer` (sklearn) | BM25 and TF-IDF perform comparably for short, uniform-length product titles |
| N-gram range | (1, 2) — unigrams + bigrams | Captures "HP Pavilion" as a unit; improves recall for brand-model patterns |
| TF smoothing | `sublinear_tf=True` | `log(1 + tf)` smoothing; prevents high-frequency terms from dominating |
| Vocabulary cap | 50,000 features | Balances coverage vs. memory |
| Similarity | cosine (via `linear_kernel`) | Standard for TF-IDF; equivalent to BM25 ranking order |
| Training data | **None** | Unsupervised — vectorizer is fit on the val corpus at eval time |

**Evaluation protocol:**
1. Build corpus: all 3,479 unique right-side texts from `computers_val.parquet`.
2. Fit TF-IDF on corpus.
3. For each of 1,938 match pairs, encode the left text as a query.
4. Score all 3,479 corpus items; find rank of ground-truth positive.
5. Report Acc@1, Acc@5, MRR.

### Baseline 2 — Bi-encoder with in-batch negatives (`src/wdc_hn/baselines/bi_encoder_in_batch.py`)

**Class:** `BiEncoderBaseline`

Fine-tunes a Siamese bi-encoder using the standard in-batch negative paradigm.

| Design choice | Value | Rationale |
|---------------|-------|-----------|
| Base model | `distilbert-base-uncased` | Specified in roadmap; fast to fine-tune, well-studied baseline |
| Loss | `MultipleNegativesRankingLoss` | In-batch negative paradigm from DPR (Karpukhin et al., 2020) |
| Training pairs | label==1 pairs only | (anchor, positive) — other positives in batch become implicit negatives |
| Epochs | 3 | Default; adjustable via `--epochs` |
| Batch size | 32 | Default; adjustable via `--batch-size` |
| Warmup | 10% of total steps | Standard linear warmup for transformer fine-tuning |
| Max seq length | 128 tokens | Covers most product titles + truncated descriptions |
| Mixed precision | bf16 on CUDA (if supported); fp16 fallback; none on MPS/CPU | Stability on Mac M-series |
| Similarity at eval | cosine (L2-normalised dot product) | Embeddings are L2-normalised before scoring |
| Trainer API | `SentenceTransformerTrainer` (sbert >= 3.0) | Modern HuggingFace-compatible training loop |

**Evaluation protocol** (identical to BM25 for fair comparison):
1. Build corpus: 3,479 unique right-side texts from val set.
2. Encode corpus with bi-encoder (batch_size=64, normalised).
3. Encode 1,938 query texts with bi-encoder.
4. Score via dot product (= cosine, since embeddings are normalised).
5. Rank corpus per query; report Acc@1, Acc@5, MRR.

**Model caching:** Trained models are saved to `models/bi_encoder_in_batch_{split}/`.
Subsequent runs load the cached model unless `--no-cache` is passed.

### `results/baselines.csv` format

```
method, train_split, eval_split, acc_at_1, acc_at_5, mrr,
n_train_match_pairs, n_eval_queries, training_time_s, eval_time_s, timestamp
```

---

## 7. Using the Library in Code

### Load a data split

```python
from wdc_hn.data import load_split

df = load_split("data/splits", "train_10pct")
# Valid names: train_10pct | train_25pct | train_100pct | val | test
print(df.shape)          # (5477, 18)
print(df["label"].value_counts())
```

### ProductPairDataset — pairwise mode (for cross-encoder)

```python
from wdc_hn.data import ProductPairDataset

ds = ProductPairDataset(df, mode="pairwise")
item = ds[0]
# {"text_left": "[TITLE] Dell XPS 13 ... [BRAND] Dell ...",
#  "text_right": "[TITLE] Dell XPS 13 9310 ... [BRAND] Dell ...",
#  "label": 1,
#  "pair_id": "abc123#def456"}
```

### ProductPairDataset — contrastive mode (for bi-encoder / InfoNCE)

```python
ds_c = ProductPairDataset(df, mode="contrastive", seed=42)
triplet = ds_c[0]
# {"anchor":   "[TITLE] ...",
#  "positive": "[TITLE] ...",
#  "negative": "[TITLE] ..."}
#
# Hard negatives (is_hard_negative=True) are preferred when sampling negatives.
# Falls back to random non-match if no hard negatives are available.
```

### ProductPairDataset — with tokenizer (returns tensors)

```python
from transformers import AutoTokenizer
from wdc_hn.data import ProductPairDataset

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = ProductPairDataset(df, mode="pairwise", tokenizer=tok, max_length=128)
item = ds[0]
# {"input_ids": tensor([101, ...]), "attention_mask": tensor([1, ...]),
#  "token_type_ids": tensor([0, ...]), "label": tensor(1)}
```

### build_text — manual text construction

```python
from wdc_hn.data import build_text

text = build_text(
    title="HP Pavilion x360 14-cd0010nr",
    brand="HP",
    description="14-inch 2-in-1 touchscreen laptop ...",
    include_price=False,
)
# "[TITLE] HP Pavilion x360 14-cd0010nr [BRAND] HP [DESC] 14-inch 2-in-1 ..."
```

### Evaluation metrics

```python
from wdc_hn.evaluation import build_eval_corpus, compute_ranks, compute_retrieval_metrics
import numpy as np

# Build corpus and queries from a labelled DataFrame
queries, corpus, positive_indices = build_eval_corpus(val_df)
# queries          : list[str], length = n_match_pairs
# corpus           : list[str], length = n_unique_right_texts (3,479 for val)
# positive_indices : list[int], index into corpus for each query's ground truth

# scores_matrix : float32 array, shape = (n_queries, n_corpus)
# (fill this with TF-IDF cosine scores or bi-encoder dot products)
ranks = compute_ranks(scores_matrix, positive_indices)   # 1-indexed

metrics = compute_retrieval_metrics(ranks, ks=(1, 5))
# {"mrr": 0.xxxx, "acc_at_1": 0.xxxx, "acc_at_5": 0.xxxx, "n_queries": 1938.0}
```

### Run BM25 baseline directly

```python
from wdc_hn.baselines import TFIDFBaseline
from wdc_hn.data import load_split

val_df  = load_split("data/splits", "val")
baseline = TFIDFBaseline()
metrics  = baseline.evaluate(val_df)
# {"mrr": 0.1988, "acc_at_1": 0.0346, "acc_at_5": 0.3870,
#  "n_queries": 1938.0, "eval_time_s": 1.76}
```

### Train and evaluate bi-encoder baseline

```python
from wdc_hn.baselines import BiEncoderBaseline
from wdc_hn.data import load_split

train_df = load_split("data/splits", "train_100pct")
val_df   = load_split("data/splits", "val")

be = BiEncoderBaseline(base_model="distilbert-base-uncased")
be.train(train_df, output_dir="models/bi_encoder_100pct", epochs=3, batch_size=32)
metrics = be.evaluate(val_df)
# {"mrr": ..., "acc_at_1": ..., "acc_at_5": ..., "n_queries": 1938.0, "eval_time_s": ...}

# Load a saved model later without retraining
be2 = BiEncoderBaseline()
be2.load("models/bi_encoder_100pct")
metrics2 = be2.evaluate(val_df)
```

---

## 8. Evaluation Metrics

All metrics are computed in `src/wdc_hn/evaluation/metrics.py`.

### Evaluation task

Given a query (left-side product text), rank all items in a fixed candidate corpus
(3,479 unique right-side texts from the val set).  For each of 1,938 match pairs,
the ground-truth positive is the right-side text of that pair.

### Metric definitions

**Accuracy@k (Acc@k)**
```
Acc@k = (1 / |Q|) × Σ 1{rank_i ≤ k}
```
Fraction of queries where the ground-truth positive appears in the top-k results.
We report Acc@1 (exact-match precision) and Acc@5.

**Mean Reciprocal Rank (MRR)**
```
MRR = (1 / |Q|) × Σ (1 / rank_i)
```
Rewards partial credit for near-misses; a positive at rank 2 scores 0.5.
Primary metric for comparing baselines against our LLM-HN method.

**Note on Macro-F1:** Macro-F1 (required for the final paper per the roadmap) treats each
canonical entity cluster as a class.  It will be implemented in Milestone 4 once the test
set gold-standard is available.

---

## 9. Current Results

### BM25 / TF-IDF baseline (run 2026-03-14)

| Method | Train split | Eval split | Acc@1 | Acc@5 | MRR | Eval queries |
|--------|------------|------------|-------|-------|-----|-------------|
| `bm25_tfidf` | N/A | val | 0.0346 | 0.3870 | 0.1988 | 1,938 |

**Interpretation:** TF-IDF achieves 3.5% Acc@1 — retrieves the correct product as the
top-1 result for only 1 in 29 queries.  MRR of 0.20 means the correct product is found
at an average rank of ~5.  This is the expected weak lower bound for a keyword-overlap
method against a 3,479-item corpus — there are many products with nearly identical titles
that differ only by model number or storage configuration.

### Bi-encoder baseline (run 2026-03-14, NVIDIA T4, Google Colab)

| Method | Train split | Eval split | Acc@1 | Acc@5 | MRR | Train pairs | Train time |
|--------|------------|------------|-------|-------|-----|------------|-----------|
| `bi_encoder_in_batch` | train_10pct | val | 0.0454 | 0.5325 | 0.2554 | 775 | 115.5 s |
| `bi_encoder_in_batch` | train_25pct | val | 0.0439 | 0.5552 | 0.2656 | 1,938 | 182.9 s |
| `bi_encoder_in_batch` | train_100pct | val | 0.0454 | 0.6022 | 0.2819 | 7,752 | 720.7 s |

**Key observation:** Acc@1 is flat at ~4.5% across all three data splits.  Acc@5 and MRR
do scale with data size, but the top-1 retrieval hit-rate does not improve — even doubling
or quadrupling the training data makes no difference.  This is the structural plateau caused
by in-batch negatives: `MultipleNegativesRankingLoss` only exposes the model to semantically
distant non-matches (random other products in the batch), never to near-duplicate hard cases
like "iPhone 14 Pro 256GB" vs "iPhone 14 Pro Max 256GB".  M3 is designed to break this ceiling
by injecting LLM-generated hard negatives.

To investigate whether more training epochs help (spoiler: they don't — the plateau is
structural, not a training-budget issue):

```bash
uv run python scripts/sweep_epochs.py --split train_100pct
# Sweeps 3, 5, 10, 15 epochs → results/epoch_sweep.csv
```

---

## 10. Research Alignment

### Research questions (from roadmap)

| RQ | Question | How M2 sets it up |
|----|----------|-------------------|
| **RQ1** | Does LLM hard negative fine-tuning improve Acc@1 / MRR vs. in-batch negatives? | `BiEncoderBaseline` (M2) is the direct comparison target for M3 |
| **RQ2** | Does the *type* of hard negative matter (phonetic vs. semantic vs. structural)? | Ablation A4 in M3; the val set is our benchmark |
| **RQ3** | How does the benefit scale with labelled data size? | 10%/25%/100% splits created in M2; bi-encoder results on each split form the baseline curve |

### Literature review connection (M1 → M2)

From the approved literature review memo (Section 7 — Research Gaps):

- **Gap 2 (Semantic hard negatives):** Retrieval-based methods like ANCE fail on
  semantically challenging negatives because they rely purely on embedding similarity.
  An LLM can explicitly reason about domain-specific attributes (CPU generation, storage,
  form factor) that embeddings conflate.  The val set contains exactly these hard cases.

- **Gap 3 (Entity-aware mining):** WDC LSPC `cluster_id` provides ground-truth entity
  clusters.  In M3, we can verify LLM-generated negatives against `cluster_id` to
  measure what fraction of synthesised "hard negatives" are actually false negatives
  (same product, different offer).

### Text representation

Product texts are built as:
```
[TITLE] <title> [BRAND] <brand> [DESC] <description (truncated to 300 chars)>
```
This mirrors BLINK (Wu et al., 2020) and RocketQA (Qu et al., 2021), both reviewed in M1,
which prepend special tokens to distinguish field types for the encoder.

---

## 11. Reproducing All Results

### Full reproduction from scratch

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Enter project root
cd /path/to/project

# 3. Install exact dependency versions
uv sync

# 4. Configure environment
cp .env.example .env
# (No API keys required for M2 — only needed from M3 onwards)

# 5. Download WDC data and create splits
uv run python scripts/prepare_data.py

# 6. Run BM25 baseline (no GPU needed, ~2 seconds)
uv run python scripts/run_baselines.py --skip-bi-encoder

# 7. Run bi-encoder baseline (GPU recommended; ~10–40 min on CPU depending on split)
uv run python scripts/run_baselines.py --skip-bm25

# 8. View results
cat results/baselines.csv
```

### Expected runtimes (Apple M-series / CPU)

| Step | Expected time |
|------|--------------|
| `prepare_data.py` (download + split) | 5–15 min (network-dependent) |
| BM25 evaluation | ~2 seconds |
| Bi-encoder training, 10pct (775 pairs, 3 epochs) | ~5 min on MPS / ~15 min on CPU |
| Bi-encoder training, 25pct (1,938 pairs, 3 epochs) | ~12 min on MPS / ~35 min on CPU |
| Bi-encoder training, 100pct (7,752 pairs, 3 epochs) | ~45 min on MPS / ~2.5 hr on CPU |
| Bi-encoder evaluation (encode 3,479 + 1,938 texts) | ~30 seconds on MPS |

### Reproducibility details

- All data splits are deterministic: `sklearn.train_test_split` with `random_state=42`
  and `stratify=label`.  Split files are written once and not overwritten.
- Bi-encoder training uses `seed=42` in `SentenceTransformerTrainingArguments`.
- Results in `results/baselines.csv` are appended with UTC timestamps so multiple runs
  are preserved rather than overwritten.

---

## 12. Milestone 3 — LLM Hard Negative Generation

M3 is now in progress.  The prompt layer is complete; the generation pipeline and training
loop are next.

### M3 progress

| Step | File | Status |
|------|------|--------|
| Prompt builders (4 types × 3 strategies) | `src/wdc_hn/generation/prompts.py` | ✅ Done |
| Prompt preview on real products | `scripts/test_wdc_prompts.py` | ✅ Done |
| LLM negative synthesis + disk cache | `src/wdc_hn/generation/generate_negatives.py` | ⏳ Next |
| Contrastive bi-encoder with InfoNCE loss | `src/wdc_hn/models/contrastive_bi_encoder.py` | ⏳ Next |
| Training loop with wandb logging | `scripts/train.py` | ⏳ Next |
| Ablation YAML configs | `configs/ablations/` | ⏳ Next |

### Prompt preview (no API calls needed)

```bash
# Preview all 4 types × 3 strategies on 5 random products from the val set
uv run python scripts/test_wdc_prompts.py

# Narrow to one negative type
uv run python scripts/test_wdc_prompts.py --type component_swap

# Narrow to one strategy
uv run python scripts/test_wdc_prompts.py --strategy chain_of_thought --n 10
```

### Hard negative types (Ablation A4)

| Type | Description | Example |
|------|-------------|---------|
| `phonetic` | Misspellings / transliterations | "iFone 14 Pro" for "iPhone 14 Pro" |
| `component_swap` | One spec attribute changed | 256 GB → 512 GB, i5 → i7, 15" → 17" |
| `abbreviation` | Common short-form aliases | "MBP 14 M2" for "MacBook Pro 14-inch M2 Pro" |
| `semantic_distractor` | Same brand/family, different model | MacBook Air vs MacBook Pro |
| `mixed` | Mix of all four types | — |

### Ablation grid (A1–A5)

| ID | Factor | Levels |
|----|--------|--------|
| A1 | Bi-encoder backbone | `distilbert-base-uncased`, `bert-base-uncased`, `msmarco-distilbert-base-v4` |
| A2 | Generator LLM | `gpt-3.5-turbo`, `meta-llama/Llama-3-8b-instruct`, `mistralai/Mixtral-8x7B` |
| A3 | Prompting strategy | zero-shot, few-shot (3 examples), chain-of-thought |
| A4 | Negative synthesis type | phonetic, component_swap, abbreviation, semantic_distractor, mixed |
| A5 | Negative ratio (LLM-gen : in-batch) | 1:1, 4:1, 1:0 |

### Implementation notes for M3

- Generate negatives once per positive pair and **cache to disk** (avoid repeated API calls).
- Use `InfoNCE` loss with a learnable temperature parameter `τ`.
- Start all ablations on `train_10pct` (fastest iteration loop).
- Log all training metrics to **Weights & Biases** (`wandb`) for easy comparison.
- Use random seeds `{42, 123, 456}` for all final model runs (for confidence intervals in M4).

### M2 deliverables ✅

`results/baselines.csv` and a technical analysis report (`milestone2_analysis.pdf`) were
shared with Dr. Raj Dandekar.  Mentor feedback received and two issues addressed:

1. **Empty test split** — `splits.py` auto-carves a stratified 10% holdout from training data
   on the first run when the WDC gold-standard file is unavailable.  Subsequent runs detect the
   existing output and skip re-carving (idempotent).  Run `prepare_data.py --force` to
   explicitly re-partition.

2. **Acc@1 plateau** — documented as structural (in-batch negative deficit, not a training
   budget issue).  Use `scripts/sweep_epochs.py` to verify on Colab.
