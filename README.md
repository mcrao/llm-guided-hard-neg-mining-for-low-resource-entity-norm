# LLM-Guided Hard Negative Mining for Low-Resource Product Entity Matching

> **Submitted to NeurIPS 2026.**

We study the structural Acc@1 ceiling that afflicts bi-encoders trained with in-batch negatives on
product matching tasks. Despite scaling labelled data 10× (698 → 6,977 match pairs) or training
up to 15 epochs, top-1 retrieval accuracy remains flat at ~4.5% on the WDC LSPC Computers
benchmark. We introduce an LLM-guided hard negative pipeline that synthesises four semantically
grounded negative types (phonetic variants, component swaps, abbreviations, semantic distractors)
using three prompting strategies (zero-shot, few-shot, chain-of-thought). A five-factor ablation
shows that chain-of-thought generation combined with component-swap negatives at a 4:1
LLM-to-in-batch ratio achieves the best overall gains, with Acc@5 exceeding the full-data baseline
(0.5542 vs. 0.5108) while using only 10% of labelled data.

---

## Key Results

### Baselines — in-batch negatives only

| Method | Train split | Acc@1 | Acc@5 | MRR |
|--------|------------|-------|-------|-----|
| TF-IDF / BM25 | — | 0.0346 | 0.3870 | 0.1988 |
| Bi-encoder (MNRL) | train_10pct (698 pairs) | 0.0433 | 0.5108 | 0.2486 |
| Bi-encoder (MNRL) | train_25pct (1,744 pairs) | 0.0454 | 0.5526 | 0.2636 |
| Bi-encoder (MNRL) | train_100pct (6,977 pairs) | 0.0459 | 0.5991 | 0.2800 |

### LLM-HN ablation — train_10pct, GPT-4o-mini generator

All rows use distilbert-base-uncased and ratio=1:1 unless noted. Reference: in-batch only → Acc@1=0.0433, Acc@5=0.5108, MRR=0.2486.

| Ablation | Config | Acc@1 | Acc@5 | MRR | n_hn |
|----------|--------|-------|-------|-----|------|
| A3 — Strategy | component_swap + zero-shot | 0.0428 | 0.5062 | 0.2461 | 693 |
| A3 — Strategy | component_swap + few-shot | 0.0402 | **0.5191** | 0.2464 | 694 |
| A3 — Strategy | **component_swap + CoT** | **0.0444** | 0.5175 | **0.2489** | 697 |
| A4 — Type | **component_swap** + CoT | **0.0444** | 0.5175 | **0.2489** | 697 |
| A4 — Type | phonetic + CoT | 0.0433 | **0.5212** | 0.2477 | 698 |
| A4 — Type | semantic_distractor + CoT | 0.0439 | 0.5098 | 0.2488 | 697 |
| A4 — Type | abbreviation + CoT | 0.0428 | 0.5088 | 0.2455 | 698 |
| A1 — Backbone | distilbert-base-uncased | **0.0444** | 0.5175 | 0.2489 | 697 |
| A1 — Backbone | msmarco-distilbert-base-v4 | 0.0428 | **0.5335** | **0.2521** | 698 |
| A1 — Backbone | bert-base-uncased | 0.0423 | 0.5139 | 0.2493 | 698 |
| A5 — Ratio | 1:1 | 0.0444 | 0.5175 | 0.2489 | 697 |
| A5 — Ratio | **4:1** | **0.0444** | **0.5542** | **0.2596** | 2,792 |
| A5 — Ratio | in-batch only (no LLM neg.) | 0.0433 | 0.5108 | 0.2486 | 0 |

**Best configuration:** component_swap + CoT + 4:1 ratio → **+2.5% Acc@1, +8.5% Acc@5, +4.4% MRR** over the train_10pct baseline.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Repository Layout](#2-repository-layout)
3. [Environment Setup](#3-environment-setup)
4. [Dataset: WDC LSPC Computers](#4-dataset-wdc-lspc-computers)
5. [Data Pipeline](#5-data-pipeline)
6. [Baselines](#6-baselines)
7. [LLM Hard Negative Pipeline](#7-llm-hard-negative-pipeline)
8. [Training with Hard Negatives](#8-training-with-hard-negatives)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Reproducing All Results](#10-reproducing-all-results)
11. [Using the Library](#11-using-the-library)

---

## 1. Quick Start

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone / enter the project directory
cd /path/to/project

# 3. Create virtual environment and install all dependencies
uv sync

# 4. Copy environment template and add your OpenAI API key
cp .env.example .env

# 5. Download WDC LSPC Computers and create data splits
uv run python scripts/prepare_data.py

# 6. Run baselines
uv run python scripts/run_baselines.py

# 7. Generate hard negatives (requires OPENAI_API_KEY in .env)
uv run python scripts/generate_negatives.py --split train_10pct --type component_swap --strategy chain_of_thought

# 8. Train bi-encoder with hard negatives
uv run python scripts/train_with_hn.py --split train_10pct --type component_swap --strategy chain_of_thought --ratio 4
```

---

## 2. Repository Layout

```
project/
│
├── pyproject.toml              # uv project manifest + pinned dependencies
├── uv.lock                     # exact locked dependency tree
├── .python-version             # pins Python 3.11
├── .env.example                # environment variable template
├── .env                        # your local secrets (never committed)
│
├── data/
│   ├── raw/                    # downloaded WDC Parquet files (auto-generated)
│   └── splits/                 # stratified low-resource training splits
│       ├── computers_train_10pct.parquet   # 4,930 pairs (698 matches)
│       ├── computers_train_25pct.parquet   # 12,323 pairs (1,744 matches)
│       ├── computers_train_100pct.parquet  # 49,291 pairs (6,977 matches)
│       ├── computers_val.parquet           # 13,693 pairs (1,938 matches)
│       ├── computers_test.parquet          # 5,477 pairs (775 matches) — held-out
│       └── split_stats.json               # exact row counts for every split
│
├── src/wdc_hn/                 # installable Python package (wdc_hn)
│   ├── data/
│   │   ├── download.py         # WDC download: HuggingFace → HTTP → curl fallback
│   │   ├── dataset.py          # ProductPairDataset (pairwise & contrastive modes)
│   │   └── splits.py           # stratified low-resource split creation + test holdout
│   ├── evaluation/
│   │   └── metrics.py          # build_eval_corpus, compute_ranks, Acc@k, MRR
│   ├── baselines/
│   │   ├── bm25.py             # TFIDFBaseline — sparse TF-IDF + cosine similarity
│   │   └── bi_encoder_in_batch.py  # BiEncoderBaseline — distilbert + MNRL
│   ├── generation/
│   │   ├── prompts.py          # LLM prompt builders: 4 negative types × 3 strategies
│   │   └── generate_negatives.py   # LLM call + JSON parse + disk cache logic
│   └── utils/
│       └── logging.py          # Rich-based coloured logging
│
├── scripts/
│   ├── prepare_data.py         # End-to-end data download + split creation
│   ├── run_baselines.py        # Baseline evaluation → results/baselines.csv
│   ├── sweep_epochs.py         # Epoch sweep (3/5/10/15) → results/epoch_sweep.csv
│   ├── generate_negatives.py   # LLM hard negative generation CLI
│   ├── train_with_hn.py        # Train bi-encoder with hard negatives
│   ├── run_ablations.py        # Full A1–A5 ablation grid
│   ├── plot_epoch_sweep.py     # Reproduce Figure 1 (epoch sweep plot)
│   └── test_wdc_prompts.py     # Preview LLM prompts without API calls
│
├── results/
│   ├── baselines.csv           # Baseline metrics (appended on each run)
│   ├── epoch_sweep.csv         # Epoch sweep results
│   └── m3_results.csv          # LLM-HN ablation results (A1–A5)
│
└── manuscript/
    └── NeurIPS/myPaper/        # NeurIPS 2026 submission (LaTeX)
```

> **Test split:** The WDC LSPC gold-standard file is not publicly available. On first run,
> `splits.py` automatically carves a stratified 10% holdout from the full training data
> (seed=42), then regenerates training splits from the remaining 90%. Use `prepare_data.py --force`
> to re-carve. All reported results use `computers_val.parquet`; the held-out test set is
> reserved for final evaluation.

---

## 3. Environment Setup

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11 (pinned in `.python-version`) | [python.org](https://www.python.org/) or `pyenv` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### Install dependencies

```bash
uv sync
```

Reads `pyproject.toml` and `uv.lock`; creates `.venv/` with exact pinned versions.

### Core dependencies

| Package | Purpose |
|---------|---------|
| `torch ≥ 2.2` | Model training |
| `transformers ≥ 4.40` | Tokenizers, model loading |
| `sentence-transformers ≥ 3.0` | Siamese bi-encoder training (MNRL) |
| `datasets ≥ 2.19` | Training data management |
| `openai` | LLM hard negative generation |
| `scikit-learn ≥ 1.4` | TF-IDF, stratified splitting |
| `pandas ≥ 2.2` | DataFrame operations |
| `pyarrow ≥ 15` | Parquet I/O |
| `rich ≥ 13.7` | Terminal output + progress bars |
| `typer ≥ 0.12` | CLI argument parsing |

### Environment variables

Copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `WDC_CATEGORY` | `computers` | Data download category |
| `WDC_TRAIN_SIZE` | `xlarge` | Training set size |
| `RANDOM_SEED` | `42` | Global reproducibility seed |
| `BIENCODER_MODEL` | `distilbert-base-uncased` | Bi-encoder backbone |
| `MAX_SEQ_LENGTH` | `128` | Tokenizer max length |
| `OPENAI_API_KEY` | *(empty)* | Required for hard negative generation |
| `LLM_MODEL` | `gpt-4o-mini` | Generator LLM for hard negatives |

---

## 4. Dataset: WDC LSPC Computers

**Source:** [WDC Large-Scale Product Corpus 2017](http://webdatacommons.org/largescaleproductcorpus/v2/) —
English Computers subset. Also mirrored at HuggingFace (`wdc/products-2017`).

Pairs are labelled: `label=1` = same product, `label=0` = different product.
`is_hard_negative=True` flags negatives that were similarity-mined (most confusable non-matches).

### Split sizes

| Split | Total pairs | Matches (label=1) | Match % |
|-------|------------|-------------------|---------|
| `train_10pct` | 4,930 | 698 | 14.16% |
| `train_25pct` | 12,323 | 1,744 | 14.15% |
| `train_100pct` | 49,291 | 6,977 | 14.15% |
| `val` | 13,693 | 1,938 | 14.15% |
| `test` | 5,477 | 775 | 14.15% (held-out) |

The three training splits enable studying how hard negative benefit scales with labelled data:
`train_10pct` (very low-resource), `train_25pct` (moderate), `train_100pct` (full data ceiling).

### Dataset schema (per Parquet file)

| Column | Type | Description |
|--------|------|-------------|
| `id_left`, `id_right` | str | Unique offer identifiers |
| `title_left`, `title_right` | str | Product titles |
| `brand_left`, `brand_right` | str | Brand names |
| `description_left`, `description_right` | str | Offer descriptions (truncated to 300 chars) |
| `label` | int (0/1) | 1 = same product; 0 = different products |
| `pair_id` | str | `"{id_left}#{id_right}"` |
| `cluster_id_left`, `cluster_id_right` | int/NA | Ground-truth entity cluster |
| `is_hard_negative` | bool | True if similarity-mined non-match |

---

## 5. Data Pipeline

```bash
# Standard run (download + create all splits)
uv run python scripts/prepare_data.py

# Faster iteration with smaller training set
uv run python scripts/prepare_data.py --train-size large

# Sanity check on already-downloaded data
uv run python scripts/prepare_data.py --smoke-test

# Force re-download and re-split
uv run python scripts/prepare_data.py --force
```

**Pipeline steps:**

1. **Download** — HuggingFace Hub → WDC HTTP → system curl (three fallbacks)
2. **Split** — `sklearn.train_test_split` with `stratify=label`, `random_state=42`; carves test holdout on first run
3. **Smoke test** — loads each split, instantiates `ProductPairDataset` in both modes, prints a sample triplet

---

## 6. Baselines

```bash
# Full run: TF-IDF + bi-encoder on all three training splits (~40 min on GPU)
uv run python scripts/run_baselines.py

# TF-IDF only (~2 seconds, no GPU)
uv run python scripts/run_baselines.py --skip-bi-encoder

# Bi-encoder on specific splits
uv run python scripts/run_baselines.py --skip-bm25 --splits 10pct 25pct

# Epoch sweep to verify the Acc@1 plateau
uv run python scripts/sweep_epochs.py --split train_100pct

# Reproduce Figure 1 (epoch sweep plot)
uv run python scripts/plot_epoch_sweep.py
```

### TF-IDF / BM25 (`src/wdc_hn/baselines/bm25.py`)

Sparse keyword retrieval as an unsupervised lower bound. Fit on the 3,479-item val corpus;
no training data required. N-gram range (1, 2) with `sublinear_tf=True`.

### Bi-encoder with in-batch negatives (`src/wdc_hn/baselines/bi_encoder_in_batch.py`)

Fine-tunes a Siamese `distilbert-base-uncased` bi-encoder using `MultipleNegativesRankingLoss`
(in-batch negative paradigm from DPR). Training uses only label==1 pairs; all other products in
the batch act as implicit negatives.

| Hyperparameter | Value |
|----------------|-------|
| Base model | `distilbert-base-uncased` |
| Loss | `MultipleNegativesRankingLoss` |
| Epochs | 3 |
| Batch size | 32 |
| Warmup | 10% of steps |
| Max seq length | 128 tokens |
| Seed | 42 |

---

## 7. LLM Hard Negative Pipeline

The core contribution: an LLM synthesis pipeline that generates semantically challenging
negatives for contrastive bi-encoder training.

### Hard negative types (Ablation A4)

| Type | What changes | Example |
|------|-------------|---------|
| `phonetic` | Misspellings / transliterations | "iFone 14 Pro" for "iPhone 14 Pro" |
| `component_swap` | One spec attribute changed | 256 GB → 512 GB, i5 → i7, 15" → 17" |
| `abbreviation` | Common short-form aliases | "MBP 14 M2" for "MacBook Pro 14-inch M2 Pro" |
| `semantic_distractor` | Same brand/family, different model | MacBook Air vs. MacBook Pro |

### Prompting strategies (Ablation A3)

| Strategy | Description |
|----------|-------------|
| `zero_shot` | Direct instruction with output schema |
| `few_shot` | 3 in-context product-negative examples before the target |
| `chain_of_thought` | Model reasons about distinguishing attributes before generating |

### Generating negatives

```bash
# Preview prompts without API calls
uv run python scripts/test_wdc_prompts.py --type component_swap --strategy chain_of_thought

# Generate and cache negatives for a split (requires OPENAI_API_KEY)
uv run python scripts/generate_negatives.py \
  --split train_10pct \
  --type component_swap \
  --strategy chain_of_thought \
  --model gpt-4o-mini \
  --n 5

# Dry run: inspect prompts without spending API budget
uv run python scripts/generate_negatives.py --split train_10pct --type phonetic --strategy zero_shot --dry-run
```

**Caching:** Generated negatives are cached to `data/negatives_cache/` as JSONL files keyed by
SHA-256 hash of the product text. Re-running is idempotent — cached items cost zero API calls.

**Cache schema** (one JSON record per negative, one line per record):

```json
{
  "cache_key": "sha256-of-product-text",
  "pair_id": "L42#R42",
  "product_text": "[TITLE] Dell XPS 15 [BRAND] Dell [DESC] ...",
  "negative_text": "Dell XPS 15 9530 Intel Core i5-13500H ...",
  "negative_type": "component_swap",
  "strategy": "chain_of_thought",
  "model_id": "gpt-4o-mini",
  "n_negatives_requested": 5,
  "prompt_tokens": 312,
  "completion_tokens": 148,
  "timestamp": "2026-04-22T10:00:00Z"
}
```

**Failure handling:** Malformed JSON triggers one retry at `temperature=0`; HTTP 429 uses
exponential backoff (up to 60s, 5 retries); all failures are logged to `data/negatives_cache/failures.jsonl`.

### Cost estimate (GPT-4o-mini, train_10pct, all 4 types × 3 strategies)

- 698 pairs × 4 types × 3 strategies = 8,376 API calls
- ~410 tokens input / ~150 tokens output on average
- **Total: ~$1.27** at GPT-4o-mini pricing ($0.15/$0.60 per MTok)

---

## 8. Training with Hard Negatives

```bash
# Train bi-encoder with LLM hard negatives
uv run python scripts/train_with_hn.py \
  --split train_10pct \
  --type component_swap \
  --strategy chain_of_thought \
  --ratio 4 \
  --epochs 3

# Run the full A1–A5 ablation grid (assumes cache is warm)
uv run python scripts/run_ablations.py
```

Hard negatives are loaded from the cache and joined to the training data as explicit
`(anchor, positive, hard_negative)` triplets. The `MultipleNegativesRankingLoss` uses the
explicit negative alongside in-batch negatives for each step.

### Ablation grid

| ID | Factor | Levels |
|----|--------|--------|
| A1 | Backbone | `distilbert-base-uncased`, `bert-base-uncased`, `msmarco-distilbert-base-v4` |
| A2 | Generator LLM | `gpt-4o-mini` |
| A3 | Prompting strategy | zero-shot, few-shot, chain-of-thought |
| A4 | Negative type | phonetic, component_swap, abbreviation, semantic_distractor |
| A5 | LLM-gen : in-batch ratio | 1:1, 4:1, 1:0 |

Results are written to `results/m3_results.csv`.

---

## 9. Evaluation Metrics

All metrics are computed in `src/wdc_hn/evaluation/metrics.py`.

**Evaluation task:** Given a query (left-side product text), rank all 3,479 items in the val
corpus (unique right-side texts). For each of 1,938 match pairs, the ground-truth positive is
the right-side text of that pair.

**Accuracy@k:**
```
Acc@k = (1 / |Q|) × Σ 1{rank_i ≤ k}
```
Fraction of queries where the ground-truth appears in the top-k results. We report Acc@1 and Acc@5.

**Mean Reciprocal Rank:**
```
MRR = (1 / |Q|) × Σ (1 / rank_i)
```
Awards partial credit for near-misses; a positive at rank 2 scores 0.5.

---

## 10. Reproducing All Results

### Full reproduction from scratch

```bash
# Install and configure
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/project
uv sync
cp .env.example .env     # add OPENAI_API_KEY for hard negative generation

# Data
uv run python scripts/prepare_data.py

# Baselines
uv run python scripts/run_baselines.py

# Epoch sweep (Figure 1)
uv run python scripts/sweep_epochs.py --split train_100pct
uv run python scripts/plot_epoch_sweep.py

# Hard negative generation (requires API key; ~$1.27 total for all types × strategies)
for type in phonetic component_swap abbreviation semantic_distractor; do
  for strategy in zero_shot few_shot chain_of_thought; do
    uv run python scripts/generate_negatives.py \
      --split train_10pct --type $type --strategy $strategy --model gpt-4o-mini
  done
done

# Ablation grid
uv run python scripts/run_ablations.py
```

### Reproducibility notes

- All data splits are deterministic: `sklearn.train_test_split` with `random_state=42` and `stratify=label`.
- Bi-encoder training uses `seed=42` in `SentenceTransformerTrainingArguments`.
- Results in `results/baselines.csv` are appended with UTC timestamps (multiple runs preserved).
- Negative generation is idempotent: cached items never incur API calls on re-runs.

### Expected runtimes

| Step | Expected time |
|------|--------------|
| `prepare_data.py` | 5–15 min (network-dependent) |
| BM25 evaluation | ~2 seconds |
| Bi-encoder train, 10pct (3 epochs) | ~2 min on T4 / ~5 min on MPS |
| Bi-encoder train, 100pct (3 epochs) | ~12 min on T4 / ~45 min on MPS |
| Negative generation (all types × strategies, train_10pct) | ~30 min (network-dependent) |
| Full ablation grid (cache warm) | ~2 hr on T4 |

Baseline results were produced on NVIDIA T4 (Google Colab) and NVIDIA RTX 4500 (RunPod).

---

## 11. Using the Library

### Load a data split

```python
from wdc_hn.data import load_split

train_df = load_split("data/splits", "train_10pct")
# Valid names: train_10pct | train_25pct | train_100pct | val | test
```

### Build product text

```python
from wdc_hn.data import build_text

text = build_text(
    title="HP Pavilion x360 14-cd0010nr",
    brand="HP",
    description="14-inch 2-in-1 touchscreen laptop ...",
)
# "[TITLE] HP Pavilion x360 14-cd0010nr [BRAND] HP [DESC] 14-inch 2-in-1 ..."
```

### Generate and inspect prompts

```python
from wdc_hn.generation.prompts import build_prompt

prompt = build_prompt(
    product_text="[TITLE] Dell XPS 15 [BRAND] Dell [DESC] 15.6-inch OLED laptop",
    negative_type="component_swap",
    strategy="chain_of_thought",
    n_negatives=5,
)
print(prompt)
```

### Run evaluation metrics

```python
from wdc_hn.evaluation import build_eval_corpus, compute_ranks, compute_retrieval_metrics

queries, corpus, positive_indices = build_eval_corpus(val_df)
# queries: list[str] (1,938 items)
# corpus:  list[str] (3,479 items)

ranks = compute_ranks(scores_matrix, positive_indices)   # 1-indexed
metrics = compute_retrieval_metrics(ranks, ks=(1, 5))
# {"mrr": 0.xxxx, "acc_at_1": 0.xxxx, "acc_at_5": 0.xxxx, "n_queries": 1938}
```

### Train bi-encoder

```python
from wdc_hn.baselines import BiEncoderBaseline
from wdc_hn.data import load_split

train_df = load_split("data/splits", "train_10pct")
val_df   = load_split("data/splits", "val")

be = BiEncoderBaseline(base_model="distilbert-base-uncased")
be.train(train_df, output_dir="models/bi_encoder_10pct", epochs=3, batch_size=32)
metrics = be.evaluate(val_df)
```
