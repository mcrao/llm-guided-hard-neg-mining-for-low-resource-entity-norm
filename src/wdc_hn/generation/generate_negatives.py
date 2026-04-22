"""
generate_negatives.py — LLM hard negative synthesis with disk caching.

Public API
----------
  load_cache(cache_dir, model_id, neg_type, strategy) -> dict[str, list[str]]
  generate_for_split(split_df, neg_type, strategy, model_id, ...) -> pd.DataFrame
  build_augmented_df(original_df, negatives_df, ratio) -> pd.DataFrame

Cache layout
------------
  data/negatives_cache/{model_id}__{neg_type}__{strategy}.jsonl
    One JSON object per line, one object per generated negative text.
  data/negatives_cache/failures.jsonl
    One JSON object per failed API call.

Output schema (negatives_df columns)
-------------------------------------
  pair_id, product_text, negative_text, negative_type, strategy, model_id, timestamp
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from wdc_hn.data.dataset import build_text
from wdc_hn.generation.prompts import build_prompt
from wdc_hn.utils import get_logger

log = get_logger(__name__)

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, model_id: str, neg_type: str, strategy: str) -> Path:
    """Return the JSONL path for a given (model, type, strategy) combination."""
    safe_model = model_id.replace("/", "_")
    return Path(cache_dir) / f"{safe_model}__{neg_type}__{strategy}.jsonl"


def _make_cache_key(product_text: str) -> str:
    """SHA-256 hash of normalised product text — the cache lookup key."""
    return hashlib.sha256(product_text.strip().lower().encode()).hexdigest()


def load_cache(
    cache_dir: Path,
    model_id: str,
    neg_type: str,
    strategy: str,
) -> dict[str, list[str]]:
    """
    Load existing cache into memory as {cache_key: [neg_text, ...]}.

    Returns an empty dict if the cache file does not exist.
    """
    path = _cache_path(cache_dir, model_id, neg_type, strategy)
    if not path.exists():
        return {}

    cache: dict[str, list[str]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = record["cache_key"]
                cache.setdefault(key, []).append(record["negative_text"])
            except (json.JSONDecodeError, KeyError):
                continue  # skip malformed lines

    log.info(f"Loaded {len(cache):,} cached product entries from {path.name}")
    return cache


def _append_to_cache(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL cache file (atomic line append)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _log_failure(cache_dir: Path, record: dict) -> None:
    """Append a failure record to failures.jsonl."""
    path = Path(cache_dir) / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── LLM client ─────────────────────────────────────────────────────────────────

def _make_client(model_id: str, api_key: Optional[str], base_url: Optional[str]):
    """
    Return an OpenAI client.

    For local models (Llama, Mixtral) served via vLLM / Ollama on RunPod,
    pass base_url="http://<runpod-ip>:<port>/v1" and api_key="EMPTY".
    """
    from openai import OpenAI

    kwargs: dict = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def call_llm_with_retry(
    client,
    prompt: str,
    model_id: str,
    max_retries: int = 5,
    temperature: float = 0.7,
) -> tuple[str, int, int]:
    """
    Call the LLM API with exponential backoff on rate limits and network errors.

    Returns:
        (response_text, prompt_tokens, completion_tokens)

    Raises:
        RuntimeError after max_retries consecutive failures.
    """
    import openai

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            usage = response.usage
            prompt_tok = usage.prompt_tokens if usage else 0
            completion_tok = usage.completion_tokens if usage else 0
            return text, prompt_tok, completion_tok

        except openai.RateLimitError:
            wait = min(2 ** attempt, 60)
            log.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait}s …")
            time.sleep(wait)

        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            wait = 5
            log.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {exc}. Waiting {wait}s …")
            time.sleep(wait)

        except Exception as exc:
            log.error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {exc}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM call failed after {max_retries} attempts: {exc}") from exc
            time.sleep(5)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts (rate limit exhausted).")


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_negatives(response_text: str) -> list[str]:
    """
    Extract the hard_negatives list from the LLM JSON response.

    The prompt contract is:
      {"hard_negatives": ["text1", "text2", ...]}
    CoT additionally has a "reasoning" key (ignored here).

    Raises:
        ValueError if the response cannot be parsed or the list is empty.
    """
    text = response_text.strip()

    # Strip markdown fences if the model added them despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    data = json.loads(text)  # raises json.JSONDecodeError on failure

    negatives = data.get("hard_negatives", [])
    if not isinstance(negatives, list) or len(negatives) == 0:
        raise ValueError(f"hard_negatives list is empty or missing: {data!r}")

    # Filter out empty strings
    negatives = [n.strip() for n in negatives if isinstance(n, str) and n.strip()]
    if not negatives:
        raise ValueError("All hard_negatives entries are empty strings.")

    return negatives


# ── Core generation ────────────────────────────────────────────────────────────

def generate_for_split(
    split_df: pd.DataFrame,
    neg_type: str,
    strategy: str,
    model_id: str,
    n_per_product: int = 5,
    cache_dir: Path = Path("data/negatives_cache"),
    dry_run: bool = False,
    force: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate hard negatives for all match pairs in split_df.

    Skips pairs whose product_text is already cached (unless force=True).
    In dry_run mode, prints prompts and returns an empty DataFrame.

    Args:
        split_df:       Training DataFrame (uses label==1 rows only).
        neg_type:       Negative type — see prompts._TYPE_INSTRUCTIONS.
        strategy:       Prompt strategy — zero_shot | few_shot | chain_of_thought.
        model_id:       OpenAI model ID or local vLLM model name.
        n_per_product:  Number of hard negatives to request per product.
        cache_dir:      Root cache directory.
        dry_run:        If True, print prompts without calling the API.
        force:          If True, re-generate even if cached.
        api_key:        OpenAI API key (reads OPENAI_API_KEY env var if None).
        base_url:       Base URL for OpenAI-compatible local API.

    Returns:
        DataFrame with columns:
          pair_id, product_text, negative_text, negative_type, strategy,
          model_id, timestamp
    """
    cache_dir = Path(cache_dir)
    cache_file = _cache_path(cache_dir, model_id, neg_type, strategy)

    # Load existing cache
    cache = {} if force else load_cache(cache_dir, model_id, neg_type, strategy)

    # Work only on match pairs
    match_df = split_df[split_df["label"] == 1].reset_index(drop=True)
    log.info(
        f"Generating [{neg_type}] × [{strategy}] × [{model_id}] "
        f"for {len(match_df):,} match pairs …"
    )

    if not dry_run:
        client = _make_client(model_id, api_key, base_url)

    records: list[dict] = []
    n_cached = 0
    n_generated = 0
    n_failed = 0
    total_prompt_tok = 0
    total_completion_tok = 0

    for idx, (_, row) in enumerate(match_df.iterrows()):
        pair_id = str(row.get("pair_id", idx))
        product_text = build_text(
            title=row.get("title_left"),
            brand=row.get("brand_left"),
            description=row.get("description_left"),
        )
        cache_key = _make_cache_key(product_text)

        # ── Cache hit ─────────────────────────────────────────────────────
        if cache_key in cache and not force:
            negatives = cache[cache_key]
            n_cached += 1
            for neg_text in negatives:
                records.append({
                    "pair_id":       pair_id,
                    "product_text":  product_text,
                    "negative_text": neg_text,
                    "negative_type": neg_type,
                    "strategy":      strategy,
                    "model_id":      model_id,
                    "timestamp":     "",
                })
            continue

        # ── Build prompt ──────────────────────────────────────────────────
        prompt = build_prompt(
            product_text=product_text,
            negative_type=neg_type,
            strategy=strategy,
            n_negatives=n_per_product,
        )

        if dry_run:
            print(f"\n{'='*60}\nPair {pair_id} | product: {product_text[:80]}...\n")
            print(prompt[:500] + "\n[... prompt truncated ...]")
            continue

        # ── API call with retry ───────────────────────────────────────────
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            response_text, prompt_tok, completion_tok = call_llm_with_retry(
                client, prompt, model_id
            )
            negatives = parse_negatives(response_text)

        except (json.JSONDecodeError, ValueError):
            # Retry once with temperature=0
            log.warning(f"Parse error for pair {pair_id}. Retrying with temperature=0 …")
            try:
                response_text, prompt_tok, completion_tok = call_llm_with_retry(
                    client, prompt, model_id, max_retries=1, temperature=0.0
                )
                negatives = parse_negatives(response_text)
            except Exception as exc2:
                log.error(f"Pair {pair_id} failed after retry: {exc2}")
                _log_failure(cache_dir, {
                    "pair_id": pair_id, "product_text": product_text,
                    "negative_type": neg_type, "strategy": strategy,
                    "model_id": model_id, "error": str(exc2), "timestamp": ts,
                })
                n_failed += 1
                continue

        except RuntimeError as exc:
            log.error(f"Pair {pair_id} failed permanently: {exc}")
            _log_failure(cache_dir, {
                "pair_id": pair_id, "product_text": product_text,
                "negative_type": neg_type, "strategy": strategy,
                "model_id": model_id, "error": str(exc), "timestamp": ts,
            })
            n_failed += 1
            continue

        total_prompt_tok += prompt_tok
        total_completion_tok += completion_tok
        n_generated += 1

        # ── Write to cache and collect records ────────────────────────────
        cache[cache_key] = negatives
        base_record = {
            "cache_key":             cache_key,
            "pair_id":               pair_id,
            "product_text":          product_text,
            "negative_type":         neg_type,
            "strategy":              strategy,
            "model_id":              model_id,
            "n_negatives_requested": n_per_product,
            "prompt_tokens":         prompt_tok,
            "completion_tokens":     completion_tok,
            "timestamp":             ts,
        }
        for neg_text in negatives:
            _append_to_cache(cache_file, {**base_record, "negative_text": neg_text})
            records.append({
                "pair_id":       pair_id,
                "product_text":  product_text,
                "negative_text": neg_text,
                "negative_type": neg_type,
                "strategy":      strategy,
                "model_id":      model_id,
                "timestamp":     ts,
            })

        # ── Progress log every 50 items ───────────────────────────────────
        if (idx + 1) % 50 == 0:
            _log_progress(idx + 1, len(match_df), n_cached, n_generated, n_failed,
                          total_prompt_tok, total_completion_tok, model_id)

    if not dry_run:
        _log_progress(len(match_df), len(match_df), n_cached, n_generated, n_failed,
                      total_prompt_tok, total_completion_tok, model_id)

    return pd.DataFrame(records)


def _log_progress(
    done: int,
    total: int,
    n_cached: int,
    n_generated: int,
    n_failed: int,
    prompt_tok: int,
    completion_tok: int,
    model_id: str,
) -> None:
    cost = _estimate_cost(model_id, prompt_tok, completion_tok)
    log.info(
        f"Progress {done}/{total} | "
        f"cached={n_cached} generated={n_generated} failed={n_failed} | "
        f"tokens={prompt_tok + completion_tok:,} | "
        f"est. cost=${cost:.3f}"
    )


# ── Cost estimation ────────────────────────────────────────────────────────────

# Approximate prices in USD per 1M tokens (input / output)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-3.5-turbo":         (0.50,  1.50),
    "gpt-3.5-turbo-0125":    (0.50,  1.50),
    "gpt-4o-mini":           (0.15,  0.60),
    "gpt-4o":                (5.00, 15.00),
}


def _estimate_cost(model_id: str, prompt_tok: int, completion_tok: int) -> float:
    """Return estimated USD cost for the given token counts."""
    # Match on prefix (handles versioned model IDs)
    price_in, price_out = 0.0, 0.0
    for key, (p_in, p_out) in _PRICING.items():
        if model_id.startswith(key):
            price_in, price_out = p_in, p_out
            break
    return (prompt_tok * price_in + completion_tok * price_out) / 1_000_000


# ── Training augmentation ──────────────────────────────────────────────────────

def build_augmented_df(
    original_df: pd.DataFrame,
    negatives_df: pd.DataFrame,
    ratio: int = 1,
) -> pd.DataFrame:
    """
    Sample hard negatives for use in triplet training.

    For each match pair in original_df, sample up to `ratio` hard negatives
    from negatives_df (joined on pair_id). Returns a DataFrame with columns:
      pair_id, anchor_text, positive_text, negative_text

    This is the format consumed by the modified BiEncoderBaseline.train()
    when hard_neg_df is provided.

    Args:
        original_df:   Original training DataFrame (unified WDC schema).
        negatives_df:  Output of generate_for_split().
        ratio:         Max number of hard negatives to sample per match pair.
                       1 → 1:1 ratio, 4 → 4:1 ratio.

    Returns:
        DataFrame with columns: pair_id, anchor_text, positive_text, negative_text
    """
    match_df = original_df[original_df["label"] == 1].reset_index(drop=True)

    rows = []
    for _, row in match_df.iterrows():
        pair_id = str(row.get("pair_id", ""))
        neg_rows = negatives_df[negatives_df["pair_id"] == pair_id]
        if neg_rows.empty:
            continue

        anchor_text   = build_text(
            title=row.get("title_left"),
            brand=row.get("brand_left"),
            description=row.get("description_left"),
        )
        positive_text = build_text(
            title=row.get("title_right"),
            brand=row.get("brand_right"),
            description=row.get("description_right"),
        )
        # Sample up to `ratio` negatives per pair
        sampled = neg_rows.sample(n=min(ratio, len(neg_rows)), random_state=42)
        for _, neg_row in sampled.iterrows():
            rows.append({
                "pair_id":        pair_id,
                "anchor_text":    anchor_text,
                "positive_text":  positive_text,
                "negative_text":  neg_row["negative_text"],
            })

    return pd.DataFrame(rows)
