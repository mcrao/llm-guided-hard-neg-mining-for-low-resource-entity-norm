"""
Download the WDC LSPC Computers dataset.

Download strategies (tried in order):
  1. HuggingFace Hub (direct Parquet/file download via huggingface_hub,
     bypassing the legacy loading script which is no longer supported)
  2. WDC direct HTTPS download — requests.Session with browser headers
  3. WDC via system curl  — subprocess fallback when requests returns 403

Output schema (unified across all sources):
  id_left, title_left, description_left, brand_left, price_left,
  id_right, title_right, description_right, brand_right, price_right,
  label,   pair_id,   cluster_id_left,   cluster_id_right,
  is_hard_negative   (bool — True if negative was similarity-mined)

Saved as Parquet files under data/raw/:
  computers_train_{size}.parquet
  computers_val.parquet
  computers_test.parquet
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from wdc_hn.utils import get_logger

log = get_logger(__name__)

# ── WDC direct download URLs (HTTPS, no redirect needed) ──────────────────────
_WDC_BASE = (
    "https://data.dws.informatik.uni-mannheim.de/"
    "largescaleproductcorpus/data/v2/"
)
_WDC_REFERER = "https://webdatacommons.org/largescaleproductcorpus/v2/"

_WDC_URLS: dict[str, str] = {
    "train_small":  _WDC_BASE + "computers/computers_train_small.json.gz",
    "train_medium": _WDC_BASE + "computers/computers_train_medium.json.gz",
    "train_large":  _WDC_BASE + "computers/computers_train_large.json.gz",
    "train_xlarge": _WDC_BASE + "computers/computers_train_xlarge.json.gz",
    "val":          _WDC_BASE + "computers/computers_valid_66pair.json.gz",
    "test":         _WDC_BASE + "computers/computers_gs.json.gz",
}

# Fallback filenames to try if the primary val name fails
_WDC_VAL_CANDIDATES = [
    "computers_valid_66pair.json.gz",
    "computers_valid.json.gz",
    "computers_validation.json.gz",
]

# HuggingFace dataset ID and config map
_HF_DATASET = "wdc/products-2017"
_HF_CONFIG_MAP = {
    "small":  "computers_small",
    "medium": "computers_medium",
    "large":  "computers_large",
    "xlarge": "computers_xlarge",
}

# ── Unified schema ──────────────────────────────────────────────────────────────
_UNIFIED_COLS = [
    "id_left", "title_left", "description_left", "brand_left",
    "price_left", "priceCurrency_left",
    "id_right", "title_right", "description_right", "brand_right",
    "price_right", "priceCurrency_right",
    "label", "pair_id", "cluster_id_left", "cluster_id_right",
    "is_hard_negative",
]


# ── Column normalisation ───────────────────────────────────────────────────────

def _normalise_hf(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise HuggingFace column names to our unified schema."""
    if "pair_id" not in df.columns:
        if "id_left" in df.columns and "id_right" in df.columns:
            df["pair_id"] = (
                df["id_left"].astype(str) + "#" + df["id_right"].astype(str)
            )
        else:
            df["pair_id"] = df.index.astype(str)

    if "is_hard_negative" not in df.columns:
        df["is_hard_negative"] = df["label"].apply(lambda x: x == 0)

    for side in ("left", "right"):
        if f"cluster_id_{side}" not in df.columns:
            df[f"cluster_id_{side}"] = pd.NA
        for field in ("price", "priceCurrency"):
            col = f"{field}_{side}"
            if col not in df.columns:
                df[col] = pd.NA

    for col in _UNIFIED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[_UNIFIED_COLS].copy()


def _normalise_wdc(records: list[dict]) -> pd.DataFrame:
    """Normalise raw WDC JSON records (from .json.gz) to unified schema."""
    rows = []
    for r in records:
        row = {
            "id_left":            r.get("id_left",  r.get("nodeId_left", "")),
            "title_left":         r.get("title_left", ""),
            "description_left":   r.get("description_left", ""),
            "brand_left":         r.get("brand_left", ""),
            "price_left":         r.get("price_left", pd.NA),
            "priceCurrency_left": r.get("priceCurrency_left", pd.NA),
            "id_right":           r.get("id_right", r.get("nodeId_right", "")),
            "title_right":        r.get("title_right", ""),
            "description_right":  r.get("description_right", ""),
            "brand_right":        r.get("brand_right", ""),
            "price_right":        r.get("price_right", pd.NA),
            "priceCurrency_right": r.get("priceCurrency_right", pd.NA),
            "label":              int(r.get("label", 0)),
            "pair_id":            r.get("pair_id", ""),
            "cluster_id_left":    r.get("cluster_id_left", pd.NA),
            "cluster_id_right":   r.get("cluster_id_right", pd.NA),
            "is_hard_negative":   bool(r.get("is_hard_negative", False)),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=_UNIFIED_COLS)


# ── Browser-like headers ────────────────────────────────────────────────────────
# The WDC university server (data.dws.informatik.uni-mannheim.de) returns 403
# for plain requests without browser-like headers. We set headers at the Session
# level so they are preserved through HTTP→HTTPS redirects.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": _WDC_REFERER,
    "Connection": "keep-alive",
}


# ── Download helpers ────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Return a requests.Session with browser-like headers pre-configured."""
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)
    return session


def _download_with_requests(url: str, dest: Path, chunk_size: int = 65536) -> Path:
    """Stream-download via requests.Session (headers survive redirects)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[requests] [cyan]{url}[/cyan] → {dest.name}")
    session = _make_session()
    with session.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def _download_with_curl(url: str, dest: Path) -> Path:
    """Fallback: download via system curl with browser headers."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[curl] [cyan]{url}[/cyan] → {dest.name}")
    cmd = [
        "curl",
        "--location",           # follow redirects
        "--fail",               # exit non-zero on 4xx/5xx
        "--silent",
        "--show-error",
        "--progress-bar",
        "-H", f"User-Agent: {_BROWSER_HEADERS['User-Agent']}",
        "-H", f"Referer: {_WDC_REFERER}",
        "-H", "Accept: */*",
        "-H", "Accept-Language: en-US,en;q=0.5",
        "--output", str(dest),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"curl exited {result.returncode}: {result.stderr.strip()}"
        )
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("curl finished but destination file is missing or empty")
    return dest


def _download_file(url: str, dest: Path) -> Path:
    """
    Download url → dest, trying requests first then curl.

    Raises RuntimeError only when both strategies fail.
    """
    # Try requests (uses Session so headers persist through redirects)
    try:
        return _download_with_requests(url, dest)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            log.warning("  requests returned 403 — trying curl …")
        else:
            raise
    except Exception as exc:
        log.warning(f"  requests failed ({exc}) — trying curl …")

    # Cleanup partial file before curl
    if dest.exists():
        dest.unlink()

    return _download_with_curl(url, dest)


def _load_json_gz(path: Path) -> list[dict]:
    """Load a gzip-compressed JSON-lines file."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Strategy 1: HuggingFace Hub (direct file download) ────────────────────────

def _try_huggingface(
    train_size: str,
    cache_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load via HuggingFace Hub by downloading data files directly.

    Uses huggingface_hub (NOT datasets.load_dataset) so we bypass the
    legacy loading script — trust_remote_code is no longer supported in
    recent versions of the datasets library.

    Looks for Parquet files in the repo; falls back to JSON.gz if absent.
    Raises RuntimeError if no data files are found for the requested config.
    """
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore

    config = _HF_CONFIG_MAP.get(train_size, f"computers_{train_size}")
    log.info(
        f"Checking HuggingFace Hub [{_HF_DATASET}] "
        f"for config=[cyan]{config}[/cyan] …"
    )

    api = HfApi()
    try:
        all_paths = [
            entry.path
            for entry in api.list_repo_tree(
                _HF_DATASET, repo_type="dataset", recursive=True
            )
        ]
    except Exception as exc:
        raise RuntimeError(f"Cannot list HuggingFace repo files: {exc}") from exc

    log.info(f"  Repo contains {len(all_paths)} file(s)")

    # --- Find data files for this config ---
    # Extract the category prefix: "computers_xlarge" → "computers"
    category = config.split("_")[0]   # e.g. "computers", "cameras", etc.

    def _matches(path: str) -> bool:
        """Only accept files that live in the correct category subdirectory."""
        path_lower = path.lower()
        # Must start with "<category>/" AND contain the size token
        return (
            path_lower.startswith(f"{category}/") and
            train_size in path_lower
        )

    parquet = [p for p in all_paths if _matches(p) and p.endswith(".parquet")]
    json_gz = [p for p in all_paths if _matches(p) and p.endswith(".json.gz")]

    if not parquet and not json_gz:
        available = [p for p in all_paths if not p.endswith(".py")][:20]
        raise RuntimeError(
            f"No data files found for config '{config}' in {_HF_DATASET}.\n"
            f"Files in repo (first 20): {available}"
        )

    cache_kwargs: dict = {}
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_kwargs["cache_dir"] = str(cache_dir)

    dfs: dict[str, list[pd.DataFrame]] = {
        "train": [], "validation": [], "test": []
    }

    def _classify(path: str) -> str:
        p = path.lower()
        if "test" in p or "gs" in p or "gold" in p:
            return "test"
        if "valid" in p or "val" in p:
            return "validation"
        return "train"

    target_files = parquet if parquet else json_gz
    log.info(f"  Downloading {len(target_files)} file(s) …")

    for filepath in target_files:
        local = hf_hub_download(
            _HF_DATASET,
            filename=filepath,
            repo_type="dataset",
            **cache_kwargs,
        )
        if filepath.endswith(".parquet"):
            df = pd.read_parquet(local)
        else:
            df = _normalise_wdc(_load_json_gz(Path(local)))
        df = _normalise_hf(df)
        split = _classify(filepath)
        dfs[split].append(df)

    def _cat(key: str) -> pd.DataFrame:
        parts = dfs[key]
        if not parts:
            return pd.DataFrame(columns=_UNIFIED_COLS)
        return pd.concat(parts, ignore_index=True)

    return _cat("train"), _cat("validation"), _cat("test")


# ── Strategy 2 & 3: Direct WDC download (requests → curl) ─────────────────────

def _try_direct_download(
    train_size: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download JSON.gz files from the WDC LSPC server.

    Uses requests.Session (headers survive HTTP→HTTPS redirect); falls back
    to system curl when requests returns 403.
    """
    key = f"train_{train_size}"
    if key not in _WDC_URLS:
        raise ValueError(
            f"Unknown train_size '{train_size}'. "
            f"Choose from: {list(_HF_CONFIG_MAP.keys())}"
        )

    # --- Training file ---
    train_url = _WDC_URLS[key]
    train_gz = raw_dir / train_url.split("/")[-1]
    if not train_gz.exists():
        _download_file(train_url, train_gz)
    log.info(f"Parsing [cyan]{train_gz.name}[/cyan] …")
    train_df = _normalise_wdc(_load_json_gz(train_gz))

    # --- Test / gold-standard file ---
    test_url = _WDC_URLS["test"]
    test_gz = raw_dir / test_url.split("/")[-1]
    try:
        if not test_gz.exists():
            _download_file(test_url, test_gz)
        test_df = _normalise_wdc(_load_json_gz(test_gz))
    except Exception as exc:
        log.warning(f"Test-set download failed ({exc}) — test split will be empty.")
        test_df = pd.DataFrame(columns=_UNIFIED_COLS)

    # --- Validation file (try multiple candidate names) ---
    val_df = pd.DataFrame(columns=_UNIFIED_COLS)
    for val_name in _WDC_VAL_CANDIDATES:
        val_url = _WDC_BASE + "computers/" + val_name
        val_gz = raw_dir / val_name
        try:
            if not val_gz.exists():
                _download_file(val_url, val_gz)
            val_df = _normalise_wdc(_load_json_gz(val_gz))
            log.info(f"Validation split: {val_name} ({len(val_df):,} pairs)")
            break
        except Exception:
            if val_gz.exists():
                val_gz.unlink()
            continue

    return train_df, val_df, test_df


# ── Public API ─────────────────────────────────────────────────────────────────

def download_wdc_computers(
    raw_dir: Path,
    train_size: str = "xlarge",
    force_redownload: bool = False,
    prefer_huggingface: bool = True,
) -> tuple[Path, Path, Path]:
    """
    Download and cache the WDC LSPC Computers dataset.

    Args:
        raw_dir:            Directory under which to save Parquet files.
        train_size:         One of 'small', 'medium', 'large', 'xlarge'.
        force_redownload:   If True, re-download even if cache exists.
        prefer_huggingface: Try HuggingFace Hub first (default True).

    Returns:
        Paths to (train.parquet, val.parquet, test.parquet).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / f"computers_train_{train_size}.parquet"
    val_path   = raw_dir / "computers_val.parquet"
    test_path  = raw_dir / "computers_test.parquet"

    if not force_redownload and all(
        p.exists() for p in [train_path, val_path, test_path]
    ):
        log.info("[green]✓[/green] Cached dataset found — skipping download.")
        _log_stats(train_path, val_path, test_path)
        return train_path, val_path, test_path

    train_df = val_df = test_df = None
    last_error: Optional[Exception] = None

    # ── Strategy 1: HuggingFace Hub ─────────────────────────────────────────
    if prefer_huggingface:
        try:
            train_df, val_df, test_df = _try_huggingface(
                train_size, cache_dir=raw_dir / ".hf_cache"
            )
            log.info("[green]✓[/green] Loaded from HuggingFace Hub.")
        except Exception as exc:
            log.warning(
                f"HuggingFace load failed: {exc}\n"
                f"  → Falling back to direct WDC download …"
            )
            last_error = exc

    # ── Strategy 2+3: Direct WDC download (requests → curl) ─────────────────
    if train_df is None:
        try:
            train_df, val_df, test_df = _try_direct_download(train_size, raw_dir)
            log.info("[green]✓[/green] Downloaded from WDC LSPC server.")
        except Exception as exc:
            last_error = exc

    if train_df is None:
        raise RuntimeError(
            f"All download strategies exhausted.\n"
            f"Last error: {last_error}\n\n"
            "─── Manual download instructions ───────────────────────────\n"
            f"  1. Open in your browser:\n"
            f"       {_WDC_REFERER}\n"
            f"  2. Scroll to 'Computers' and download:\n"
            f"       • computers_train_{train_size}.json.gz  (training)\n"
            f"       • computers_gs.json.gz                 (test / gold-standard)\n"
            f"       • computers_valid_66pair.json.gz       (validation, optional)\n"
            f"  3. Place the downloaded files in:\n"
            f"       {raw_dir.resolve()}\n"
            f"  4. Re-run:  uv run python scripts/prepare_data.py --no-hf\n"
            "──────────────────────────────────────────────────────────────"
        ) from last_error

    # ── Derive val from train if empty ──────────────────────────────────────
    if val_df is None or len(val_df) == 0:
        from sklearn.model_selection import train_test_split

        log.warning(
            "No validation split available — splitting 10% off training data."
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.10, random_state=42, stratify=train_df["label"]
        )

    # ── Validate ────────────────────────────────────────────────────────────
    _validate(train_df, "train")
    _validate(val_df, "val")
    _validate(test_df, "test")

    # ── Save ────────────────────────────────────────────────────────────────
    log.info(f"Saving Parquet files to [cyan]{raw_dir}[/cyan] …")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    _log_stats(train_path, val_path, test_path)
    return train_path, val_path, test_path


def _validate(df: pd.DataFrame, split: str) -> None:
    """Assert minimum data quality requirements."""
    required = {"title_left", "title_right", "label"}
    missing = required - set(df.columns)
    assert not missing, f"[{split}] Missing required columns: {missing}"
    if len(df) == 0:
        log.warning(f"  [{split}] is empty (may be expected for test/val)")
        return
    assert not df["label"].isna().any(), f"[{split}] Found NaN in 'label' column"
    assert set(df["label"].unique()).issubset({0, 1}), (
        f"[{split}] 'label' contains values other than 0/1: {df['label'].unique()}"
    )
    log.info(
        f"  [{split}] {len(df):,} pairs  |  "
        f"matches={df['label'].sum():,}  "
        f"non-matches={(df['label'] == 0).sum():,}  |  "
        f"hard-neg={(df['is_hard_negative'] == True).sum():,}"
    )


def _log_stats(train_path: Path, val_path: Path, test_path: Path) -> None:
    for path, name in [
        (train_path, "train"), (val_path, "val"), (test_path, "test")
    ]:
        df = pd.read_parquet(path)
        log.info(
            f"  [{name}] {len(df):,} pairs  |  "
            f"matches={df['label'].sum():,}  "
            f"non-matches={(df['label'] == 0).sum():,}"
        )
