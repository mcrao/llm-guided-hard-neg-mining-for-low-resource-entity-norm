"""
Unit tests for src/wdc_hn/data/dataset.py — build_text.
"""
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wdc_hn.data.dataset import build_text, MAX_DESC_CHARS


def test_all_fields():
    """All fields present → output has all three prefixed sections."""
    result = build_text(
        title="Dell XPS 13 Laptop",
        brand="Dell",
        description="13.4-inch FHD+ display",
    )
    assert result == "[TITLE] Dell XPS 13 Laptop [BRAND] Dell [DESC] 13.4-inch FHD+ display"


def test_missing_brand():
    """Omitting brand → [BRAND] token absent, title and desc still present."""
    result = build_text(
        title="Dell XPS 13",
        brand=None,
        description="13.4-inch display",
    )
    assert "[BRAND]" not in result
    assert "[TITLE] Dell XPS 13" in result
    assert "[DESC] 13.4-inch display" in result


def test_description_truncated():
    """Description longer than MAX_DESC_CHARS chars is truncated to exactly MAX_DESC_CHARS."""
    long_desc = "x" * (MAX_DESC_CHARS + 50)
    result = build_text(title="Product", description=long_desc)
    # Extract the [DESC] portion
    desc_part = result.split("[DESC] ")[1]
    assert len(desc_part) == MAX_DESC_CHARS
    assert desc_part == "x" * MAX_DESC_CHARS


def test_empty_input():
    """All None inputs → returns '[EMPTY]'."""
    assert build_text() == "[EMPTY]"
    assert build_text(title=None, brand=None, description=None) == "[EMPTY]"


def test_empty_strings():
    """All-empty-string inputs → returns '[EMPTY]'."""
    assert build_text(title="", brand="", description="") == "[EMPTY]"


def test_nan_handled():
    """float('nan') for any field is treated as missing (not output as 'nan')."""
    result = build_text(
        title=float("nan"),
        brand=float("nan"),
        description=float("nan"),
    )
    assert result == "[EMPTY]"
    assert "nan" not in result.lower()


def test_nan_mixed_with_valid():
    """NaN brand with valid title → [BRAND] absent, [TITLE] present."""
    result = build_text(title="Apple iPhone 14", brand=float("nan"))
    assert "[TITLE] Apple iPhone 14" in result
    assert "[BRAND]" not in result


def test_title_only():
    """Only title provided → correct format, no other prefixes."""
    result = build_text(title="Samsung Galaxy S23")
    assert result == "[TITLE] Samsung Galaxy S23"
    assert "[BRAND]" not in result
    assert "[DESC]" not in result
