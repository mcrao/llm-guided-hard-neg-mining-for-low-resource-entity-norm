"""
prompts.py — LLM prompt builders for WDC Computers hard-negative generation.

Hard negatives are confusable product descriptions that share most surface
features with the query product but differ on exactly ONE discriminating
attribute.  The ablation grid (A4) defines four synthesis types:

  phonetic          — misspellings / transliterations that look like the product
                      e.g. "iFone 14 Pro" for "iPhone 14 Pro"
  component_swap    — one spec attribute changed to a similar-but-wrong value
                      e.g. 256 GB → 512 GB, i5 → i7, 15-inch → 17-inch
  abbreviation      — alternate short-form that could plausibly match
                      e.g. "MBP 14 M2" for "MacBook Pro 14-inch M2 Pro"
  semantic_distractor — same brand/category, different model entirely
                      e.g. "Apple MacBook Air 13 M2" for "Apple MacBook Pro 14 M2"

Public API
----------
  build_prompt(product_text, negative_type, strategy, n_negatives) -> str

  Convenience wrappers (used by generate_negatives.py):
    zero_shot(product_text, negative_type, n_negatives) -> str
    few_shot(product_text, negative_type, n_negatives) -> str
    chain_of_thought(product_text, negative_type, n_negatives) -> str

Output contract (all strategies)
---------------------------------
  The LLM must return valid JSON with this exact structure:
    {"hard_negatives": ["text1", "text2", ...]}
  CoT additionally includes:
    {"reasoning": "...", "hard_negatives": ["text1", ...]}
"""

from __future__ import annotations

# ── Few-shot examples (WDC Computers domain) ─────────────────────────────────
# Each tuple: (product_text, negative_type, hard_negative_list)
# Chosen to cover the four types with realistic WDC-style product descriptions.

_FEW_SHOT_EXAMPLES: dict[str, list[tuple[str, list[str]]]] = {
    "phonetic": [
        (
            "Apple iPhone 14 Pro 256GB Space Black",
            ["iFone 14 Pro 256GB Space Black", "Apple iPone 14 Pro 256 GB Space Black"],
        ),
        (
            "Samsung Galaxy S23 Ultra 512GB Phantom Black",
            ["Samsong Galaxy S23 Ultra 512GB Phantom Black", "Samsung Galxy S23 Ultra 512 GB"],
        ),
    ],
    "component_swap": [
        (
            "Apple iPhone 14 Pro 256GB Space Black",
            ["Apple iPhone 14 Pro 512GB Space Black", "Apple iPhone 14 Pro Max 256GB Space Black"],
        ),
        (
            "Dell XPS 15 9520 Intel Core i7-12700H 16GB RAM 512GB SSD",
            ["Dell XPS 15 9520 Intel Core i5-12500H 16GB RAM 512GB SSD",
             "Dell XPS 15 9520 Intel Core i7-12700H 32GB RAM 512GB SSD"],
        ),
    ],
    "abbreviation": [
        (
            "Apple MacBook Pro 14-inch M2 Pro 16GB 512GB Space Gray",
            ["MBP 14 M2 Pro 16GB 512GB Space Gray", "Apple MBP 14\" M2 Pro 16/512 SG"],
        ),
        (
            "Sony PlayStation 5 Digital Edition Console White",
            ["PS5 Digital Edition Console White", "Sony PS5 DE White"],
        ),
    ],
    "semantic_distractor": [
        (
            "Apple MacBook Pro 14-inch M2 Pro 16GB 512GB Space Gray",
            ["Apple MacBook Air 13-inch M2 8GB 512GB Space Gray",
             "Apple MacBook Pro 16-inch M2 Max 32GB 1TB Space Gray"],
        ),
        (
            "Dell XPS 15 9520 Intel Core i7-12700H 16GB RAM 512GB SSD",
            ["Dell XPS 13 9315 Intel Core i5-1230U 8GB RAM 512GB SSD",
             "Dell Inspiron 15 3520 Intel Core i7-1255U 16GB RAM 512GB SSD"],
        ),
    ],
    "mixed": [
        (
            "Apple MacBook Pro 14-inch M2 Pro 16GB 512GB Space Gray",
            [
                "Apple MacBook Pro 14-inch M2 Pro 16GB 256GB Space Gray (component_swap)",
                "MBP 14 M2 Pro 16/512 Space Gray (abbreviation)",
                "Apple MacBook Pro 14-inch M2 Pro 16GB 512GB Space Grey (phonetic)",
                "Apple MacBook Air 13-inch M2 8GB 512GB Space Gray (semantic_distractor)",
            ],
        ),
    ],
}

# ── Type-specific instructions ────────────────────────────────────────────────

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "phonetic": (
        "Generate phonetic/misspelling variants: alternative spellings or "
        "transliterations that look like the same product but differ in "
        "characters (e.g., 'iFone' for 'iPhone', 'Samsong' for 'Samsung'). "
        "Keep all numeric specs (storage, RAM, model number) identical."
    ),
    "component_swap": (
        "Generate component-swap variants: identical product text except ONE "
        "numeric or categorical specification is changed to a plausible "
        "alternative value (e.g., 256GB → 512GB, i5 → i7, 15\" → 17\", "
        "Pro → Max). The rest of the description must stay the same."
    ),
    "abbreviation": (
        "Generate abbreviation variants: shorter product names that could "
        "plausibly refer to the same item but use common abbreviations "
        "(e.g., 'MBP 14 M2' for 'MacBook Pro 14-inch M2 Pro', "
        "'PS5 DE' for 'PlayStation 5 Digital Edition'). "
        "Preserve brand, model, and key spec values."
    ),
    "semantic_distractor": (
        "Generate semantic-distractor variants: different models from the "
        "same brand and product family that share most surface tokens but "
        "are distinct products (e.g., MacBook Air vs MacBook Pro, "
        "XPS 13 vs XPS 15). The hard negative should be a real, plausible "
        "product that someone might confuse with the query."
    ),
    "mixed": (
        "Generate hard negatives using a mix of strategies: some phonetic "
        "variants, some component swaps, some abbreviations, and some "
        "semantic distractors. Label each with its type in parentheses, "
        "e.g., 'iFone 14 Pro 256GB (phonetic)'."
    ),
}

# ── Shared output instruction ─────────────────────────────────────────────────

_OUTPUT_INSTRUCTION = (
    'Return ONLY valid JSON — no markdown fences, no extra text:\n'
    '{"hard_negatives": ["<text1>", "<text2>", ...]}'
)

_COT_OUTPUT_INSTRUCTION = (
    'Return ONLY valid JSON — no markdown fences, no extra text:\n'
    '{"reasoning": "<your analysis>", "hard_negatives": ["<text1>", "<text2>", ...]}'
)


# ── Prompt builders ───────────────────────────────────────────────────────────

def zero_shot(
    product_text: str,
    negative_type: str = "component_swap",
    n_negatives: int = 5,
) -> str:
    """Zero-shot prompt for hard negative generation."""
    _check_type(negative_type)
    instruction = _TYPE_INSTRUCTIONS[negative_type]
    return (
        f"You are an expert in e-commerce product entity matching.\n\n"
        f"Task: Given a product description from the WDC Computers dataset, "
        f"generate {n_negatives} hard negative descriptions.\n\n"
        f"Strategy: {instruction}\n\n"
        f"Rules:\n"
        f"  1. Each hard negative must be plausibly confusable with the query.\n"
        f"  2. Each hard negative must be a DIFFERENT product (not a match).\n"
        f"  3. Use realistic product text in the style of web-scraped listings.\n"
        f"  4. Do not include explanations in the output strings.\n\n"
        f"Product: {product_text}\n\n"
        f"{_OUTPUT_INSTRUCTION}"
    )


def few_shot(
    product_text: str,
    negative_type: str = "component_swap",
    n_negatives: int = 5,
) -> str:
    """Few-shot (3-example) prompt for hard negative generation."""
    _check_type(negative_type)
    instruction = _TYPE_INSTRUCTIONS[negative_type]
    examples = _build_few_shot_block(negative_type)
    return (
        f"You are an expert in e-commerce product entity matching.\n\n"
        f"Task: Given a product description from the WDC Computers dataset, "
        f"generate {n_negatives} hard negative descriptions.\n\n"
        f"Strategy: {instruction}\n\n"
        f"Rules:\n"
        f"  1. Each hard negative must be plausibly confusable with the query.\n"
        f"  2. Each hard negative must be a DIFFERENT product (not a match).\n"
        f"  3. Use realistic product text in the style of web-scraped listings.\n"
        f"  4. Do not include explanations in the output strings.\n\n"
        f"--- Examples ---\n"
        f"{examples}\n"
        f"--- End of examples ---\n\n"
        f"Now generate for:\n"
        f"Product: {product_text}\n\n"
        f"{_OUTPUT_INSTRUCTION}"
    )


def chain_of_thought(
    product_text: str,
    negative_type: str = "component_swap",
    n_negatives: int = 5,
) -> str:
    """Chain-of-thought prompt: reason about discriminating attributes first."""
    _check_type(negative_type)
    instruction = _TYPE_INSTRUCTIONS[negative_type]
    examples = _build_few_shot_block(negative_type)
    return (
        f"You are an expert in e-commerce product entity matching.\n\n"
        f"Task: Given a product description from the WDC Computers dataset, "
        f"generate {n_negatives} hard negative descriptions.\n\n"
        f"Strategy: {instruction}\n\n"
        f"Rules:\n"
        f"  1. Each hard negative must be plausibly confusable with the query.\n"
        f"  2. Each hard negative must be a DIFFERENT product (not a match).\n"
        f"  3. Use realistic product text in the style of web-scraped listings.\n"
        f"  4. Do not include explanations in the output strings.\n\n"
        f"--- Examples ---\n"
        f"{examples}\n"
        f"--- End of examples ---\n\n"
        f"Now reason step by step about the product below:\n"
        f"  (a) Identify all discriminating attributes (brand, model, storage,\n"
        f"      RAM, CPU, colour, screen size, etc.).\n"
        f"  (b) Decide which attribute(s) to perturb for the '{negative_type}' type.\n"
        f"  (c) Write {n_negatives} hard negatives that apply the perturbation.\n\n"
        f"Product: {product_text}\n\n"
        f"{_COT_OUTPUT_INSTRUCTION}"
    )


# ── Public dispatcher ─────────────────────────────────────────────────────────

def build_prompt(
    product_text: str,
    negative_type: str = "component_swap",
    strategy: str = "zero_shot",
    n_negatives: int = 5,
) -> str:
    """
    Build an LLM prompt for hard negative generation.

    Args:
        product_text:   Raw product description from the WDC dataset
                        (typically title + brand + description concatenated).
        negative_type:  One of: phonetic, component_swap, abbreviation,
                        semantic_distractor, mixed.
        strategy:       One of: zero_shot, few_shot, chain_of_thought.
        n_negatives:    Number of hard negatives to generate.

    Returns:
        Prompt string ready to send to the LLM.
    """
    _check_type(negative_type)
    dispatch = {
        "zero_shot":        zero_shot,
        "few_shot":         few_shot,
        "chain_of_thought": chain_of_thought,
    }
    if strategy not in dispatch:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(dispatch.keys())}"
        )
    return dispatch[strategy](product_text, negative_type, n_negatives)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_type(negative_type: str) -> None:
    valid = set(_TYPE_INSTRUCTIONS.keys())
    if negative_type not in valid:
        raise ValueError(
            f"Unknown negative_type '{negative_type}'. "
            f"Choose from: {sorted(valid)}"
        )


def _build_few_shot_block(negative_type: str) -> str:
    """Build the few-shot examples section for a given negative type."""
    key = negative_type if negative_type in _FEW_SHOT_EXAMPLES else "component_swap"
    lines = []
    for i, (product, negatives) in enumerate(_FEW_SHOT_EXAMPLES[key], start=1):
        neg_json = '["' + '", "'.join(negatives) + '"]'
        lines.append(
            f"Example {i}:\n"
            f"  Product: {product}\n"
            f'  Output: {{"hard_negatives": {neg_json}}}'
        )
    return "\n\n".join(lines)
