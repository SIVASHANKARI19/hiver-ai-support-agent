"""Data loading and preprocessing utilities for Apple Support conversations."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import CORPUS_PATH, GOLDEN_SET_PATH


def load_historical_corpus(corpus_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load pre-processed historical customer-support resolution pairs."""
    path = corpus_path or CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_golden_set(golden_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the 200-sample hand-labelled golden evaluation dataset."""
    path = golden_path or GOLDEN_SET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Golden set file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_corpus_statistics(corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for the loaded historical corpus."""
    total_pairs = len(corpus)
    avg_customer_len = sum(len(x["customer_text"]) for x in corpus) / max(total_pairs, 1)
    avg_brand_len = sum(len(x["brand_reply"]) for x in corpus) / max(total_pairs, 1)
    return {
        "total_pairs": total_pairs,
        "avg_customer_len": round(avg_customer_len, 1),
        "avg_brand_len": round(avg_brand_len, 1)
    }
