"""Fit and persist a TF-IDF vectorizer on the STS Benchmark corpus.

Usage:
    python -m scripts.fit_tfidf [--max-features 50000]

Produces ``artifacts/tfidf/vectorizer.joblib``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from api.core.config import TFIDF_PATH


def load_stsb_sentences(split: str = "train") -> List[str]:
    """Load all sentences (sentence1 + sentence2) from an STSB split."""
    from datasets import load_dataset

    dataset = load_dataset("PhilipMay/stsb_multi_mt", "en", split=split)
    sentences: List[str] = []
    for row in dataset:
        sentences.append(str(row["sentence1"]))
        sentences.append(str(row["sentence2"]))
    return sentences


def fit_and_save(
    sentences: Iterable[str],
    path: Path | str = TFIDF_PATH,
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on ``sentences`` and persist it to ``path``."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(list(sentences))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    return vectorizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit TF-IDF vectorizer on STSB.")
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    print("Loading STSB train sentences...")
    sentences = load_stsb_sentences("train")
    print(f"Fitting TF-IDF on {len(sentences)} sentences...")
    vectorizer = fit_and_save(sentences, max_features=args.max_features)
    n_features = len(vectorizer.get_feature_names_out())
    print(f"Saved vectorizer with {n_features} features to {TFIDF_PATH}")


if __name__ == "__main__":
    main()
