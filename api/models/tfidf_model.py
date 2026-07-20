"""TF-IDF + cosine similarity model."""

from __future__ import annotations

from pathlib import Path

import joblib

from api.core.config import TFIDF_PATH
from api.models.base import SimilarityModel
from api.utils.similarity import clamp01, cosine_similarity


class TfidfModel(SimilarityModel):
    """Bag-of-words TF-IDF model; similarity is the cosine of TF-IDF vectors."""

    model_id = "tfidf"

    def __init__(self, vectorizer) -> None:
        self._vectorizer = vectorizer
        # Number of TF-IDF features acts as the "embedding" dimensionality.
        self.embedding_dim = int(len(vectorizer.get_feature_names_out()))

    @classmethod
    def load(cls, path: Path | str = TFIDF_PATH) -> "TfidfModel":
        """Load a fitted vectorizer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {path}. "
                "Run `python -m scripts.fit_tfidf` first."
            )
        vectorizer = joblib.load(path)
        return cls(vectorizer)

    def predict(self, text1: str, text2: str) -> float:
        matrix = self._vectorizer.transform([text1, text2])
        # ``matrix`` is sparse; densify the two rows for the cosine helper.
        v1 = matrix[0].toarray().ravel()
        v2 = matrix[1].toarray().ravel()
        return clamp01(cosine_similarity(v1, v2))
