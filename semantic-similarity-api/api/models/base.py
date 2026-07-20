"""Common interface for all similarity models.

Every backend (TF-IDF, fine-tuned BERT, Sentence-Transformers) implements
:class:`SimilarityModel`, exposing a uniform ``predict(text1, text2) -> float``
that returns a cosine similarity mapped to the ``[0, 1]`` range.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SimilarityModel(ABC):
    """Abstract base class for semantic similarity models."""

    #: Canonical model id (matches the registry key).
    model_id: str
    #: Embedding dimensionality, or ``None`` when not applicable/unknown.
    embedding_dim: int | None = None

    @abstractmethod
    def predict(self, text1: str, text2: str) -> float:
        """Return the similarity between ``text1`` and ``text2`` in ``[0, 1]``."""
        raise NotImplementedError

    @property
    def device(self) -> str:
        """Device the model runs on. Overridden by neural backends."""
        return "cpu"
