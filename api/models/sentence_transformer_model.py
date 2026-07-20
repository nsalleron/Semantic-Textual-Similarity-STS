"""Sentence-Transformers backed similarity model.

Wraps any ``sentence-transformers`` model (e.g. all-MiniLM-L6-v2,
all-mpnet-base-v2, intfloat/e5-base-v2). Embeddings are L2-normalized so that
cosine similarity reduces to a dot product. e5 models require ``query:`` /
``passage:`` prefixes, configured via the registry spec.
"""

from __future__ import annotations

from api.core.config import ModelSpec
from api.core.device import get_device
from api.models.base import SimilarityModel
from api.utils.similarity import clamp01, cosine_similarity


def _embedding_dimension(model) -> int:
    """Return the embedding dimension across sentence-transformers versions."""
    # sentence-transformers >= 5 renamed the method to ``get_embedding_dimension``.
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, attr, None)
        if callable(fn):
            return int(fn())
    raise AttributeError("Cannot determine embedding dimension for model.")


class SentenceTransformerModel(SimilarityModel):
    """Similarity model backed by a Sentence-Transformers encoder."""

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = spec.model_id
        self._query_prefix = spec.query_prefix
        self._passage_prefix = spec.passage_prefix
        self._device = device or get_device()
        self._model = SentenceTransformer(spec.source, device=self._device)
        self.embedding_dim = int(_embedding_dimension(self._model))

    @property
    def device(self) -> str:
        return self._device

    def _encode(self, text1: str, text2: str):
        # For asymmetric models (e5) the first text is treated as the query and
        # the second as the passage. For symmetric models the prefixes are
        # empty strings, so this is a no-op.
        t1 = f"{self._query_prefix}{text1}"
        t2 = f"{self._passage_prefix}{text2}"
        return self._model.encode(
            [t1, t2],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def predict(self, text1: str, text2: str) -> float:
        emb = self._encode(text1, text2)
        return clamp01(cosine_similarity(emb[0], emb[1]))
