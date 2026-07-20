"""Tests for the Sentence-Transformers wrapper.

These require downloading models from the Hugging Face Hub and the torch stack,
so they are marked ``download``/``slow`` and skipped unless those extras are
installed. Run explicitly with::

    pytest -m download tests/test_sentence_transformer.py
"""

from __future__ import annotations

import pytest

from api.core.config import MODEL_REGISTRY

pytestmark = [pytest.mark.download, pytest.mark.slow]

pytest.importorskip("sentence_transformers")


def _make(model_key: str):
    from api.models.sentence_transformer_model import SentenceTransformerModel

    return SentenceTransformerModel(MODEL_REGISTRY[model_key])


@pytest.mark.parametrize(
    "model_key,expected_dim",
    [
        ("all-MiniLM-L6-v2", 384),
        ("all-mpnet-base-v2", 768),
        ("e5-base-v2", 768),
    ],
)
def test_embedding_dim(model_key, expected_dim):
    model = _make(model_key)
    assert model.embedding_dim == expected_dim


def test_identical_text_high_similarity():
    model = _make("all-MiniLM-L6-v2")
    sim = model.predict("I love machine learning", "I love machine learning")
    assert sim > 0.99


def test_related_greater_than_unrelated():
    model = _make("all-MiniLM-L6-v2")
    related = model.predict("I love machine learning", "I enjoy AI and ML")
    unrelated = model.predict("I love machine learning", "The weather is cold today")
    assert related > unrelated
    assert 0.0 <= related <= 1.0 and 0.0 <= unrelated <= 1.0


def test_e5_prefixes_applied():
    model = _make("e5-base-v2")
    assert model._query_prefix == "query: "
    assert model._passage_prefix == "passage: "
