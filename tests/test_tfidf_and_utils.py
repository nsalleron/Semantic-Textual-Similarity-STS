"""Tests for utils and the TF-IDF model."""

from __future__ import annotations

import math

from api.models.tfidf_model import TfidfModel
from api.utils.confidence import confidence_level
from api.utils.similarity import clamp01, cosine_similarity
from api.utils.timing import measure_ms
from scripts.fit_tfidf import fit_and_save


# --- cosine similarity -----------------------------------------------------
def test_cosine_identical_vectors():
    assert math.isclose(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0, abs_tol=1e-9)


def test_cosine_orthogonal_vectors():
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0, abs_tol=1e-9)


def test_cosine_zero_vector_is_zero():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_clamp01():
    assert clamp01(-0.5) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.42) == 0.42


# --- confidence ------------------------------------------------------------
def test_confidence_high_when_far_from_threshold():
    assert confidence_level(0.99, 0.8, high_margin=0.15, medium_margin=0.05) == "High"


def test_confidence_medium():
    assert confidence_level(0.88, 0.8, high_margin=0.15, medium_margin=0.05) == "Medium"


def test_confidence_low_near_threshold():
    assert confidence_level(0.81, 0.8, high_margin=0.15, medium_margin=0.05) == "Low"


# --- timing ----------------------------------------------------------------
def test_measure_ms_returns_positive_frozen_value():
    with measure_ms() as elapsed:
        sum(range(1000))
    value = elapsed()
    assert value >= 0.0
    # Frozen after exit: two reads are identical.
    assert elapsed() == value


# --- TF-IDF model ----------------------------------------------------------
def _tiny_tfidf_model(tmp_path):
    sentences = [
        "the cat sat on the mat",
        "a dog ran in the park",
        "machine learning is fun",
        "i love artificial intelligence",
    ]
    path = tmp_path / "vectorizer.joblib"
    fit_and_save(sentences, path=path, max_features=1000, ngram_range=(1, 2))
    return TfidfModel.load(path)


def test_tfidf_identical_text_similarity_is_one(tmp_path):
    model = _tiny_tfidf_model(tmp_path)
    sim = model.predict("machine learning is fun", "machine learning is fun")
    assert math.isclose(sim, 1.0, abs_tol=1e-6)


def test_tfidf_disjoint_text_similarity_is_low(tmp_path):
    model = _tiny_tfidf_model(tmp_path)
    sim = model.predict("the cat sat on the mat", "machine learning is fun")
    assert sim < 0.2


def test_tfidf_similarity_in_unit_range(tmp_path):
    model = _tiny_tfidf_model(tmp_path)
    sim = model.predict("a dog ran in the park", "the cat sat on the mat")
    assert 0.0 <= sim <= 1.0


def test_tfidf_embedding_dim_is_feature_count(tmp_path):
    model = _tiny_tfidf_model(tmp_path)
    assert isinstance(model.embedding_dim, int)
    assert model.embedding_dim > 0


def test_tfidf_load_missing_file_raises(tmp_path):
    try:
        TfidfModel.load(tmp_path / "nope.joblib")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
