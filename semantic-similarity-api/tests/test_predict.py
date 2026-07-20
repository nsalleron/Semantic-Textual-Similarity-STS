"""Tests for the /predict endpoint using a fake in-memory model."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.core.model_manager import ModelManager
from api.models.base import SimilarityModel


class FakeModel(SimilarityModel):
    model_id = "all-mpnet-base-v2"
    embedding_dim = 768

    def __init__(self, score: float) -> None:
        self._score = score

    def predict(self, text1: str, text2: str) -> float:
        return self._score

    @property
    def device(self) -> str:
        return "cpu"


@pytest.fixture(autouse=True)
def _reset_manager():
    ModelManager.reset()
    yield
    ModelManager.reset()


def _client_with(score: float, threshold: float | None = None) -> TestClient:
    mgr = ModelManager.instance()
    mgr.register(FakeModel(score))
    if threshold is not None:
        mgr._thresholds["all-mpnet-base-v2"] = threshold
    # No `with` context -> lifespan startup does not run (no real model load).
    return TestClient(app)


def test_predict_full_response_shape():
    client = _client_with(0.9, threshold=0.81)
    resp = client.post(
        "/predict",
        json={"model": "mpnet", "text1": "a", "text2": "b"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "model", "similarity", "prediction", "threshold",
        "confidence", "inference_time_ms", "embedding_dim", "device",
    }
    assert body["model"] == "mpnet"
    assert body["similarity"] == 0.9
    assert body["threshold"] == 0.81
    assert body["prediction"] is True
    assert body["embedding_dim"] == 768
    assert body["device"] == "cpu"
    assert body["inference_time_ms"] >= 0.0
    assert body["confidence"] in {"High", "Medium", "Low"}


def test_predict_below_threshold_is_false():
    client = _client_with(0.5, threshold=0.81)
    resp = client.post(
        "/predict", json={"model": "mpnet", "text1": "a", "text2": "b"}
    )
    assert resp.status_code == 200
    assert resp.json()["prediction"] is False


def test_predict_confidence_high_when_far():
    client = _client_with(0.99, threshold=0.8)
    body = client.post(
        "/predict", json={"model": "mpnet", "text1": "a", "text2": "b"}
    ).json()
    assert body["confidence"] == "High"


def test_predict_unknown_model_404():
    client = _client_with(0.9)
    resp = client.post(
        "/predict", json={"model": "nope", "text1": "a", "text2": "b"}
    )
    assert resp.status_code == 404


def test_predict_unloaded_model_503():
    # Register mpnet but request e5 (known id, not loaded) -> 503.
    client = _client_with(0.9)
    resp = client.post(
        "/predict", json={"model": "e5", "text1": "a", "text2": "b"}
    )
    assert resp.status_code == 503


def test_predict_blank_text_422():
    client = _client_with(0.9)
    resp = client.post(
        "/predict", json={"model": "mpnet", "text1": "   ", "text2": "b"}
    )
    assert resp.status_code == 422


def test_predict_missing_field_422():
    client = _client_with(0.9)
    resp = client.post("/predict", json={"model": "mpnet", "text1": "a"})
    assert resp.status_code == 422
