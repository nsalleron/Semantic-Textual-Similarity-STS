"""End-to-end test: health -> models -> predict through the HTTP stack.

Uses a registered in-memory model so it is fast and requires no downloads,
while still exercising real routing, schemas, and the ModelManager.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.core.model_manager import ModelManager
from api.models.base import SimilarityModel


class E2EModel(SimilarityModel):
    model_id = "all-mpnet-base-v2"
    embedding_dim = 768

    def predict(self, text1: str, text2: str) -> float:
        # Trivial lexical overlap similarity, deterministic and in [0, 1].
        a, b = set(text1.lower().split()), set(text2.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


@pytest.fixture(autouse=True)
def _reset_manager():
    ModelManager.reset()
    yield
    ModelManager.reset()


def test_end_to_end_flow():
    mgr = ModelManager.instance()
    mgr.register(E2EModel())
    mgr._thresholds["all-mpnet-base-v2"] = 0.5
    client = TestClient(app)

    # 1. health
    assert client.get("/health").json() == {"status": "OK"}

    # 2. models lists the registered model with its threshold + aliases
    models_body = client.get("/models").json()
    ids = [m["id"] for m in models_body["models"]]
    assert "all-mpnet-base-v2" in ids
    mpnet_info = next(m for m in models_body["models"] if m["id"] == "all-mpnet-base-v2")
    assert mpnet_info["threshold"] == 0.5
    assert "mpnet" in mpnet_info["aliases"]

    # 3. predict on a highly-overlapping pair -> prediction True
    resp = client.post(
        "/predict",
        json={
            "model": "mpnet",
            "text1": "I love machine learning",
            "text2": "I love machine learning too",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] is True
    assert 0.0 <= body["similarity"] <= 1.0
    assert body["embedding_dim"] == 768
    assert body["confidence"] in {"High", "Medium", "Low"}
