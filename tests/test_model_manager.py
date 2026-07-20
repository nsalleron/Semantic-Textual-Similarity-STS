"""Tests for the ModelManager singleton and the /models endpoint.

These use a lightweight fake model so no downloads are required.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.core.model_manager import ModelManager
from api.models.base import SimilarityModel


class FakeMpnet(SimilarityModel):
    model_id = "all-mpnet-base-v2"
    embedding_dim = 768

    def predict(self, text1: str, text2: str) -> float:
        return 1.0 if text1 == text2 else 0.0


@pytest.fixture(autouse=True)
def _reset_manager():
    ModelManager.reset()
    yield
    ModelManager.reset()


def test_singleton_identity():
    assert ModelManager.instance() is ModelManager.instance()


def test_register_and_alias_resolution():
    mgr = ModelManager.instance()
    mgr.register(FakeMpnet())
    # canonical id and alias both resolve to the same object
    assert mgr.get("all-mpnet-base-v2") is mgr.get("mpnet")
    assert mgr.is_loaded("mpnet")
    assert "all-mpnet-base-v2" in mgr.available()


def test_get_unknown_raises_keyerror():
    mgr = ModelManager.instance()
    with pytest.raises(KeyError):
        mgr.get("does-not-exist")


def test_threshold_fallback_default():
    mgr = ModelManager.instance()
    # No thresholds loaded -> default (0.8)
    assert mgr.threshold_for("all-mpnet-base-v2") == 0.8


def test_threshold_from_file(tmp_path):
    mgr = ModelManager.instance()
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"all-mpnet-base-v2": 0.81}))
    mgr.load_thresholds(path)
    assert mgr.threshold_for("all-mpnet-base-v2") == 0.81


def test_model_infos_shape():
    mgr = ModelManager.instance()
    mgr.register(FakeMpnet())
    infos = mgr.model_infos()
    assert len(infos) == 1
    info = infos[0]
    assert info["id"] == "all-mpnet-base-v2"
    assert info["kind"] == "sentence_transformer"
    assert info["embedding_dim"] == 768
    assert "mpnet" in info["aliases"]


def test_models_endpoint_lists_registered_model():
    # Do not use the `with` context manager so the startup lifespan (which would
    # try to load real models) does not run; register a fake model instead.
    ModelManager.instance().register(FakeMpnet())
    client = TestClient(app)
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "device" in body
    ids = [m["id"] for m in body["models"]]
    assert "all-mpnet-base-v2" in ids
