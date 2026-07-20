"""Tests for device detection and the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app
from api.core.device import get_device

client = TestClient(app)


def test_get_device_returns_valid_value():
    assert get_device() in {"cuda", "mps", "cpu"}


def test_get_device_is_cached():
    # Cached result: repeated calls return the same object/value.
    assert get_device() == get_device()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "OK"}
