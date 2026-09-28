"""
Tests for GET /api/health.

Uses FastAPI's TestClient. Database/model availability will legitimately be
False in an environment without MySQL/trained weights — these tests check
the endpoint's CONTRACT (it always returns 200 with the documented shape),
not that a fully deployed environment is present.

Run with (from backend/):
    pytest tests/test_health.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_response_shape():
    response = client.get("/api/health")
    data = response.json()
    assert "status" in data
    assert "database_connected" in data
    assert "models_available" in data
    assert "message" in data
    assert isinstance(data["database_connected"], bool)
    assert isinstance(data["models_available"], bool)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()
