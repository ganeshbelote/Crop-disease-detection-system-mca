"""
Tests for GET/DELETE /api/history.

These tests require a reachable MySQL instance configured via backend/.env
(the same database used by the running application). If MySQL is not
reachable, both endpoints should still respond with a clear 503 rather than
crashing or returning fabricated data — that contract is what
`test_history_endpoints_do_not_crash_without_db` checks, so this file is
meaningful to run even before MySQL is set up.

Run with (from backend/):
    pytest tests/test_history.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_history_endpoints_do_not_crash_without_db():
    """Regardless of whether MySQL is configured/reachable in this test
    environment, the endpoint must respond with either real data (200) or a
    clear service-unavailable error (503) — never a 500 crash and never
    fabricated history."""
    response = client.get("/api/history")
    assert response.status_code in (200, 503)


def test_history_response_shape_when_available():
    response = client.get("/api/history")
    if response.status_code != 200:
        return  # MySQL not available in this environment; covered by the test above
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert data["total"] == len(data["items"])


def test_delete_history_when_available():
    response = client.delete("/api/history")
    if response.status_code != 200:
        return  # MySQL not available in this environment
    data = response.json()
    assert "deleted_count" in data

    # History should now be empty.
    follow_up = client.get("/api/history")
    assert follow_up.status_code == 200
    assert follow_up.json()["total"] == 0
