"""
Tests for image upload validation (app/utils/image_validation.py) and the
POST /api/predict endpoint's handling of invalid input, which does not
require the ML models or MySQL to be available.

Run with (from backend/):
    pytest tests/test_validation.py -v
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402
from app.utils.image_validation import _safe_filename  # noqa: E402

client = TestClient(app)


def make_test_jpeg_bytes(size=(64, 64), color=(100, 150, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


class TestSafeFilename:
    def test_strips_directory_components(self):
        assert _safe_filename("../../etc/passwd") == "passwd"

    def test_removes_unsafe_characters(self):
        result = _safe_filename("my photo!@#.jpg")
        assert result == "myphoto.jpg"

    def test_falls_back_when_empty(self):
        assert _safe_filename("") == "upload.jpg"


class TestPredictEndpointValidation:
    def test_rejects_non_image_extension(self):
        response = client.post(
            "/api/predict",
            files={"file": ("notes.txt", b"just some text", "text/plain")},
        )
        assert response.status_code == 400
        assert "extension" in response.json()["detail"].lower() or "content type" in response.json()["detail"].lower()

    def test_rejects_empty_file(self):
        response = client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_rejects_corrupted_image_bytes(self):
        response = client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", b"not a real jpeg file content", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_accepts_valid_image_but_may_report_models_unavailable(self):
        """A structurally valid JPEG should pass validation. Whether the
        prediction itself succeeds depends on trained weights being present,
        which this test environment may not have — in that case a 503 with
        a clear message is the CORRECT behavior (never a fabricated
        prediction), so both 200 and 503 are acceptable here."""
        image_bytes = make_test_jpeg_bytes()
        response = client.post(
            "/api/predict",
            files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code in (200, 503)
        if response.status_code == 503:
            assert "model" in response.json()["detail"].lower() or "torch" in response.json()["detail"].lower()
