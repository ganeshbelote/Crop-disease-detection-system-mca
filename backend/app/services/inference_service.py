"""
Inference service: thin adapter between the FastAPI layer and the shared
`ml/src/inference.py` module.

Deliberately does not reimplement any prediction logic — it imports and
reuses the exact same CropDiseasePredictor class used by the offline
evaluation/CLI tooling, so the backend can never drift from (or fake) what
the ML pipeline actually produces.
"""

import sys
from pathlib import Path
from typing import Optional

from app.config import settings

# Make ml/src importable. The ml/ pipeline is deliberately kept as a
# standalone, backend-agnostic package (see ml/src/inference.py's docstring)
# and is reused here rather than duplicated.
ML_SRC_PATH = str(Path(settings.ML_ROOT) / "src")
if ML_SRC_PATH not in sys.path:
    sys.path.insert(0, ML_SRC_PATH)


class InferenceService:
    """Lazily loads the trained models on first use and caches the
    predictor instance for the lifetime of the process."""

    def __init__(self):
        self._predictor = None
        self._import_error: Optional[str] = None

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            from inference import CropDiseasePredictor  # noqa: E402 (path set up above)
        except ImportError as exc:
            self._import_error = (
                "PyTorch/torchvision are not installed in this environment. "
                f"Install backend/requirements.txt to enable predictions. ({exc})"
            )
            raise RuntimeError(self._import_error) from exc

        self._predictor = CropDiseasePredictor(
            autoencoder_weights=settings.AUTOENCODER_WEIGHTS_PATH,
            classifier_weights=settings.CLASSIFIER_WEIGHTS_PATH,
            class_names_path=settings.CLASS_NAMES_PATH,
            image_size=settings.IMAGE_SIZE,
        )
        return self._predictor

    def models_available(self) -> bool:
        """Check whether trained weight files exist, without raising."""
        try:
            predictor = self._get_predictor()
            predictor._ensure_weights_exist()
            return True
        except Exception:
            return False

    def predict(self, image_bytes: bytes):
        """Run the full pipeline. Propagates ModelsNotAvailableError and
        InvalidImageError from ml/src/inference.py unchanged so the API
        layer can map them to the correct HTTP status codes."""
        predictor = self._get_predictor()
        return predictor.predict(image_bytes)


# Singleton used by the API routes (created once per process).
inference_service = InferenceService()
