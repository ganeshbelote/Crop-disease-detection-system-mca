"""
Reusable inference module.

This is the single place where "upload an image -> get a real prediction"
happens. It is imported directly by the FastAPI backend
(backend/app/services/inference_service.py) so the exact same code path used
for offline testing is used in production — there is no separate/duplicated
"demo" prediction logic anywhere in the project.

Pipeline:
    1. Validate + load image
    2. Resize/normalize to the size the models were trained with
    3. Run the trained autoencoder to denoise the image
    4. Run the trained ResNet18 classifier on the denoised image
    5. Softmax -> probabilities -> top-3 predictions

If the trained model weight files are missing, `ModelsNotAvailableError` is
raised with a clear, actionable message. This module NEVER returns a
fabricated prediction.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
from PIL import Image

import config
from autoencoder import build_autoencoder
from classifier import build_resnet18_classifier
from dataset import load_class_mapping


class ModelsNotAvailableError(RuntimeError):
    """Raised when the trained model weight files cannot be found on disk."""


class InvalidImageError(ValueError):
    """Raised when the uploaded file is not a valid, readable image."""


@dataclass
class ClassPrediction:
    class_name: str
    confidence: float


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    top_predictions: List[ClassPrediction]
    original_image_base64: str
    denoised_image_base64: str


class CropDiseasePredictor:
    """Loads the trained models once and exposes a `.predict()` method.

    Intended to be instantiated once (e.g. as a FastAPI startup singleton)
    since loading model weights on every request would be slow.
    """

    def __init__(
        self,
        autoencoder_weights: str = config.AUTOENCODER_WEIGHTS_PATH,
        classifier_weights: str = config.CLASSIFIER_WEIGHTS_PATH,
        class_names_path: str = config.CLASS_NAMES_PATH,
        image_size: int = config.IMAGE_SIZE,
        device: str = None,
    ):
        self.autoencoder_weights = autoencoder_weights
        self.classifier_weights = classifier_weights
        self.class_names_path = class_names_path
        self.image_size = image_size
        self._device_name = device
        self._loaded = False

    def _ensure_weights_exist(self):
        missing = [
            p
            for p in [self.autoencoder_weights, self.classifier_weights, self.class_names_path]
            if not os.path.exists(p)
        ]
        if missing:
            raise ModelsNotAvailableError(
                "Trained model file(s) not found: " + ", ".join(missing) + ". "
                "The models must be trained before predictions can be made. "
                "See the README's 'Autoencoder Training' and 'Classifier Training' "
                "sections (ml/src/train_autoencoder.py and ml/src/train_classifier.py)."
            )

    def load(self):
        """Lazily import torch and load model weights. Raises
        ModelsNotAvailableError if the weight files are missing, and
        ImportError if torch/torchvision are not installed."""
        if self._loaded:
            return

        self._ensure_weights_exist()

        import torch

        self.torch = torch
        self.device = torch.device(
            self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        class_to_idx = load_class_mapping(self.class_names_path)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = len(class_to_idx)

        self.autoencoder = build_autoencoder().to(self.device)
        ae_ckpt = torch.load(self.autoencoder_weights, map_location=self.device)
        self.autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
        self.autoencoder.eval()

        self.classifier = build_resnet18_classifier(num_classes=self.num_classes, pretrained=False).to(self.device)
        clf_ckpt = torch.load(self.classifier_weights, map_location=self.device)
        self.classifier.load_state_dict(clf_ckpt["model_state_dict"])
        self.classifier.eval()

        self._loaded = True

    @staticmethod
    def _array_to_base64(arr: np.ndarray) -> str:
        """Convert a float [0,1] HWC numpy array to a base64-encoded PNG string."""
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict(self, image_bytes: bytes) -> PredictionResult:
        """Run the full pipeline on raw image bytes and return a PredictionResult.

        Raises InvalidImageError if the bytes cannot be decoded as an image,
        and ModelsNotAvailableError if models have not been trained yet.
        """
        self.load()
        torch = self.torch

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                original_arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception as exc:
            raise InvalidImageError(f"Could not read the uploaded file as an image: {exc}") from exc

        tensor = torch.from_numpy(original_arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            denoised_tensor = self.autoencoder(tensor)

            from dataset import IMAGENET_MEAN, IMAGENET_STD

            mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
            classifier_input = (denoised_tensor - mean) / std

            logits = self.classifier(classifier_input)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top3_idx = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            ClassPrediction(class_name=self.idx_to_class[int(i)], confidence=float(probabilities[i]))
            for i in top3_idx
        ]

        denoised_arr = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return PredictionResult(
            predicted_class=top_predictions[0].class_name,
            confidence=top_predictions[0].confidence,
            top_predictions=top_predictions,
            original_image_base64=self._array_to_base64(original_arr),
            denoised_image_base64=self._array_to_base64(denoised_arr),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a single image from the command line")
    parser.add_argument("image_path")
    args = parser.parse_args()

    predictor = CropDiseasePredictor()
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    result = predictor.predict(image_bytes)
    print(f"Predicted: {result.predicted_class} ({result.confidence * 100:.2f}%)")
    print("Top-3:")
    for p in result.top_predictions:
        print(f"  {p.class_name}: {p.confidence * 100:.2f}%")
