"""
Central configuration for the ML pipeline.

All paths are relative to the `ml/` directory so the project can be moved
or cloned anywhere without editing absolute paths. Every value here can be
overridden via command-line arguments in the individual training scripts.
"""

import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ML_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ML_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")          # expected PlantVillage-style folders (you populate this)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DEMO_SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "demo_synthetic")  # tiny procedurally generated smoke-test set, NOT real data

MODELS_DIR = os.path.join(ML_ROOT, "models")
OUTPUTS_DIR = os.path.join(ML_ROOT, "outputs")

AUTOENCODER_WEIGHTS_PATH = os.path.join(MODELS_DIR, "autoencoder.pth")
CLASSIFIER_WEIGHTS_PATH = os.path.join(MODELS_DIR, "resnet18_classifier.pth")
CLASSIFIER_BASELINE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "resnet18_baseline.pth")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Default subset of PlantVillage classes used by this project. If the raw
# dataset folder contains a different set of class sub-folders, the dataset
# loader will use whatever is actually present instead of this list, but
# this is what the project was designed and documented around.
DEFAULT_CLASSES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Potato_Healthy",
    "Potato_Early_Blight",
    "Potato_Late_Blight",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed python, numpy and torch (if available) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch may not be installed in every environment (e.g. this sandbox
        # has no internet access to install it). The rest of the codebase
        # still needs to be importable for tooling/tests that don't need it.
        pass


# ---------------------------------------------------------------------------
# Image / training defaults (overridable via CLI flags in the scripts)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128          # square images, kept small so CPU training is feasible
BATCH_SIZE = 16
NUM_WORKERS = 2

AE_EPOCHS = 20
AE_LEARNING_RATE = 1e-3
AE_NOISE_STD = 0.15       # Gaussian noise std (on 0-1 scaled images) added during AE training

CLF_EPOCHS = 15
CLF_LEARNING_RATE = 1e-4

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
