"""
Dataset and preprocessing utilities.

Handles:
- discovering classes from a PlantVillage-style folder layout
  (raw/<ClassName>/*.jpg)
- building a reproducible, leak-free train/val/test split (split is done on
  the list of file paths BEFORE any augmentation is attached, so the same
  physical image is never seen in two different splits)
- image loading, resizing and normalization
- class distribution analysis
- torchvision-based Dataset classes for the autoencoder (unsupervised,
  returns (clean_image, noisy_image) pairs) and the classifier (returns
  (image, label))

This module intentionally has no dependency on FastAPI or the web backend so
it can be reused as-is by the training scripts, the evaluation scripts and
the backend inference service.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class SplitData:
    """Container for one split: parallel lists of file paths and integer labels."""

    paths: List[str]
    labels: List[int]


def discover_classes(raw_data_dir: str) -> List[str]:
    """Return the sorted list of class names found as sub-folders of raw_data_dir.

    Raises a clear error if the directory does not exist or contains no
    class sub-folders, since this is the most common setup mistake (the
    dataset has not been downloaded/placed yet).
    """
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_data_dir}\n"
            "Please download the dataset and place class folders inside "
            "ml/data/raw/ as documented in the main README (Dataset Setup "
            "section) before running this script."
        )

    classes = sorted(
        d
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d)) and not d.startswith(".")
    )

    if not classes:
        raise FileNotFoundError(
            f"No class sub-folders found inside {raw_data_dir}.\n"
            "Expected a layout like:\n"
            "  ml/data/raw/Tomato_Healthy/*.jpg\n"
            "  ml/data/raw/Tomato_Early_Blight/*.jpg\n"
            "  ...\n"
            "See the Dataset Setup section of the README."
        )

    return classes


def list_image_paths(class_dir: str) -> List[str]:
    return sorted(
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def build_file_index(raw_data_dir: str, classes: List[str]) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Walk the dataset folder once and build parallel (paths, labels) lists.

    Returns also the class -> index mapping so it can be persisted and
    reused consistently across training, evaluation and inference.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_paths: List[str] = []
    all_labels: List[int] = []

    for class_name in classes:
        class_dir = os.path.join(raw_data_dir, class_name)
        image_paths = list_image_paths(class_dir)
        if not image_paths:
            raise FileNotFoundError(
                f"Class folder '{class_name}' contains no supported images "
                f"({', '.join(SUPPORTED_EXTENSIONS)})."
            )
        all_paths.extend(image_paths)
        all_labels.extend([class_to_idx[class_name]] * len(image_paths))

    return all_paths, all_labels, class_to_idx


def class_distribution(labels: List[int], class_names: List[str]) -> Dict[str, int]:
    counts = Counter(labels)
    return {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}


def stratified_split(
    paths: List[str],
    labels: List[int],
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[SplitData, SplitData, SplitData]:
    """Stratified train/val/test split performed independently per class.

    Splitting per class (rather than globally) prevents rare classes from
    ending up entirely in one split, and is done purely on file paths before
    any Dataset object is constructed, which prevents data leakage: no image
    file can appear in more than one split.
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    rng = np.random.RandomState(seed)
    paths_by_class: Dict[int, List[str]] = {}
    for p, l in zip(paths, labels):
        paths_by_class.setdefault(l, []).append(p)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for label, class_paths in paths_by_class.items():
        class_paths = list(class_paths)
        rng.shuffle(class_paths)
        n = len(class_paths)
        n_train = max(1, int(round(n * train_split)))
        n_val = max(1, int(round(n * val_split))) if n - n_train > 1 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_paths += class_paths[:n_train]
        val_paths += class_paths[n_train : n_train + n_val]
        test_paths += class_paths[n_train + n_val :]

        train_labels += [label] * n_train
        val_labels += [label] * n_val
        test_labels += [label] * n_test

    return (
        SplitData(train_paths, train_labels),
        SplitData(val_paths, val_labels),
        SplitData(test_paths, test_labels),
    )


def load_and_resize(path: str, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load an image from disk, convert to RGB and resize to a square.

    Returns a float32 numpy array in [0, 1] with shape (H, W, 3).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_class_mapping(class_to_idx: Dict[str, int], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(out_path, "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)


def load_class_mapping(path: str) -> Dict[str, int]:
    with open(path) as f:
        data = json.load(f)
    return data["class_to_idx"]


# ---------------------------------------------------------------------------
# PyTorch Dataset classes (imported lazily so this module stays importable
# even in environments where torch is not installed, e.g. for unit-testing
# the pure-numpy helpers above).
# ---------------------------------------------------------------------------
def _torch_and_deps():
    import torch
    from torchvision import transforms

    return torch, transforms


def float_array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float [0,1] HWC array to uint8 so torchvision's ToPILImage
    accepts it. Module-level (not a lambda) so DataLoader workers can pickle it."""
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def get_classifier_transforms(image_size: int = IMAGE_SIZE, train: bool = True):
    """Torchvision transform pipeline for the ResNet18 classifier.

    Augmentation is intentionally limited to flips, mild rotation and mild
    color jitter. We avoid biologically inappropriate augmentation such as
    vertical flips combined with heavy hue shifts (leaves have a consistent
    "up" orientation relative to sunlight/gravity in field photos, and
    disease diagnosis depends on realistic leaf coloration), so we only use
    a small hue/saturation jitter and no vertical flip beyond what a real
    photo could plausibly show.
    """
    _, transforms = _torch_and_deps()

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        return transforms.Compose(
            [
                transforms.Lambda(float_array_to_uint8),
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(float_array_to_uint8),
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ClassificationDataset:
    """torch.utils.data.Dataset for the ResNet18 classifier.

    Defined as a factory function-backed class so importing this module does
    not require torch unless this class is actually instantiated.
    """

    def __new__(cls, split: SplitData, image_size: int = IMAGE_SIZE, train: bool = True):
        torch, _ = _torch_and_deps()
        from torch.utils.data import Dataset

        transform = get_classifier_transforms(image_size, train)

        class _ClassificationDataset(Dataset):
            def __len__(self_inner):
                return len(split.paths)

            def __getitem__(self_inner, idx):
                arr = load_and_resize(split.paths[idx], image_size)
                tensor = transform(arr)
                label = split.labels[idx]
                return tensor, label

        return _ClassificationDataset()


class DenoisingDataset:
    """torch.utils.data.Dataset for the autoencoder.

    Returns (noisy_image, clean_image) tensor pairs in [0, 1], both resized
    to `image_size` x `image_size`. Gaussian noise is added on the fly so
    every epoch sees a slightly different noisy version of each image.
    """

    def __new__(cls, split: SplitData, image_size: int = IMAGE_SIZE, noise_std: float = 0.15, seed: int = RANDOM_SEED):
        torch, _ = _torch_and_deps()
        from torch.utils.data import Dataset

        class _DenoisingDataset(Dataset):
            def __len__(self_inner):
                return len(split.paths)

            def __getitem__(self_inner, idx):
                clean = load_and_resize(split.paths[idx], image_size)  # (H, W, 3) in [0,1]
                noise = np.random.normal(loc=0.0, scale=noise_std, size=clean.shape).astype(np.float32)
                noisy = np.clip(clean + noise, 0.0, 1.0)

                clean_t = torch.from_numpy(clean.transpose(2, 0, 1))
                noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1))
                return noisy_t, clean_t

        return _DenoisingDataset()
