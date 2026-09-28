"""
Unit tests for the pure data/preprocessing helpers in ml/src/dataset.py.

These tests deliberately avoid importing torch/torchvision so they can run
in any environment (including CI without a deep learning stack installed).
Run with:

    cd ml/tests
    python -m pytest test_preprocessing.py -v

or, without pytest installed:

    cd ml/tests
    python test_preprocessing.py
"""

import os
import sys
import tempfile
import shutil

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import (  # noqa: E402
    build_file_index,
    class_distribution,
    discover_classes,
    list_image_paths,
    load_and_resize,
    stratified_split,
)


def make_fake_dataset(root, classes_and_counts):
    """Create a temp folder tree of tiny solid-color JPEGs for testing."""
    for class_name, count in classes_and_counts.items():
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (32, 32), color=(i % 255, 0, 0))
            img.save(os.path.join(class_dir, f"img_{i}.jpg"))


class TestDiscoverClasses:
    def test_discovers_all_class_folders(self):
        tmp = tempfile.mkdtemp()
        try:
            make_fake_dataset(tmp, {"A": 2, "B": 2})
            classes = discover_classes(tmp)
            assert classes == ["A", "B"]
        finally:
            shutil.rmtree(tmp)

    def test_raises_on_missing_dir(self):
        try:
            discover_classes("/this/path/does/not/exist")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_raises_on_empty_dir(self):
        tmp = tempfile.mkdtemp()
        try:
            try:
                discover_classes(tmp)
                assert False, "Expected FileNotFoundError"
            except FileNotFoundError:
                pass
        finally:
            shutil.rmtree(tmp)


class TestBuildFileIndex:
    def test_builds_consistent_paths_and_labels(self):
        tmp = tempfile.mkdtemp()
        try:
            make_fake_dataset(tmp, {"Healthy": 3, "Diseased": 5})
            classes = discover_classes(tmp)
            paths, labels, class_to_idx = build_file_index(tmp, classes)
            assert len(paths) == 8
            assert len(labels) == 8
            assert set(class_to_idx.keys()) == {"Healthy", "Diseased"}
            assert labels.count(class_to_idx["Diseased"]) == 5
            assert labels.count(class_to_idx["Healthy"]) == 3
        finally:
            shutil.rmtree(tmp)


class TestClassDistribution:
    def test_counts_each_class(self):
        dist = class_distribution([0, 0, 1, 2, 2, 2], ["a", "b", "c"])
        assert dist == {"a": 2, "b": 1, "c": 3}

    def test_zero_count_for_absent_class(self):
        dist = class_distribution([0, 0], ["a", "b"])
        assert dist == {"a": 2, "b": 0}


class TestStratifiedSplit:
    def test_no_overlap_between_splits(self):
        tmp = tempfile.mkdtemp()
        try:
            make_fake_dataset(tmp, {"A": 20, "B": 20})
            classes = discover_classes(tmp)
            paths, labels, _ = build_file_index(tmp, classes)

            train, val, test = stratified_split(paths, labels, 0.7, 0.15, 0.15)

            train_set = set(train.paths)
            val_set = set(val.paths)
            test_set = set(test.paths)

            # No data leakage: an image path can only appear in one split.
            assert train_set.isdisjoint(val_set)
            assert train_set.isdisjoint(test_set)
            assert val_set.isdisjoint(test_set)

            # All images accounted for.
            assert len(train.paths) + len(val.paths) + len(test.paths) == len(paths)
        finally:
            shutil.rmtree(tmp)

    def test_every_class_represented_in_train(self):
        tmp = tempfile.mkdtemp()
        try:
            make_fake_dataset(tmp, {"A": 10, "B": 10, "C": 10})
            classes = discover_classes(tmp)
            paths, labels, class_to_idx = build_file_index(tmp, classes)
            train, _, _ = stratified_split(paths, labels)
            train_labels = set(train.labels)
            assert train_labels == set(class_to_idx.values())
        finally:
            shutil.rmtree(tmp)

    def test_reproducible_with_same_seed(self):
        tmp = tempfile.mkdtemp()
        try:
            make_fake_dataset(tmp, {"A": 10, "B": 10})
            classes = discover_classes(tmp)
            paths, labels, _ = build_file_index(tmp, classes)

            train1, _, _ = stratified_split(paths, labels, seed=123)
            train2, _, _ = stratified_split(paths, labels, seed=123)
            assert train1.paths == train2.paths
        finally:
            shutil.rmtree(tmp)


class TestLoadAndResize:
    def test_output_shape_and_range(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "img.jpg")
            Image.new("RGB", (64, 64), color=(200, 50, 50)).save(path)

            arr = load_and_resize(path, image_size=128)
            assert arr.shape == (128, 128, 3)
            assert arr.dtype == np.float32
            assert arr.min() >= 0.0 and arr.max() <= 1.0
        finally:
            shutil.rmtree(tmp)


class TestListImagePaths:
    def test_ignores_non_image_files(self):
        tmp = tempfile.mkdtemp()
        try:
            Image.new("RGB", (10, 10)).save(os.path.join(tmp, "a.jpg"))
            with open(os.path.join(tmp, "notes.txt"), "w") as f:
                f.write("not an image")
            paths = list_image_paths(tmp)
            assert len(paths) == 1
            assert paths[0].endswith("a.jpg")
        finally:
            shutil.rmtree(tmp)


def _run_all():
    """Fallback runner if pytest is not installed."""
    import inspect

    module = sys.modules[__name__]
    failures = 0
    total = 0
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if not name.startswith("Test"):
            continue
        instance = obj()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, method_name)()
                    print(f"PASS: {name}.{method_name}")
                except AssertionError as e:
                    failures += 1
                    print(f"FAIL: {name}.{method_name}: {e}")
    print(f"\n{total - failures}/{total} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
