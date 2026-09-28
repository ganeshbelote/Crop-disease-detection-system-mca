"""
Generate a small SYNTHETIC demo dataset.

IMPORTANT — READ THIS BEFORE USING:
This script does NOT produce real crop leaf images and must never be
mistaken for the PlantVillage dataset. It procedurally draws simple
leaf-shaped blobs with class-specific color/blotch patterns purely so that:

  1. The full pipeline (dataset splitting -> autoencoder training ->
     classifier training -> evaluation -> comparison -> inference -> API)
     can be exercised end-to-end in environments without internet access
     (e.g. to smoke-test that the code runs without errors), and
  2. Reviewers can see real (not fabricated) metrics, plots and predictions
     coming out of the pipeline, even though those numbers are meaningless
     for actual agricultural diagnosis.

For the real academic project, download PlantVillage (or an equivalent
dataset) as documented in the main README's "Dataset Setup" section and
place it in ml/data/raw/ instead. Do not submit results produced from this
synthetic generator as if they were results on real leaf photographs.

Usage:
    python generate_synthetic_demo_data.py --out ../data/demo_synthetic --images-per-class 40

Then smoke-test the pipeline against it, e.g.:
    python train_autoencoder.py --data-dir ../data/demo_synthetic --epochs 3
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

import config

# One base color + blotch color per class, chosen only to be visually
# distinguishable from each other — these are NOT derived from real disease
# appearance and carry no biological meaning.
CLASS_STYLES = {
    "Tomato_Healthy": {"base": (34, 139, 34), "blotch": None, "blotch_count": 0},
    "Tomato_Early_Blight": {"base": (60, 120, 40), "blotch": (120, 90, 30), "blotch_count": 6},
    "Tomato_Late_Blight": {"base": (50, 100, 50), "blotch": (70, 60, 45), "blotch_count": 10},
    "Potato_Healthy": {"base": (46, 125, 50), "blotch": None, "blotch_count": 0},
    "Potato_Early_Blight": {"base": (80, 130, 40), "blotch": (110, 80, 35), "blotch_count": 5},
    "Potato_Late_Blight": {"base": (55, 95, 55), "blotch": (65, 55, 40), "blotch_count": 9},
}


def draw_synthetic_leaf(size: int, base_color, blotch_color, blotch_count: int, rng: np.random.RandomState) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(245, 245, 235))
    draw = ImageDraw.Draw(img)

    # Simple leaf silhouette: an ellipse with slight random deformation.
    cx, cy = size // 2, size // 2
    rx, ry = size * 0.38, size * 0.46
    jitter = size * 0.03
    bbox = [
        cx - rx + rng.uniform(-jitter, jitter),
        cy - ry + rng.uniform(-jitter, jitter),
        cx + rx + rng.uniform(-jitter, jitter),
        cy + ry + rng.uniform(-jitter, jitter),
    ]
    color = tuple(int(np.clip(c + rng.randint(-10, 10), 0, 255)) for c in base_color)
    draw.ellipse(bbox, fill=color)

    # Central vein
    draw.line([(cx, cy - ry * 0.9), (cx, cy + ry * 0.9)], fill=(20, 70, 20), width=2)

    # Disease blotches (spots), if any
    if blotch_color and blotch_count > 0:
        for _ in range(blotch_count):
            bx = rng.uniform(cx - rx * 0.7, cx + rx * 0.7)
            by = rng.uniform(cy - ry * 0.7, cy + ry * 0.7)
            r = rng.uniform(size * 0.02, size * 0.06)
            bcolor = tuple(int(np.clip(c + rng.randint(-15, 15), 0, 255)) for c in blotch_color)
            draw.ellipse([bx - r, by - r, bx + r, by + r], fill=bcolor)

    return img


def parse_args():
    p = argparse.ArgumentParser(description="Generate a synthetic demo dataset for pipeline smoke-testing")
    p.add_argument("--out", default=config.DEMO_SYNTHETIC_DATA_DIR)
    p.add_argument("--images-per-class", type=int, default=40)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    os.makedirs(args.out, exist_ok=True)

    for class_name, style in CLASS_STYLES.items():
        class_dir = os.path.join(args.out, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(args.images_per_class):
            img = draw_synthetic_leaf(
                args.size, style["base"], style["blotch"], style["blotch_count"], rng
            )
            img.save(os.path.join(class_dir, f"{class_name.lower()}_{i:04d}.jpg"), quality=90)
        print(f"Generated {args.images_per_class} synthetic images for {class_name}")

    print(f"\nSynthetic demo dataset written to: {args.out}")
    print("Reminder: this is NOT the real PlantVillage dataset. Replace it with real data "
          "for meaningful results (see README > Dataset Setup).")


if __name__ == "__main__":
    main()
