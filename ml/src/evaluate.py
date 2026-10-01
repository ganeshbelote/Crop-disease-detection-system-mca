"""
Evaluate a trained ResNet18 classifier on the held-out TEST split.

Produces real (never invented) metrics:
    - accuracy, precision, recall, F1 (macro and weighted)
    - full classification report
    - confusion matrix (figure + raw counts)

Usage:
    python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_baseline.pth
    python evaluate.py --data-dir ../data/raw --weights ../models/resnet18_classifier.pth \
        --use-autoencoder --autoencoder-weights ../models/autoencoder.pth
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

import config
from autoencoder import build_autoencoder
from classifier import build_resnet18_classifier
from dataset import (
    ClassificationDataset,
    build_file_index,
    discover_classes,
    stratified_split,
)
from train_classifier import DenoisedWrapperDataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained classifier on the test split")
    p.add_argument("--data-dir", default=config.RAW_DATA_DIR)
    p.add_argument("--weights", required=True)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--use-autoencoder", action="store_true")
    p.add_argument("--autoencoder-weights", default=config.AUTOENCODER_WEIGHTS_PATH)
    p.add_argument("--outputs-dir", default=config.OUTPUTS_DIR)
    p.add_argument("--tag", default=None, help="Label used in output filenames (defaults to baseline/denoised)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    config.set_seed(args.seed)
    tag = args.tag or ("denoised" if args.use_autoencoder else "baseline")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Model weights not found at {args.weights}. Train the classifier first "
            "(train_classifier.py) — metrics cannot be computed without a trained model."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = discover_classes(args.data_dir)
    paths, labels, class_to_idx = build_file_index(args.data_dir, classes)
    _, _, test_split = stratified_split(paths, labels)
    print(f"Test images: {len(test_split.paths)}")

    if args.use_autoencoder:
        ae = build_autoencoder().to(device)
        ae_ckpt = torch.load(args.autoencoder_weights, map_location=device)
        ae.load_state_dict(ae_ckpt["model_state_dict"])
        ae.eval()
        test_ds = DenoisedWrapperDataset(test_split.paths, test_split.labels, ae, device, args.image_size, train=False)
        num_workers = 0
    else:
        test_ds = ClassificationDataset(test_split, image_size=args.image_size, train=False)
        num_workers = 0

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = build_resnet18_classifier(num_classes=len(classes), pretrained=False).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, true_labels = collect_predictions(model, test_loader, device)

    accuracy = float((preds == true_labels).mean())
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted", zero_division=0
    )
    report = classification_report(true_labels, preds, target_names=classes, zero_division=0, output_dict=True)
    report_text = classification_report(true_labels, preds, target_names=classes, zero_division=0)

    print(report_text)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro    P/R/F1: {precision_macro:.4f} / {recall_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted P/R/F1: {precision_weighted:.4f} / {recall_weighted:.4f} / {f1_weighted:.4f}")

    cm = confusion_matrix(true_labels, preds, labels=list(range(len(classes))))

    os.makedirs(args.outputs_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputs_dir, f"confusion_matrix_{tag}.png"), dpi=120)
    plt.close()

    metrics = {
        "tag": tag,
        "weights_path": args.weights,
        "num_test_images": len(test_split.paths),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }

    out_json = os.path.join(args.outputs_dir, f"evaluation_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {out_json}")
    print(f"Saved confusion matrix to confusion_matrix_{tag}.png")


if __name__ == "__main__":
    main()
