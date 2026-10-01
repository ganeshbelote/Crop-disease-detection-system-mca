"""
Train the ResNet18 crop-disease classifier.

Can optionally train on autoencoder-denoised images (--use-autoencoder) so
that the SAME script produces both the "baseline" and "denoised" classifiers
needed for the comparison experiment (compare_models.py).

Usage:
    python train_classifier.py --data-dir ../data/raw --epochs 15
    python train_classifier.py --data-dir ../data/raw --epochs 15 --use-autoencoder \
        --autoencoder-weights ../models/autoencoder.pth \
        --output ../models/resnet18_classifier.pth

Produces:
    ml/models/resnet18_classifier.pth (or resnet18_baseline.pth)
    ml/outputs/classifier_training_curve.png
    ml/outputs/classifier_training_log.json
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from autoencoder import build_autoencoder
from classifier import build_resnet18_classifier
from dataset import (
    ClassificationDataset,
    build_file_index,
    class_distribution,
    discover_classes,
    save_class_mapping,
    stratified_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the ResNet18 classifier")
    p.add_argument("--data-dir", default=config.RAW_DATA_DIR)
    p.add_argument("--epochs", type=int, default=config.CLF_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--lr", type=float, default=config.CLF_LEARNING_RATE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument(
        "--use-autoencoder",
        action="store_true",
        help="Denoise images with a trained autoencoder before classification "
        "(Experiment B). Requires --autoencoder-weights.",
    )
    p.add_argument("--autoencoder-weights", default=config.AUTOENCODER_WEIGHTS_PATH)
    p.add_argument("--output", default=None, help="Defaults depend on --use-autoencoder")
    p.add_argument("--outputs-dir", default=config.OUTPUTS_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


class DenoisedWrapperDataset(torch.utils.data.Dataset):
    """Wraps a ClassificationDataset, running each image through a frozen
    autoencoder before it is returned. The autoencoder operates on raw
    [0,1] images; ImageNet normalization for the classifier is applied
    AFTER denoising so the classifier always receives properly normalized
    input regardless of whether it is used standalone or after the AE.
    """

    def __init__(self, base_paths, base_labels, autoencoder, device, image_size, train, noise_free=True):
        from dataset import get_classifier_transforms, load_and_resize

        self.paths = base_paths
        self.labels = base_labels
        self.autoencoder = autoencoder
        self.device = device
        self.image_size = image_size
        self.transform = get_classifier_transforms(image_size, train=train)
        self._load_and_resize = load_and_resize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = self._load_and_resize(self.paths[idx], self.image_size)  # HWC in [0,1]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            denoised = self.autoencoder(tensor).squeeze(0).cpu().clamp(0, 1)
        denoised_np = denoised.permute(1, 2, 0).numpy()
        out = self.transform(denoised_np)
        return out, self.labels[idx]


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def main():
    args = parse_args()
    config.set_seed(args.seed)

    if args.output is None:
        args.output = (
            config.CLASSIFIER_WEIGHTS_PATH if args.use_autoencoder else config.CLASSIFIER_BASELINE_WEIGHTS_PATH
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = discover_classes(args.data_dir)
    paths, labels, class_to_idx = build_file_index(args.data_dir, classes)
    print(f"Class distribution (all data): {class_distribution(labels, classes)}")

    save_class_mapping(class_to_idx, config.CLASS_NAMES_PATH)

    train_split, val_split, _ = stratified_split(paths, labels)
    print(f"Train images: {len(train_split.paths)} | Val images: {len(val_split.paths)}")

    if args.use_autoencoder:
        print(f"Loading autoencoder from {args.autoencoder_weights} to denoise inputs (Experiment B)...")
        if not os.path.exists(args.autoencoder_weights):
            raise FileNotFoundError(
                f"Autoencoder weights not found at {args.autoencoder_weights}. "
                "Run train_autoencoder.py first."
            )
        ae = build_autoencoder().to(device)
        checkpoint = torch.load(args.autoencoder_weights, map_location=device)
        ae.load_state_dict(checkpoint["model_state_dict"])
        ae.eval()
        for p in ae.parameters():
            p.requires_grad = False

        train_ds = DenoisedWrapperDataset(train_split.paths, train_split.labels, ae, device, args.image_size, train=True)
        val_ds = DenoisedWrapperDataset(val_split.paths, val_split.labels, ae, device, args.image_size, train=False)
    else:
        train_ds = ClassificationDataset(train_split, image_size=args.image_size, train=True)
        val_ds = ClassificationDataset(val_split, image_size=args.image_size, train=False)

    # num_workers=0 when using the autoencoder wrapper: the AE forward pass
    # inside __getitem__ uses `device`, which does not pickle safely across
    # worker processes for CUDA tensors and is unnecessary overhead on CPU.
    num_workers = 0 

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = build_resnet18_classifier(
        num_classes=len(classes), pretrained=True, freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "used_autoencoder": args.use_autoencoder,
                },
                args.output,
            )
            print(f"  -> new best model saved to {args.output} (val_acc={val_acc:.4f})")

    os.makedirs(args.outputs_dir, exist_ok=True)
    tag = "denoised" if args.use_autoencoder else "baseline"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title(f"Loss ({tag})")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title(f"Accuracy ({tag})")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outputs_dir, f"classifier_training_curve_{tag}.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(args.outputs_dir, f"classifier_training_log_{tag}.json"), "w") as f:
        json.dump(
            {
                "history": history,
                "best_val_acc": best_val_acc,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "used_autoencoder": args.use_autoencoder,
                "classes": classes,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best weights: {args.output}")


if __name__ == "__main__":
    main()
