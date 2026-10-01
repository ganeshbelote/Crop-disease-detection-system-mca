"""
Train the convolutional denoising autoencoder.

Usage:
    python train_autoencoder.py \
        --data-dir ../data/raw \
        --epochs 20 \
        --batch-size 16 \
        --image-size 128 \
        --lr 1e-3 \
        --noise-std 0.15 \
        --output ../models/autoencoder.pth

Produces:
    ml/models/autoencoder.pth                  (best checkpoint by val loss)
    ml/outputs/autoencoder_training_curve.png   (train/val loss curves)
    ml/outputs/autoencoder_examples.png         (original/noisy/reconstructed)
    ml/outputs/autoencoder_training_log.json    (raw per-epoch metrics)
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from autoencoder import build_autoencoder
from dataset import (
    DenoisingDataset,
    build_file_index,
    discover_classes,
    stratified_split,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train the denoising autoencoder")
    p.add_argument("--data-dir", default=config.RAW_DATA_DIR)
    p.add_argument("--epochs", type=int, default=config.AE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--lr", type=float, default=config.AE_LEARNING_RATE)
    p.add_argument("--noise-std", type=float, default=config.AE_NOISE_STD)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", default=config.AUTOENCODER_WEIGHTS_PATH)
    p.add_argument("--outputs-dir", default=config.OUTPUTS_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            if train:
                optimizer.zero_grad()

            reconstructed = model(noisy)
            loss = criterion(reconstructed, clean)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def save_example_grid(model, dataset, device, out_path, n_examples=5):
    model.eval()
    fig, axes = plt.subplots(3, n_examples, figsize=(3 * n_examples, 9))
    indices = np.random.RandomState(0).choice(len(dataset), size=min(n_examples, len(dataset)), replace=False)

    with torch.no_grad():
        for col, idx in enumerate(indices):
            noisy, clean = dataset[idx]
            recon = model(noisy.unsqueeze(0).to(device)).cpu().squeeze(0)

            clean_img = clean.permute(1, 2, 0).numpy()
            noisy_img = noisy.permute(1, 2, 0).numpy()
            recon_img = recon.permute(1, 2, 0).clamp(0, 1).numpy()

            axes[0, col].imshow(clean_img)
            axes[0, col].set_title("Original")
            axes[1, col].imshow(noisy_img)
            axes[1, col].set_title("Noisy")
            axes[2, col].imshow(recon_img)
            axes[2, col].set_title("Reconstructed")
            for row in range(3):
                axes[row, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    config.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = discover_classes(args.data_dir)
    print(f"Discovered {len(classes)} classes: {classes}")

    paths, labels, class_to_idx = build_file_index(args.data_dir, classes)
    train_split, val_split, _ = stratified_split(paths, labels)
    print(f"Train images: {len(train_split.paths)} | Val images: {len(val_split.paths)}")

    train_ds = DenoisingDataset(train_split, image_size=args.image_size, noise_std=args.noise_std, seed=args.seed)
    val_ds = DenoisingDataset(val_split, image_size=args.image_size, noise_std=args.noise_std, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_autoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                args.output,
            )
            print(f"  -> new best model saved to {args.output} (val_loss={val_loss:.5f})")

    # Plots and logs
    os.makedirs(args.outputs_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder Training/Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputs_dir, "autoencoder_training_curve.png"), dpi=120)
    plt.close()

    # Reload best checkpoint for the example grid
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_example_grid(
        model, val_ds, device, os.path.join(args.outputs_dir, "autoencoder_examples.png")
    )

    with open(os.path.join(args.outputs_dir, "autoencoder_training_log.json"), "w") as f:
        json.dump(
            {
                "history": history,
                "best_val_loss": best_val_loss,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "noise_std": args.noise_std,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Best validation loss: {best_val_loss:.5f}")
    print(f"Best weights: {args.output}")


if __name__ == "__main__":
    main()
