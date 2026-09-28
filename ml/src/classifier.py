"""
ResNet18-based crop disease classifier (transfer learning).

We load the torchvision ResNet18 backbone (ImageNet-pretrained weights when
they can be downloaded/are cached; falls back to random initialization with
a clear warning if not, so the script never silently produces a different
model than what it logs) and replace the final fully-connected layer with a
new one sized to the number of crop-disease classes used in this project.
"""

import warnings

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Create a ResNet18 with its final layer replaced for `num_classes`.

    Args:
        num_classes: number of crop-disease classes.
        pretrained: try to load ImageNet weights (requires internet access
            the first time, torchvision caches them afterwards). If
            downloading fails, falls back to a randomly initialized network
            and prints a clear warning so results are never silently
            misreported as "transfer learning" when they are not.
        freeze_backbone: if True, freeze all convolutional layers and only
            train the new final layer (faster, useful on CPU / small data).
    """
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception as exc:  # pragma: no cover - depends on torchvision version
            warnings.warn(f"Could not resolve pretrained ResNet18 weights enum: {exc}")
            weights = None

    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        warnings.warn(
            "Failed to download/load pretrained ResNet18 weights "
            f"(reason: {exc}). Falling back to a randomly initialized "
            "ResNet18. Reported metrics will reflect training FROM SCRATCH, "
            "not transfer learning, until this is resolved (usually by "
            "ensuring internet access on first run so torchvision can cache "
            "the weights)."
        )
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable
    return model


if __name__ == "__main__":
    # Quick shape sanity check: python classifier.py
    m = build_resnet18_classifier(num_classes=6, pretrained=False)
    dummy = torch.randn(2, 3, 128, 128)
    out = m(dummy)
    print("Output shape:", out.shape)
    assert out.shape == (2, 6)
    print("OK - classifier produces logits for the expected number of classes.")
