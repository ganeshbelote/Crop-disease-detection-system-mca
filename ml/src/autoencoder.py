"""
Convolutional Denoising Autoencoder.

Architecture (for 128x128x3 input):

Encoder:
  Conv(3->32)  + ReLU + MaxPool   -> 64x64x32
  Conv(32->64) + ReLU + MaxPool   -> 32x32x64
  Conv(64->128)+ ReLU + MaxPool   -> 16x16x128   <- latent representation

Decoder (mirror, using transposed convolutions):
  ConvT(128->64) + ReLU           -> 32x32x64
  ConvT(64->32)  + ReLU           -> 64x64x32
  ConvT(32->3)   + Sigmoid        -> 128x128x3   <- reconstructed image

The network is trained to map a noisy input image back to the corresponding
clean image (denoising objective), which forces the encoder to learn a
compact latent representation that captures the underlying leaf structure
rather than pixel-level sensor/compression noise.
"""

import torch
import torch.nn as nn


class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8  -> this is the latent representation
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # *2

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # *4

            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # *8, output back in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def build_autoencoder() -> ConvDenoisingAutoencoder:
    return ConvDenoisingAutoencoder()


if __name__ == "__main__":
    # Quick shape sanity check: python autoencoder.py
    model = build_autoencoder()
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print("Input :", dummy.shape)
    print("Output:", out.shape)
    assert out.shape == dummy.shape, "Autoencoder output shape must match input shape"
    print("OK - autoencoder reconstructs images at the same resolution.")
