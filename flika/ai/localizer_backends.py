"""Localization backends for particle detection.

Provides an abstract base class and a built-in DeepSTORM-style backend
using a PyTorch U-Net for density map prediction.

Heavy imports (torch) are deferred.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..logger import logger


@dataclass
class LocalizerConfig:
    """Configuration for particle localizer training and prediction."""
    backend: str = 'deepstorm'
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: str = 'Auto'
    experiment_name: str = 'flika_localizer'
    checkpoint_path: str = ''
    # Post-processing
    detection_threshold: float = 0.2
    min_distance: int = 3
    # PSF simulation defaults
    psf_sigma: float = 1.5
    n_particles: int = 50
    n_training_frames: int = 500


class LocalizationBackend(ABC):
    """Abstract base class for particle localization backends."""

    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""

    @abstractmethod
    def train(self, images: np.ndarray, density_maps: np.ndarray,
              callback: Optional[Callable] = None) -> None:
        """Train the localizer.

        Parameters
        ----------
        images : ndarray, shape (N, H, W)
            Training images.
        density_maps : ndarray, shape (N, H, W)
            Target density maps.
        callback : callable, optional
            Called as callback(epoch, loss) during training.
        """

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict a density map from a single image.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            Input image.

        Returns
        -------
        density : ndarray, shape (H, W)
            Predicted density map.
        """

    @abstractmethod
    def is_trained(self) -> bool:
        """Return True if the backend has been trained."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a trained model from disk."""


class DeepSTORMBackend(LocalizationBackend):
    """DeepSTORM-style particle localizer using a U-Net.

    Predicts a Gaussian density map from which particle positions
    are extracted via peak detection + centroid refinement.
    """

    def __init__(self, config: Optional[LocalizerConfig] = None):
        self._config = config or LocalizerConfig()
        self._model = None

    def name(self) -> str:
        return 'DeepSTORM'

    def train(self, images: np.ndarray, density_maps: np.ndarray,
              callback: Optional[Callable] = None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ..utils.accel import get_torch_device

        cfg = self._config
        device = get_torch_device(cfg.device)
        if device is None:
            device = torch.device('cpu')

        # Build model
        self._model = _get_deepstorm_net_class()().to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        # Prepare data: (N, 1, H, W)
        X = torch.from_numpy(images.astype(np.float32)).unsqueeze(1).to(device)
        Y = torch.from_numpy(density_maps.astype(np.float32)).unsqueeze(1).to(device)

        # Normalize input images per-frame
        for i in range(X.shape[0]):
            xmin, xmax = X[i].min(), X[i].max()
            if xmax > xmin:
                X[i] = (X[i] - xmin) / (xmax - xmin)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            drop_last=False)

        logger.info("Training DeepSTORM with %d frames on %s", len(images), device)

        self._model.train()
        for epoch in range(cfg.epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self._model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if callback:
                callback(epoch, avg_loss)

        self._model.eval()
        logger.info("DeepSTORM training complete")

    def predict(self, image: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("Model not trained")

        device = next(self._model.parameters()).device
        self._model.eval()

        # Normalize
        img = image.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

        # Pad to multiple of 16 for U-Net
        h, w = img.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = self._model(x)

        density = pred.squeeze().cpu().numpy()

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            density = density[:h, :w]

        return density.astype(np.float32)

    def is_trained(self) -> bool:
        return self._model is not None

    def save(self, path: str) -> None:
        import torch
        if self._model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'state_dict': self._model.state_dict()}, path)
        logger.info("Saved DeepSTORM model to %s", path)

    def load(self, path: str) -> None:
        import torch
        from ..utils.accel import get_torch_device

        device = get_torch_device(self._config.device)
        if device is None:
            device = torch.device('cpu')

        self._model = _get_deepstorm_net_class()().to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self._model.load_state_dict(ckpt['state_dict'])
        self._model.eval()
        logger.info("Loaded DeepSTORM model from %s", path)


def _make_deepstorm_net():
    """Create the DeepSTORM U-Net model. Called only when torch is available."""
    import torch
    import torch.nn as nn

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class DeepSTORMNet(nn.Module):
        """U-Net for density map prediction.

        Architecture:
        - Encoder: DoubleConv + MaxPool x4 (1->32->64->128->256)
        - Bottleneck: DoubleConv (256->512)
        - Decoder: ConvTranspose + DoubleConv x4 with skip connections
        - Output: Conv 1x1 + Sigmoid
        """

        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = _DoubleConv(1, 32)
            self.enc2 = _DoubleConv(32, 64)
            self.enc3 = _DoubleConv(64, 128)
            self.enc4 = _DoubleConv(128, 256)
            self.pool = nn.MaxPool2d(2)

            # Bottleneck
            self.bottleneck = _DoubleConv(256, 512)

            # Decoder
            self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec4 = _DoubleConv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = _DoubleConv(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = _DoubleConv(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = _DoubleConv(64, 32)

            # Output
            self.out_conv = nn.Conv2d(32, 1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            # Bottleneck
            b = self.bottleneck(self.pool(e4))

            # Decoder with skip connections
            d4 = self.up4(b)
            d4 = self._pad_and_cat(d4, e4)
            d4 = self.dec4(d4)

            d3 = self.up3(d4)
            d3 = self._pad_and_cat(d3, e3)
            d3 = self.dec3(d3)

            d2 = self.up2(d3)
            d2 = self._pad_and_cat(d2, e2)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = self._pad_and_cat(d1, e1)
            d1 = self.dec1(d1)

            return self.sigmoid(self.out_conv(d1))

        @staticmethod
        def _pad_and_cat(x, skip):
            """Pad x to match skip dimensions, then concatenate."""
            import torch.nn.functional as F
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh > 0 or dw > 0:
                x = F.pad(x, [0, dw, 0, dh])
            return torch.cat([x, skip], dim=1)

    return DeepSTORMNet


# Lazy class creation — cached helper (module __getattr__ doesn't work for intra-module access)
_DeepSTORMNet = None


def _get_deepstorm_net_class():
    """Return the DeepSTORMNet class, creating it on first call."""
    global _DeepSTORMNet
    if _DeepSTORMNet is None:
        _DeepSTORMNet = _make_deepstorm_net()
    return _DeepSTORMNet


def create_backend(config: LocalizerConfig) -> LocalizationBackend:
    """Factory function to create a localizer backend from config."""
    if config.backend == 'deepstorm':
        return DeepSTORMBackend(config)
    else:
        raise ValueError(f"Unknown localizer backend: {config.backend}")
