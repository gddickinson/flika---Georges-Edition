"""Classifier backends for pixel classification.

Provides an abstract base class and two implementations:
- RandomForestBackend: sklearn-based, fast training, no GPU needed
- CNNBackend: PyTorch encoder-decoder, GPU-accelerated

Heavy imports (sklearn, torch) are deferred.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from flika.logger import logger


@dataclass
class ClassifierConfig:
    """Configuration for pixel classifier training."""
    backend: str = 'random_forest'   # 'random_forest' or 'cnn'
    n_estimators: int = 100          # RF
    max_depth: int = 10              # RF
    epochs: int = 50                 # CNN
    batch_size: int = 32             # CNN
    learning_rate: float = 1e-3      # CNN
    device: str = 'Auto'
    experiment_name: str = 'flika_classifier'


class ClassifierBackend(ABC):
    """Abstract base class for pixel classification backends."""

    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray,
              n_classes: int, callback: Optional[Callable] = None) -> None:
        """Train the classifier.

        Parameters
        ----------
        features : ndarray, shape (N, F)
            Feature vectors for labeled pixels.
        labels : ndarray, shape (N,)
            Class labels (1-based).
        n_classes : int
            Total number of classes.
        callback : callable, optional
            Called as callback(epoch, loss) during training.
        """

    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities.

        Parameters
        ----------
        features : ndarray, shape (N, F)
            Feature vectors for all pixels.

        Returns
        -------
        labels : ndarray, shape (N,)
            Predicted class labels.
        probabilities : ndarray, shape (N, C)
            Class probabilities.
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


class RandomForestBackend(ClassifierBackend):
    """Pixel classifier using sklearn RandomForest."""

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self._config = config or ClassifierConfig()
        self._model = None

    def name(self) -> str:
        return 'Random Forest'

    def train(self, features: np.ndarray, labels: np.ndarray,
              n_classes: int, callback: Optional[Callable] = None) -> None:
        from sklearn.ensemble import RandomForestClassifier

        cfg = self._config
        self._model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            n_jobs=-1,
            random_state=42,
        )

        logger.info("Training Random Forest with %d samples, %d features, %d classes",
                     features.shape[0], features.shape[1], n_classes)

        # RF training is not iterative, but we report progress
        if callback:
            callback(0, 0.0)

        self._model.fit(features, labels)

        if callback:
            callback(1, 0.0)

        logger.info("Random Forest training complete (OOB not enabled)")

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model not trained")
        labels = self._model.predict(features)
        probs = self._model.predict_proba(features)
        return labels.astype(np.int32), probs.astype(np.float32)

    def is_trained(self) -> bool:
        return self._model is not None

    def save(self, path: str) -> None:
        import joblib
        if self._model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Saved Random Forest model to %s", path)

    def load(self, path: str) -> None:
        import joblib
        self._model = joblib.load(path)
        logger.info("Loaded Random Forest model from %s", path)


class CNNBackend(ClassifierBackend):
    """Pixel classifier using a PyTorch encoder-decoder CNN."""

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self._config = config or ClassifierConfig()
        self._model = None
        self._n_features = None
        self._n_classes = None

    def name(self) -> str:
        return 'CNN'

    def train(self, features: np.ndarray, labels: np.ndarray,
              n_classes: int, callback: Optional[Callable] = None) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from flika.utils.accel import get_torch_device

        cfg = self._config
        device = get_torch_device(cfg.device)
        if device is None:
            device = torch.device('cpu')

        self._n_features = features.shape[1]
        self._n_classes = n_classes

        # Build model
        self._model = _PixelCNN(self._n_features, n_classes).to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Prepare data: features (N, F) and labels (N,) — train as 1D
        X = torch.from_numpy(features.astype(np.float32)).to(device)
        y = torch.from_numpy((labels - 1).astype(np.int64)).to(device)  # 0-based for CE

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            drop_last=False)

        logger.info("Training CNN with %d samples, %d features, %d classes on %s",
                     features.shape[0], self._n_features, n_classes, device)

        self._model.train()
        for epoch in range(cfg.epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self._model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if callback:
                callback(epoch, avg_loss)

        self._model.eval()
        logger.info("CNN training complete")

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        if self._model is None:
            raise RuntimeError("Model not trained")

        device = next(self._model.parameters()).device
        self._model.eval()

        X = torch.from_numpy(features.astype(np.float32)).to(device)

        with torch.no_grad():
            # Process in chunks to avoid OOM
            chunk_size = 65536
            all_probs = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i+chunk_size]
                out = self._model(chunk)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0).astype(np.float32)
        labels = probs.argmax(axis=1) + 1  # Back to 1-based
        return labels.astype(np.int32), probs

    def is_trained(self) -> bool:
        return self._model is not None

    def save(self, path: str) -> None:
        import torch
        if self._model is None:
            raise RuntimeError("No model to save")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'state_dict': self._model.state_dict(),
            'n_features': self._n_features,
            'n_classes': self._n_classes,
        }, path)
        logger.info("Saved CNN model to %s", path)

    def load(self, path: str) -> None:
        import torch
        from flika.utils.accel import get_torch_device

        device = get_torch_device(self._config.device)
        if device is None:
            device = torch.device('cpu')

        ckpt = torch.load(path, map_location=device, weights_only=False)
        self._n_features = ckpt['n_features']
        self._n_classes = ckpt['n_classes']
        self._model = _PixelCNN(self._n_features, self._n_classes).to(device)
        self._model.load_state_dict(ckpt['state_dict'])
        self._model.eval()
        logger.info("Loaded CNN model from %s", path)


class _PixelCNN:
    """Simple MLP-based pixel classifier (operates on feature vectors, not spatial patches).

    This is implemented as a plain class wrapping a torch.nn.Sequential,
    created only when torch is available.
    """

    def __new__(cls, n_features: int, n_classes: int):
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        return model


def create_backend(config: ClassifierConfig) -> ClassifierBackend:
    """Factory function to create a classifier backend from config."""
    if config.backend == 'random_forest':
        return RandomForestBackend(config)
    elif config.backend == 'cnn':
        return CNNBackend(config)
    else:
        raise ValueError(f"Unknown classifier backend: {config.backend}")
