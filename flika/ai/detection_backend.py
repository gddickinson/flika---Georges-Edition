"""Object detection backend abstraction.

Provides an abstract ``DetectionBackend`` and a concrete
``UltralyticsBackend`` that wraps YOLOv8/v11 from the *ultralytics*
package.  Heavy imports are deferred so the module can be imported
without the AI packages installed.

Follows the same pattern as ``classifier_backends.py``.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..logger import logger
from .annotations import BoundingBox


@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    model_size: str = 'n'           # n/s/m/l/x
    model_path: str = ''            # custom .pt path (overrides model_size)
    confidence: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = 'Auto'
    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    freeze_layers: int = 0          # for fine-tuning
    augment: bool = True
    experiment_name: str = 'flika_detector'
    # Large images
    use_tiling: bool = False
    tile_size: int = 640
    tile_overlap: int = 64


def prepare_image_for_detection(image: np.ndarray) -> np.ndarray:
    """Convert a flika image (any dtype, 1-4 channels) to uint8 RGB.

    Handles 16-bit, float32/64, grayscale-to-RGB, and applies percentile
    contrast stretching (p2/p98) for microscopy images.
    """
    img = np.asarray(image, dtype=np.float64)

    # Collapse extra dims
    if img.ndim > 3:
        img = img.squeeze()
    if img.ndim > 3:
        img = img[..., :3]

    # Normalize to 0-255 with percentile stretch
    if img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2) * 255, 0, 255)
    else:
        img = np.zeros_like(img)
    img = img.astype(np.uint8)

    # Ensure 3-channel
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    elif img.ndim == 3 and img.shape[-1] != 3:
        # Channels-first to channels-last
        if img.shape[0] in (1, 3, 4):
            img = np.moveaxis(img, 0, -1)
            if img.shape[-1] == 1:
                img = np.concatenate([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
        else:
            # Treat as grayscale
            img = img[:, :, 0] if img.shape[-1] > 3 else img
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)

    return img


class DetectionBackend(ABC):
    """Abstract base class for object detection backends."""

    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""

    @abstractmethod
    def predict(self, image: np.ndarray, config: DetectionConfig) -> List[BoundingBox]:
        """Run detection on a single image.

        Returns a list of BoundingBox detections.
        """

    @abstractmethod
    def train(self, data_yaml: str, config: DetectionConfig,
              callback: Optional[Callable] = None) -> str:
        """Train a model from scratch. Returns saved model path."""

    @abstractmethod
    def fine_tune(self, data_yaml: str, config: DetectionConfig,
                  callback: Optional[Callable] = None) -> str:
        """Fine-tune a pretrained model. Returns saved model path."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a model from *path*."""

    @abstractmethod
    def available_models(self) -> List[dict]:
        """Return list of available model descriptors."""

    @abstractmethod
    def class_names(self) -> List[str]:
        """Return class names from the loaded model."""


def _models_dir() -> str:
    """Return the flika models directory for detectors, creating it if needed."""
    d = os.path.join(os.path.expanduser('~'), '.FLIKA', 'models', 'detectors')
    os.makedirs(d, exist_ok=True)
    return d


def _configure_ultralytics():
    """Point Ultralytics cache/models dir to ~/.FLIKA/models/detectors/.

    This prevents model weights from being downloaded into the current
    working directory.
    """
    try:
        from ultralytics import settings as ul_settings
        models_path = _models_dir()
        ul_settings.update({'weights_dir': models_path})
    except Exception:
        pass


def _resolve_model_path(path_or_name: str) -> str:
    """Resolve a model name like 'yolov8n.pt' to a full path.

    If *path_or_name* is already an absolute path that exists, return it.
    Otherwise check ~/.FLIKA/models/detectors/ first, then fall back to
    the bare name (which lets ultralytics download it).
    """
    if os.path.isabs(path_or_name) and os.path.isfile(path_or_name):
        return path_or_name

    # Check flika model cache
    cached = os.path.join(_models_dir(), os.path.basename(path_or_name))
    if os.path.isfile(cached):
        return cached

    return path_or_name


class UltralyticsBackend(DetectionBackend):
    """Object detection using Ultralytics YOLO (v8/v11)."""

    def __init__(self):
        self._model = None
        self._model_path: str = ''

    def name(self) -> str:
        return 'Ultralytics YOLO'

    def _ensure_model(self, config: DetectionConfig):
        """Load or reload the YOLO model based on config."""
        raw_path = config.model_path or f'yolov8{config.model_size}.pt'
        path = _resolve_model_path(raw_path)
        if self._model is not None and self._model_path == path:
            return
        from ultralytics import YOLO
        _configure_ultralytics()
        # Change to the models dir so any download lands there
        old_cwd = os.getcwd()
        try:
            os.chdir(_models_dir())
            self._model = YOLO(path)
        finally:
            os.chdir(old_cwd)
        self._model_path = path
        logger.info("Loaded YOLO model: %s", path)

    def _get_device(self, config: DetectionConfig) -> str:
        if config.device == 'Auto':
            from ..utils.accel import get_torch_device
            dev = get_torch_device('Auto')
            return str(dev) if dev is not None else 'cpu'
        return config.device.lower()

    def predict(self, image: np.ndarray, config: DetectionConfig) -> List[BoundingBox]:
        from ultralytics import YOLO

        self._ensure_model(config)
        device = self._get_device(config)
        img = prepare_image_for_detection(image)

        if config.use_tiling and (img.shape[0] > config.tile_size or
                                   img.shape[1] > config.tile_size):
            return self._predict_tiled(img, config, device)

        results = self._model.predict(
            img, conf=config.confidence, iou=config.iou_threshold,
            imgsz=config.image_size, device=device, verbose=False,
        )
        return self._results_to_boxes(results)

    def _predict_tiled(self, img: np.ndarray, config: DetectionConfig,
                       device: str) -> List[BoundingBox]:
        """Run tiled prediction for large images and merge with NMS."""
        h, w = img.shape[:2]
        ts = config.tile_size
        overlap = config.tile_overlap
        step = ts - overlap
        all_boxes: List[BoundingBox] = []

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1 = min(y0 + ts, h)
                x1 = min(x0 + ts, w)
                tile = img[y0:y1, x0:x1]

                results = self._model.predict(
                    tile, conf=config.confidence, iou=config.iou_threshold,
                    imgsz=config.image_size, device=device, verbose=False,
                )
                tile_boxes = self._results_to_boxes(results)
                # Offset to full-image coordinates
                for b in tile_boxes:
                    b.x += x0
                    b.y += y0
                all_boxes.extend(tile_boxes)

        return self._nms_boxes(all_boxes, config.iou_threshold)

    def _results_to_boxes(self, results) -> List[BoundingBox]:
        """Convert ultralytics Results to BoundingBox list."""
        boxes = []
        names = self._model.names if self._model else {}
        for result in results:
            if result.boxes is None:
                continue
            for det in result.boxes:
                xyxy = det.xyxy[0].cpu().numpy()
                conf = float(det.conf[0].cpu().numpy())
                cls_id = int(det.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, f'class_{cls_id}')
                boxes.append(BoundingBox(
                    x=float(xyxy[0]), y=float(xyxy[1]),
                    width=float(xyxy[2] - xyxy[0]),
                    height=float(xyxy[3] - xyxy[1]),
                    class_id=cls_id, class_name=cls_name,
                    confidence=conf,
                ))
        return boxes

    @staticmethod
    def _nms_boxes(boxes: List[BoundingBox], iou_thresh: float) -> List[BoundingBox]:
        """Non-maximum suppression across tiles."""
        if not boxes:
            return []
        coords = np.array([[b.x, b.y, b.x + b.width, b.y + b.height] for b in boxes])
        scores = np.array([b.confidence for b in boxes])

        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def train(self, data_yaml: str, config: DetectionConfig,
              callback: Optional[Callable] = None) -> str:
        from ultralytics import YOLO
        _configure_ultralytics()

        model = YOLO(f'yolov8{config.model_size}.yaml')
        device = self._get_device(config)
        save_dir = _models_dir()

        results = model.train(
            data=data_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.image_size,
            lr0=config.learning_rate,
            device=device,
            augment=config.augment,
            name=config.experiment_name,
            project=save_dir,
            verbose=False,
        )
        best_path = os.path.join(save_dir, config.experiment_name, 'weights', 'best.pt')
        final_path = os.path.join(save_dir, f'{config.experiment_name}.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
        logger.info("Training complete. Model saved to %s", final_path)
        return final_path

    def fine_tune(self, data_yaml: str, config: DetectionConfig,
                  callback: Optional[Callable] = None) -> str:
        from ultralytics import YOLO
        _configure_ultralytics()

        base_path = _resolve_model_path(config.model_path or f'yolov8{config.model_size}.pt')
        model = YOLO(base_path)
        device = self._get_device(config)
        save_dir = _models_dir()

        results = model.train(
            data=data_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.image_size,
            lr0=config.learning_rate,
            device=device,
            augment=config.augment,
            freeze=config.freeze_layers,
            name=config.experiment_name,
            project=save_dir,
            verbose=False,
        )
        best_path = os.path.join(save_dir, config.experiment_name, 'weights', 'best.pt')
        final_path = os.path.join(save_dir, f'{config.experiment_name}.pt')
        if os.path.exists(best_path):
            import shutil
            shutil.copy2(best_path, final_path)
        logger.info("Fine-tuning complete. Model saved to %s", final_path)
        return final_path

    def load_model(self, path: str) -> None:
        from ultralytics import YOLO
        _configure_ultralytics()
        resolved = _resolve_model_path(path)
        self._model = YOLO(resolved)
        self._model_path = resolved

    def available_models(self) -> List[dict]:
        models = []
        for variant in ('n', 's', 'm', 'l', 'x'):
            models.append({'name': f'yolov8{variant}', 'path': f'yolov8{variant}.pt',
                           'description': f'YOLOv8 {variant.upper()} — COCO pretrained'})
        for variant in ('n', 's', 'm', 'l', 'x'):
            models.append({'name': f'yolo11{variant}', 'path': f'yolo11{variant}.pt',
                           'description': f'YOLO11 {variant.upper()} — COCO pretrained'})
        # Check for custom models
        custom_dir = os.path.join(os.path.expanduser('~'), '.FLIKA', 'models', 'detectors')
        if os.path.isdir(custom_dir):
            for f in sorted(os.listdir(custom_dir)):
                if f.endswith('.pt'):
                    models.append({'name': f[:-3], 'path': os.path.join(custom_dir, f),
                                   'description': 'Custom model'})
        return models

    def class_names(self) -> List[str]:
        if self._model is not None and hasattr(self._model, 'names'):
            names = self._model.names
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
            return list(names)
        return []
