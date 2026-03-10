"""Interactive SAM (Segment Anything Model) segmentation dialog.

Uses segment_anything for model loading and inference, providing point-
and box-prompt based segmentation within a pyqtgraph viewer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from flika.logger import logger
import flika.global_vars as g


# SAM checkpoint URLs (official Meta weights)
_SAM_CHECKPOINTS = {
    'vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'filename': 'sam_vit_b_01ec64.pth',
    },
    'vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'filename': 'sam_vit_l_0b3195.pth',
    },
    'vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'filename': 'sam_vit_h_4b8939.pth',
    },
}

_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.FLIKA', 'models')


def _checkpoint_path(model_type: str) -> str:
    """Return the expected local path for a SAM checkpoint."""
    info = _SAM_CHECKPOINTS.get(model_type)
    if info is None:
        raise ValueError(f"Unknown SAM model type: {model_type}")
    return os.path.join(_MODELS_DIR, info['filename'])


def _is_checkpoint_downloaded(model_type: str) -> bool:
    path = _checkpoint_path(model_type)
    return os.path.isfile(path) and os.path.getsize(path) > 1_000_000


@dataclass
class PromptPoint:
    """A single prompt point for SAM."""
    x: float
    y: float
    is_positive: bool = True


class SAMDownloadWorker(QtCore.QThread):
    """Download a SAM checkpoint in a background thread."""
    finished = QtCore.Signal(str)  # model_type
    progress = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, model_type: str, parent=None):
        super().__init__(parent)
        self.model_type = model_type

    def run(self):
        try:
            import requests
            info = _SAM_CHECKPOINTS[self.model_type]
            os.makedirs(_MODELS_DIR, exist_ok=True)
            dest = _checkpoint_path(self.model_type)
            self.progress.emit(f'Downloading SAM {self.model_type} checkpoint...')
            resp = requests.get(info['url'], stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        self.progress.emit(
                            f'Downloading SAM {self.model_type}... {pct}%'
                        )
            self.finished.emit(self.model_type)
        except Exception as e:
            self.error.emit(str(e))


class SAMComputeWorker(QtCore.QThread):
    """Run SAM inference in a background thread."""
    finished = QtCore.Signal(object)  # mask array
    error = QtCore.Signal(str)

    def __init__(self, image, points=None, box=None, model_type='vit_b',
                 device='auto', parent=None):
        super().__init__(parent)
        self.image = image
        self.points = points  # list of PromptPoint
        self.box = box  # (x1, y1, x2, y2) or None
        self.model_type = model_type
        self.device = device

    def run(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor

            checkpoint = _checkpoint_path(self.model_type)
            if not os.path.isfile(checkpoint):
                self.error.emit(
                    f'SAM checkpoint not found. Please download the '
                    f'{self.model_type} model first.'
                )
                return

            # Resolve device
            device = self.device
            if device == 'auto':
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 'cuda'
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = 'mps'
                    else:
                        device = 'cpu'
                except ImportError:
                    device = 'cpu'

            sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            sam.to(device)
            predictor = SamPredictor(sam)

            # Prepare image for SAM (expects uint8 RGB)
            img = self.image
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.dtype != np.uint8:
                img_min, img_max = float(img.min()), float(img.max())
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            predictor.set_image(img)

            point_coords = None
            point_labels = None
            box_input = None

            if self.points:
                point_coords = np.array([[p.x, p.y] for p in self.points])
                point_labels = np.array([1 if p.is_positive else 0 for p in self.points])

            if self.box is not None:
                box_input = np.array(self.box)

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_input,
                multimask_output=True,
            )
            # Take the best mask
            best_idx = np.argmax(scores)
            self.finished.emit(masks[best_idx])
        except Exception as e:
            self.error.emit(str(e))


class SAMSegmentationDialog(QtWidgets.QDialog):
    """Interactive dialog for SAM-based segmentation with point/box prompts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('SAM Interactive Segmentation')
        self.resize(900, 700)
        self._points: List[PromptPoint] = []
        self._mask: Optional[np.ndarray] = None
        self._worker = None
        self._point_items: list = []
        self._overlay_item: Optional[pg.ImageItem] = None
        self._setup_ui()
        self._load_image()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Top controls
        ctrl_layout = QtWidgets.QHBoxLayout()

        ctrl_layout.addWidget(QtWidgets.QLabel('Model:'))
        self.model_combo = QtWidgets.QComboBox()
        for mt in _SAM_CHECKPOINTS:
            label = mt
            if _is_checkpoint_downloaded(mt):
                label += ' (downloaded)'
            self.model_combo.addItem(label, mt)
        ctrl_layout.addWidget(self.model_combo)

        self.download_btn = QtWidgets.QPushButton('Download')
        self.download_btn.clicked.connect(self._download_model)
        ctrl_layout.addWidget(self.download_btn)

        ctrl_layout.addWidget(QtWidgets.QLabel('Device:'))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(['auto', 'cpu', 'cuda', 'mps'])
        ctrl_layout.addWidget(self.device_combo)

        ctrl_layout.addWidget(QtWidgets.QLabel('Mode:'))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(['Point', 'Box'])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        ctrl_layout.addWidget(self.mode_combo)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Image viewer
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        # Connect click handler
        self.image_view.getView().scene().sigMouseClicked.connect(self._on_click)
        layout.addWidget(self.image_view)

        # Box ROI (hidden by default)
        self._box_roi = pg.RectROI([0, 0], [100, 100], pen='cyan')
        self._box_roi.hide()
        self.image_view.getView().addItem(self._box_roi)

        # Status
        self.status_label = QtWidgets.QLabel(
            'Left-click: positive prompt, Right-click: negative prompt'
        )
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        predict_btn = QtWidgets.QPushButton('Predict Mask')
        predict_btn.clicked.connect(self._predict)
        btn_layout.addWidget(predict_btn)

        clear_btn = QtWidgets.QPushButton('Clear Prompts')
        clear_btn.clicked.connect(self._clear_prompts)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()

        accept_win_btn = QtWidgets.QPushButton('Accept as Window')
        accept_win_btn.clicked.connect(self._accept_as_window)
        btn_layout.addWidget(accept_win_btn)

        accept_roi_btn = QtWidgets.QPushButton('Accept as ROI')
        accept_roi_btn.clicked.connect(self._accept_as_roi)
        btn_layout.addWidget(accept_roi_btn)

        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _selected_model_type(self) -> str:
        return self.model_combo.currentData()

    def _refresh_model_combo(self):
        current = self._selected_model_type()
        self.model_combo.clear()
        for mt in _SAM_CHECKPOINTS:
            label = mt
            if _is_checkpoint_downloaded(mt):
                label += ' (downloaded)'
            self.model_combo.addItem(label, mt)
        # Restore selection
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == current:
                self.model_combo.setCurrentIndex(i)
                break

    def _download_model(self):
        model_type = self._selected_model_type()
        if _is_checkpoint_downloaded(model_type):
            self.status_label.setText(
                f'{model_type} is already downloaded. Ready to use.'
            )
            return
        self.download_btn.setEnabled(False)
        self.status_label.setText(f'Downloading SAM {model_type}...')
        worker = SAMDownloadWorker(model_type, self)
        worker.progress.connect(lambda msg: self.status_label.setText(msg))
        worker.finished.connect(self._on_download_finished)
        worker.error.connect(self._on_download_error)
        worker.start()
        self._worker = worker

    def _on_download_finished(self, model_type):
        self.download_btn.setEnabled(True)
        self.status_label.setText(f'SAM {model_type} downloaded successfully')
        self._refresh_model_combo()

    def _on_download_error(self, msg):
        self.download_btn.setEnabled(True)
        self.status_label.setText(f'Download error: {msg}')

    def _load_image(self):
        if g.win is None:
            self.status_label.setText(
                'No window selected. Open an image first.'
            )
            return
        img = g.win.image
        if img.ndim == 3:
            img = img[g.win.currentIndex]
        self.image_view.setImage(img.T)
        self._source_image = img

    def _on_mode_changed(self, mode):
        if mode == 'Box':
            self._box_roi.show()
            self.status_label.setText(
                'Drag the cyan rectangle to define a box prompt'
            )
        else:
            self._box_roi.hide()
            self.status_label.setText(
                'Left-click: positive prompt, Right-click: negative prompt'
            )

    def _on_click(self, event):
        if self.mode_combo.currentText() != 'Point':
            return
        pos = self.image_view.getView().mapSceneToView(event.scenePos())
        x, y = pos.x(), pos.y()

        # Left = positive, right = negative
        is_positive = (event.button() == QtCore.Qt.LeftButton)
        point = PromptPoint(x=x, y=y, is_positive=is_positive)
        self._points.append(point)

        # Draw marker
        color = 'g' if is_positive else 'r'
        scatter = pg.ScatterPlotItem(
            [x], [y], size=12, pen=pg.mkPen(color, width=2),
            brush=pg.mkBrush(color), symbol='o',
        )
        self.image_view.getView().addItem(scatter)
        self._point_items.append(scatter)

        sign = '+' if is_positive else '-'
        self.status_label.setText(
            f'Added {sign} prompt at ({x:.0f}, {y:.0f}). '
            f'Total: {len(self._points)}'
        )

    def _predict(self):
        if not hasattr(self, '_source_image'):
            g.alert('No image loaded.')
            return

        model_type = self._selected_model_type()
        if not _is_checkpoint_downloaded(model_type):
            g.alert(
                f'SAM {model_type} checkpoint not found.\n'
                f'Click "Download" to fetch the model weights first.'
            )
            return

        box = None
        points = self._points if self._points else None

        if self.mode_combo.currentText() == 'Box':
            pos = self._box_roi.pos()
            size = self._box_roi.size()
            box = [pos.x(), pos.y(), pos.x() + size.x(), pos.y() + size.y()]
            points = None  # box mode ignores points

        if points is None and box is None:
            g.alert('Add at least one prompt point or use box mode.')
            return

        self.status_label.setText('Running SAM prediction...')
        self._worker = SAMComputeWorker(
            image=self._source_image,
            points=points,
            box=box,
            model_type=model_type,
            device=self.device_combo.currentText(),
            parent=self,
        )
        self._worker.finished.connect(self._on_predict_finished)
        self._worker.error.connect(self._on_predict_error)
        self._worker.start()

    def _on_predict_finished(self, mask):
        self._mask = mask
        # Show semi-transparent green overlay
        if self._overlay_item is not None:
            self.image_view.getView().removeItem(self._overlay_item)
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask > 0] = [0, 255, 0, 100]  # green semi-transparent
        self._overlay_item = pg.ImageItem(overlay)
        self.image_view.getView().addItem(self._overlay_item)
        self.status_label.setText(
            f'Mask predicted: {mask.sum()} pixels selected'
        )

    def _on_predict_error(self, msg):
        self.status_label.setText(f'SAM error: {msg}')
        g.alert(f'SAM prediction failed:\n{msg}')

    def _clear_prompts(self):
        self._points.clear()
        for item in self._point_items:
            self.image_view.getView().removeItem(item)
        self._point_items.clear()
        if self._overlay_item is not None:
            self.image_view.getView().removeItem(self._overlay_item)
            self._overlay_item = None
        self._mask = None
        self.status_label.setText('Prompts cleared')

    def _accept_as_window(self):
        if self._mask is None:
            g.alert('No mask to accept. Run prediction first.')
            return
        from flika.window import Window
        Window(self._mask.astype(np.float32), 'SAM Mask')
        self.close()

    def _accept_as_roi(self):
        if self._mask is None:
            g.alert('No mask to accept. Run prediction first.')
            return
        if g.win is None:
            g.alert('No target window selected.')
            return
        from flika.roi import makeROI_from_mask
        makeROI_from_mask(g.win, self._mask.astype(bool))
        self.status_label.setText('ROI created from mask')
        self.close()
