"""Pixel Classifier dialog for flika.

Provides a self-contained dialog with:
- Embedded ImageView with paint overlay for labeling
- Class management (add/remove/color)
- Feature extraction and visualization
- Training (RF or CNN) with live loss curve
- Prediction with result output as a new Window
- Full stack support: labels can be painted on any frame, prediction
  runs on all frames
"""
from __future__ import annotations

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Signal
import pyqtgraph as pg
import numpy as np

from .. import global_vars as g
from ..logger import logger


# ---------------------------------------------------------------------------
# Paint Overlay
# ---------------------------------------------------------------------------

class PaintOverlay(pg.ImageItem):
    """Transparent RGBA overlay for brush painting on an ImageView.

    For stacks, maintains a per-frame label mask ``(T, H, W)`` and displays
    the slice corresponding to the current frame index.
    """

    sigLabelsChanged = Signal()

    def __init__(self, shape, n_frames=1, class_colors=None, parent=None):
        super().__init__(parent=parent)
        h, w = shape[:2]
        self.n_frames = n_frames
        # label_masks: (T, H, W) — 0=unlabeled, 1..N=class
        self.label_masks = np.zeros((n_frames, h, w), dtype=np.int32)
        self._current_frame = 0
        self.class_colors = class_colors or {1: (255, 0, 0), 2: (0, 0, 255)}
        self.active_class = 1
        self.brush_size = 5
        self.opacity_val = 0.5
        self._painting = False
        self.setZValue(10)
        self.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        self._update_display()

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, idx):
        idx = max(0, min(idx, self.n_frames - 1))
        if idx != self._current_frame:
            self._current_frame = idx
            self._update_display()

    @property
    def label_mask(self):
        """Current frame's label mask (convenience accessor)."""
        return self.label_masks[self._current_frame]

    def set_shape(self, shape, n_frames=1):
        """Reset the masks to a new shape."""
        h, w = shape[:2]
        self.n_frames = n_frames
        self.label_masks = np.zeros((n_frames, h, w), dtype=np.int32)
        self._current_frame = 0
        self._update_display()

    def _update_display(self):
        """Rebuild the RGBA overlay from the current frame's label mask."""
        mask = self.label_masks[self._current_frame]
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for cls, color in self.class_colors.items():
            sel = mask == cls
            if sel.any():
                rgba[sel, 0] = color[0]
                rgba[sel, 1] = color[1]
                rgba[sel, 2] = color[2]
                rgba[sel, 3] = int(255 * self.opacity_val)
        self.setImage(rgba)

    def _paint_at(self, pos):
        """Paint a circular brush stroke at the given position."""
        # pyqtgraph: pos.x() → array axis 0, pos.y() → array axis 1
        y, x = int(round(pos.x())), int(round(pos.y()))
        mask = self.label_masks[self._current_frame]
        h, w = mask.shape
        r = self.brush_size

        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)

        if y0 >= y1 or x0 >= x1:
            return

        yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        circle = (yy**2 + xx**2) <= r**2
        mask[y0:y1, x0:x1][circle] = self.active_class
        self._update_display()

    def _erase_at(self, pos):
        """Erase (set to 0) at the given position."""
        # pyqtgraph: pos.x() → array axis 0, pos.y() → array axis 1
        y, x = int(round(pos.x())), int(round(pos.y()))
        mask = self.label_masks[self._current_frame]
        h, w = mask.shape
        r = self.brush_size

        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)

        if y0 >= y1 or x0 >= x1:
            return

        yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        circle = (yy**2 + xx**2) <= r**2
        mask[y0:y1, x0:x1][circle] = 0
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._painting = True
            pos = self.mapFromScene(event.scenePos())
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self._erase_at(pos)
            else:
                self._paint_at(pos)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._painting:
            pos = self.mapFromScene(event.scenePos())
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self._erase_at(pos)
            else:
                self._paint_at(pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._painting:
            self._painting = False
            self.sigLabelsChanged.emit()
        else:
            super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Feature Viewer
# ---------------------------------------------------------------------------

class FeatureOptionsDialog(QtWidgets.QDialog):
    """Dialog for selecting which features to compute and their parameters."""

    def __init__(self, config, parent=None):
        from .features import FeatureConfig
        super().__init__(parent)
        self.setWindowTitle("Feature Options")
        self.setMinimumWidth(420)
        self._config = config  # FeatureConfig to edit
        self._build_ui()
        self._update_count()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- Feature group checkboxes ---
        groups_box = QtWidgets.QGroupBox("Feature Groups")
        groups_layout = QtWidgets.QVBoxLayout(groups_box)

        self.chk_intensity = QtWidgets.QCheckBox("Intensity (1 feature)")
        self.chk_intensity.setChecked(self._config.include_intensity)
        groups_layout.addWidget(self.chk_intensity)

        self.chk_gaussian = QtWidgets.QCheckBox("Gaussian blur / gradient / Laplacian")
        self.chk_gaussian.setChecked(self._config.include_gaussian)
        groups_layout.addWidget(self.chk_gaussian)

        self.chk_edges = QtWidgets.QCheckBox("Edges — Sobel, Scharr, Roberts, Prewitt (4 features)")
        self.chk_edges.setChecked(self._config.include_edges)
        groups_layout.addWidget(self.chk_edges)

        self.chk_lbp = QtWidgets.QCheckBox("Local Binary Pattern (1 feature)")
        self.chk_lbp.setChecked(self._config.include_lbp)
        groups_layout.addWidget(self.chk_lbp)

        self.chk_hessian = QtWidgets.QCheckBox("Hessian — determinant & trace")
        self.chk_hessian.setChecked(self._config.include_hessian)
        groups_layout.addWidget(self.chk_hessian)

        self.chk_gabor = QtWidgets.QCheckBox("Gabor filters")
        self.chk_gabor.setChecked(self._config.include_gabor)
        groups_layout.addWidget(self.chk_gabor)

        self.chk_extras = QtWidgets.QCheckBox("Extras — entropy, structure tensor (4 features)")
        self.chk_extras.setChecked(self._config.include_extras)
        groups_layout.addWidget(self.chk_extras)

        layout.addWidget(groups_box)

        # --- Parameters ---
        params_box = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QGridLayout(params_box)

        params_layout.addWidget(QtWidgets.QLabel("Gaussian sigmas:"), 0, 0)
        self.sigma_edit = QtWidgets.QLineEdit(
            ", ".join(str(s) for s in self._config.gaussian_sigmas))
        self.sigma_edit.setToolTip(
            "Comma-separated sigma values for Gaussian, Hessian features")
        params_layout.addWidget(self.sigma_edit, 0, 1)

        params_layout.addWidget(QtWidgets.QLabel("Gabor frequencies:"), 1, 0)
        self.gabor_freq_edit = QtWidgets.QLineEdit(
            ", ".join(str(f) for f in self._config.gabor_frequencies))
        params_layout.addWidget(self.gabor_freq_edit, 1, 1)

        params_layout.addWidget(QtWidgets.QLabel("Gabor orientations:"), 2, 0)
        self.gabor_orient_spin = QtWidgets.QSpinBox()
        self.gabor_orient_spin.setRange(1, 16)
        self.gabor_orient_spin.setValue(self._config.gabor_orientations)
        params_layout.addWidget(self.gabor_orient_spin, 2, 1)

        layout.addWidget(params_box)

        # --- Feature count ---
        self.count_label = QtWidgets.QLabel("")
        layout.addWidget(self.count_label)

        # Connect all checkboxes and edits to update the count
        for chk in (self.chk_intensity, self.chk_gaussian, self.chk_edges,
                    self.chk_lbp, self.chk_hessian, self.chk_gabor,
                    self.chk_extras):
            chk.toggled.connect(self._update_count)
        self.sigma_edit.textChanged.connect(self._update_count)
        self.gabor_freq_edit.textChanged.connect(self._update_count)
        self.gabor_orient_spin.valueChanged.connect(self._update_count)

        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addStretch()

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)

    def _parse_float_list(self, text):
        """Parse a comma-separated string into a tuple of floats."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        values = []
        for p in parts:
            try:
                values.append(float(p))
            except ValueError:
                pass
        return tuple(values) if values else (1.0,)

    def _build_preview_config(self):
        """Build a FeatureConfig from current UI state (for counting)."""
        from .features import FeatureConfig
        sigmas = self._parse_float_list(self.sigma_edit.text())
        gabor_freqs = self._parse_float_list(self.gabor_freq_edit.text())
        return FeatureConfig(
            gaussian_sigmas=sigmas,
            gabor_frequencies=gabor_freqs,
            gabor_orientations=self.gabor_orient_spin.value(),
            include_intensity=self.chk_intensity.isChecked(),
            include_gaussian=self.chk_gaussian.isChecked(),
            include_edges=self.chk_edges.isChecked(),
            include_lbp=self.chk_lbp.isChecked(),
            include_hessian=self.chk_hessian.isChecked(),
            include_gabor=self.chk_gabor.isChecked(),
            include_extras=self.chk_extras.isChecked(),
        )

    def _update_count(self):
        from .features import FeatureExtractor
        cfg = self._build_preview_config()
        n = FeatureExtractor(cfg).n_features()

        # Build per-group breakdown
        parts = []
        sigmas = cfg.gaussian_sigmas
        gabor_freqs = cfg.gabor_frequencies
        n_orient = cfg.gabor_orientations
        if cfg.include_intensity:
            parts.append("1 intensity")
        if cfg.include_gaussian:
            parts.append(f"{len(sigmas)*3} gaussian")
        if cfg.include_edges:
            parts.append("4 edge")
        if cfg.include_lbp:
            parts.append("1 LBP")
        if cfg.include_hessian:
            parts.append(f"{len(sigmas)*2} hessian")
        if cfg.include_gabor:
            parts.append(f"{len(gabor_freqs)*n_orient} gabor")
        if cfg.include_extras:
            parts.append("4 extras")

        breakdown = " + ".join(parts) if parts else "none"
        self.count_label.setText(f"Total features: {n}  ({breakdown})")

    def _reset_defaults(self):
        from .features import FeatureConfig
        d = FeatureConfig()
        self.chk_intensity.setChecked(d.include_intensity)
        self.chk_gaussian.setChecked(d.include_gaussian)
        self.chk_edges.setChecked(d.include_edges)
        self.chk_lbp.setChecked(d.include_lbp)
        self.chk_hessian.setChecked(d.include_hessian)
        self.chk_gabor.setChecked(d.include_gabor)
        self.chk_extras.setChecked(d.include_extras)
        self.sigma_edit.setText(", ".join(str(s) for s in d.gaussian_sigmas))
        self.gabor_freq_edit.setText(", ".join(str(f) for f in d.gabor_frequencies))
        self.gabor_orient_spin.setValue(d.gabor_orientations)

    def get_config(self):
        """Return the FeatureConfig built from current UI state."""
        return self._build_preview_config()


class FeatureViewerDialog(QtWidgets.QDialog):
    """Grid display of extracted feature images."""

    def __init__(self, feature_stack, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Viewer")
        self.setMinimumSize(700, 500)
        self._feature_stack = feature_stack
        self._feature_names = feature_names
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Scroll area with grid of small ImageView widgets
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)

        n_features = self._feature_stack.shape[-1]
        cols = 3
        for i in range(n_features):
            row, col = divmod(i, cols)
            frame = QtWidgets.QGroupBox(self._feature_names[i])
            frame_layout = QtWidgets.QVBoxLayout(frame)

            img_view = pg.ImageView()
            img_view.setImage(self._feature_stack[:, :, i])
            img_view.setMinimumSize(200, 150)
            frame_layout.addWidget(img_view)

            feat = self._feature_stack[:, :, i]
            stats = f"min={feat.min():.4f}  max={feat.max():.4f}  " \
                    f"mean={feat.mean():.4f}  std={feat.std():.4f}"
            frame_layout.addWidget(QtWidgets.QLabel(stats))
            grid.addWidget(frame, row, col)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Open as Window button
        btn = QtWidgets.QPushButton("Open as Window")
        btn.clicked.connect(self._open_as_window)
        layout.addWidget(btn)

    def _open_as_window(self):
        """Export the feature stack as a flika Window."""
        from ..window import Window
        # Transpose to (N_features, H, W) for a stack
        stack = np.transpose(self._feature_stack, (2, 0, 1))
        Window(stack, name='Feature Stack')
        self.close()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class ClassifierTrainingWorker(QtCore.QThread):
    """Background thread for classifier training."""

    sig_epoch_done = Signal(int, float)    # epoch, loss
    sig_progress = Signal(int)             # percent 0-100
    sig_status = Signal(str)
    sig_finished = Signal()
    sig_error = Signal(str)

    def __init__(self, backend, features, labels, n_classes, config, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.features = features
        self.labels = labels
        self.n_classes = n_classes
        self.config = config
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            total_epochs = self.config.epochs if self.config.backend == 'cnn' else 1

            def callback(epoch, loss):
                self.sig_epoch_done.emit(epoch, loss)
                pct = int(100 * (epoch + 1) / max(total_epochs, 1))
                self.sig_progress.emit(min(pct, 100))

            self.sig_status.emit(f"Training {self.backend.name()}...")
            self.backend.train(self.features, self.labels, self.n_classes,
                               callback=callback)
            self.sig_progress.emit(100)
            self.sig_status.emit("Training complete.")
            self.sig_finished.emit()
        except Exception as e:
            logger.exception("Classifier training failed")
            self.sig_error.emit(str(e))


class ClassifierPredictionWorker(QtCore.QThread):
    """Background thread for classifier prediction over all frames."""

    sig_result = Signal(object, object)   # labels_stack, probs_stack
    sig_progress = Signal(int)
    sig_status = Signal(str)
    sig_error = Signal(str)

    def __init__(self, backend, extractor, image_stack, parent=None):
        """
        Parameters
        ----------
        backend : ClassifierBackend
        extractor : FeatureExtractor
        image_stack : ndarray, shape (T, H, W) or (H, W)
        """
        super().__init__(parent)
        self.backend = backend
        self.extractor = extractor
        self.image_stack = image_stack

    def run(self):
        try:
            stack = self.image_stack
            is_stack = stack.ndim == 3
            n_frames = stack.shape[0] if is_stack else 1

            all_labels = []
            all_probs = []

            for i in range(n_frames):
                frame = stack[i] if is_stack else stack
                self.sig_status.emit(
                    f"Predicting frame {i+1}/{n_frames}...")

                features = self.extractor.extract(frame)
                h, w, n_feat = features.shape
                features_flat = features.reshape(-1, n_feat)

                labels, probs = self.backend.predict(features_flat)
                all_labels.append(labels.reshape(h, w))
                all_probs.append(probs.reshape(h, w, -1))

                pct = int(100 * (i + 1) / n_frames)
                self.sig_progress.emit(pct)

            if is_stack:
                labels_out = np.stack(all_labels, axis=0)
                probs_out = np.stack(all_probs, axis=0)
            else:
                labels_out = all_labels[0]
                probs_out = all_probs[0]

            self.sig_status.emit("Prediction complete.")
            self.sig_result.emit(labels_out, probs_out)
        except Exception as e:
            logger.exception("Classifier prediction failed")
            self.sig_error.emit(str(e))


class FeatureExtractionWorker(QtCore.QThread):
    """Background thread for feature extraction."""

    sig_result = Signal(object)   # feature array (H, W, N)
    sig_status = Signal(str)
    sig_error = Signal(str)

    def __init__(self, extractor, image, parent=None):
        super().__init__(parent)
        self.extractor = extractor
        self.image = image

    def run(self):
        try:
            self.sig_status.emit("Extracting features...")
            features = self.extractor.extract(self.image)
            self.sig_status.emit(f"Extracted {features.shape[-1]} features.")
            self.sig_result.emit(features)
        except Exception as e:
            logger.exception("Feature extraction failed")
            self.sig_error.emit(str(e))


# ---------------------------------------------------------------------------
# Default class colors
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    (255, 0, 0),      # red
    (0, 0, 255),      # blue
    (0, 200, 0),      # green
    (255, 165, 0),    # orange
    (148, 0, 211),    # violet
    (0, 200, 200),    # cyan
    (255, 192, 203),  # pink
    (128, 128, 0),    # olive
]


# ---------------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------------

class PixelClassifierDialog(QtWidgets.QDialog):
    """Self-contained dialog for interactive pixel classification.

    Supports both 2D images and 3D stacks.  When a stack is loaded,
    labels can be painted on any frame — navigate with the ImageView
    timeline.  Training pools labeled pixels from all labeled frames.

    Separate window selectors for training (labeling) and prediction
    allow classifying a different image or a sub-range of frames.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Classifier")
        self.setMinimumSize(900, 750)

        from .features import FeatureConfig
        self._feature_config = FeatureConfig()
        self._train_worker = None
        self._predict_worker = None
        self._feature_worker = None
        self._features = None          # (H, W, N) for current frame only (viewer)
        self._train_losses = []
        self._backend = None
        self._is_stack = False
        self._image_data = None        # full image array from train window
        self._class_list = [
            {'name': 'Foreground', 'color': _DEFAULT_COLORS[0]},
            {'name': 'Background', 'color': _DEFAULT_COLORS[1]},
        ]

        self._build_ui()
        self._connect_signals()
        self._refresh_window_combos()
        self._update_class_list_widget()

    # --- UI Construction ---

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left: ImageView with paint overlay
        left = QtWidgets.QVBoxLayout()

        # Train image selector row
        train_row = QtWidgets.QHBoxLayout()
        train_row.addWidget(QtWidgets.QLabel("Train Image:"))
        self.train_window_combo = QtWidgets.QComboBox()
        train_row.addWidget(self.train_window_combo, stretch=1)

        train_row.addWidget(QtWidgets.QLabel("Backend:"))
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(['Random Forest', 'CNN'])
        train_row.addWidget(self.backend_combo)

        train_row.addWidget(QtWidgets.QLabel("Device:"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        train_row.addWidget(self.device_combo)
        left.addLayout(train_row)

        # Frame info label (shown for stacks)
        self.frame_label = QtWidgets.QLabel("")
        left.addWidget(self.frame_label)

        # Embedded ImageView
        self.image_view = pg.ImageView()
        self.image_view.setMinimumSize(400, 300)
        self.paint_overlay = None
        left.addWidget(self.image_view)

        main_layout.addLayout(left, stretch=2)

        # Right panel
        right = QtWidgets.QVBoxLayout()

        # Class list
        class_group = QtWidgets.QGroupBox("Classes")
        class_layout = QtWidgets.QVBoxLayout(class_group)
        self.class_list_widget = QtWidgets.QListWidget()
        self.class_list_widget.setMaximumHeight(150)
        class_layout.addWidget(self.class_list_widget)

        class_btn_row = QtWidgets.QHBoxLayout()
        self.add_class_btn = QtWidgets.QPushButton("+ Add")
        self.remove_class_btn = QtWidgets.QPushButton("- Remove")
        class_btn_row.addWidget(self.add_class_btn)
        class_btn_row.addWidget(self.remove_class_btn)
        class_layout.addLayout(class_btn_row)
        right.addWidget(class_group)

        # Brush controls
        brush_group = QtWidgets.QGroupBox("Brush")
        brush_layout = QtWidgets.QGridLayout(brush_group)

        brush_layout.addWidget(QtWidgets.QLabel("Size:"), 0, 0)
        self.brush_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_size_slider.setRange(1, 30)
        self.brush_size_slider.setValue(5)
        self.brush_size_label = QtWidgets.QLabel("5")
        brush_layout.addWidget(self.brush_size_slider, 0, 1)
        brush_layout.addWidget(self.brush_size_label, 0, 2)

        brush_layout.addWidget(QtWidgets.QLabel("Opacity:"), 1, 0)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(50)
        self.opacity_label = QtWidgets.QLabel("0.50")
        brush_layout.addWidget(self.opacity_slider, 1, 1)
        brush_layout.addWidget(self.opacity_label, 1, 2)

        right.addWidget(brush_group)

        # Feature buttons
        feat_btn_row = QtWidgets.QHBoxLayout()
        self.feature_options_btn = QtWidgets.QPushButton("Feature Options...")
        self.view_features_btn = QtWidgets.QPushButton("View Features")
        feat_btn_row.addWidget(self.feature_options_btn)
        feat_btn_row.addWidget(self.view_features_btn)
        right.addLayout(feat_btn_row)

        # CNN parameters
        self.cnn_group = QtWidgets.QGroupBox("CNN Parameters")
        cnn_layout = QtWidgets.QGridLayout(self.cnn_group)

        cnn_layout.addWidget(QtWidgets.QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        cnn_layout.addWidget(self.epochs_spin, 0, 1)

        cnn_layout.addWidget(QtWidgets.QLabel("Batch:"), 0, 2)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        cnn_layout.addWidget(self.batch_spin, 0, 3)

        cnn_layout.addWidget(QtWidgets.QLabel("LR:"), 1, 0)
        self.lr_edit = QtWidgets.QLineEdit("0.001")
        cnn_layout.addWidget(self.lr_edit, 1, 1)

        self.cnn_group.setVisible(False)
        right.addWidget(self.cnn_group)

        # --- Prediction target ---
        predict_group = QtWidgets.QGroupBox("Prediction")
        predict_layout = QtWidgets.QGridLayout(predict_group)

        predict_layout.addWidget(QtWidgets.QLabel("Predict Image:"), 0, 0)
        self.predict_window_combo = QtWidgets.QComboBox()
        predict_layout.addWidget(self.predict_window_combo, 0, 1, 1, 3)

        self.all_frames_check = QtWidgets.QCheckBox("All frames")
        self.all_frames_check.setChecked(True)
        predict_layout.addWidget(self.all_frames_check, 1, 0)

        predict_layout.addWidget(QtWidgets.QLabel("Start:"), 1, 1)
        self.frame_start_spin = QtWidgets.QSpinBox()
        self.frame_start_spin.setRange(0, 0)
        self.frame_start_spin.setEnabled(False)
        predict_layout.addWidget(self.frame_start_spin, 1, 2)

        predict_layout.addWidget(QtWidgets.QLabel("End:"), 1, 3)
        self.frame_end_spin = QtWidgets.QSpinBox()
        self.frame_end_spin.setRange(0, 0)
        self.frame_end_spin.setEnabled(False)
        predict_layout.addWidget(self.frame_end_spin, 1, 4)

        right.addWidget(predict_group)

        # Loss curve
        self.loss_plot = pg.PlotWidget(title="Loss")
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_curve = self.loss_plot.plot(pen='y', name='Train')
        self.loss_plot.setMinimumHeight(120)
        self.loss_plot.setMaximumHeight(200)
        right.addWidget(self.loss_plot)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        right.addWidget(self.progress_bar)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        right.addWidget(self.status_label)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Train")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.predict_btn = QtWidgets.QPushButton("Predict")
        self.export_btn = QtWidgets.QPushButton("Export Model")
        self.close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.predict_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.close_btn)
        right.addLayout(btn_row)

        right.addStretch()
        main_layout.addLayout(right, stretch=1)

    def _connect_signals(self):
        self.train_window_combo.currentIndexChanged.connect(self._on_train_window_changed)
        self.predict_window_combo.currentIndexChanged.connect(self._on_predict_window_changed)
        self.all_frames_check.toggled.connect(self._on_all_frames_toggled)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.class_list_widget.currentRowChanged.connect(self._on_class_selected)
        self.add_class_btn.clicked.connect(self._add_class)
        self.remove_class_btn.clicked.connect(self._remove_class)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.feature_options_btn.clicked.connect(self._open_feature_options)
        self.view_features_btn.clicked.connect(self._view_features)
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.predict_btn.clicked.connect(self._start_prediction)
        self.export_btn.clicked.connect(self._export_model)
        self.close_btn.clicked.connect(self.close)

    # --- Window management ---

    def _refresh_window_combos(self):
        for combo in (self.train_window_combo, self.predict_window_combo):
            combo.clear()
            for w in g.windows:
                name = getattr(w, 'name', str(w))
                combo.addItem(name)

    @staticmethod
    def _image_from_combo(combo):
        """Return (H, W) or (T, H, W) image array for the window selected
        in *combo*, or None."""
        idx = combo.currentIndex()
        if idx < 0 or idx >= len(g.windows):
            return None
        img = g.windows[idx].image
        if img is None:
            return None
        if img.ndim <= 3:
            return img
        if img.ndim == 4:
            return img[:, 0]
        return img[0]

    def _on_train_window_changed(self, idx):
        """Load the training window's image into the embedded ImageView."""
        img = self._image_from_combo(self.train_window_combo)
        if img is None:
            return

        self._image_data = img
        self._is_stack = img.ndim == 3
        self._features = None

        self.image_view.setImage(img)

        n_frames = img.shape[0] if self._is_stack else 1
        frame_shape = img.shape[1:] if self._is_stack else img.shape

        # Create/reset paint overlay with per-frame masks
        colors = {i + 1: cls['color'] for i, cls in enumerate(self._class_list)}
        if self.paint_overlay is not None:
            self.image_view.getView().removeItem(self.paint_overlay)

        self.paint_overlay = PaintOverlay(
            frame_shape, n_frames=n_frames, class_colors=colors)
        self.image_view.getView().addItem(self.paint_overlay)

        if self._is_stack:
            self.image_view.timeLine.sigPositionChanged.connect(
                self._on_frame_changed)
            self.frame_label.setText(
                f"Stack: {n_frames} frames  |  Labels on 0 frame(s)")
            self.frame_label.show()
        else:
            self.frame_label.hide()

        self.status_label.setText(
            "Image loaded. Paint labels on any frame, then Train.")

    def _on_predict_window_changed(self, idx):
        """Update frame-range spinboxes to match the prediction window."""
        img = self._image_from_combo(self.predict_window_combo)
        if img is None:
            return
        if img.ndim == 3:
            max_frame = img.shape[0] - 1
        else:
            max_frame = 0
        self.frame_start_spin.setRange(0, max_frame)
        self.frame_end_spin.setRange(0, max_frame)
        self.frame_end_spin.setValue(max_frame)

    def _on_all_frames_toggled(self, checked):
        self.frame_start_spin.setEnabled(not checked)
        self.frame_end_spin.setEnabled(not checked)

    def _on_frame_changed(self):
        """Sync paint overlay to the current ImageView frame."""
        if self.paint_overlay is None or not self._is_stack:
            return
        idx, _t = self.image_view.timeIndex(self.image_view.timeLine)
        self.paint_overlay.current_frame = int(_t)
        self._update_frame_label()

    def _update_frame_label(self):
        """Update the frame info label."""
        if not self._is_stack or self.paint_overlay is None:
            return
        n_frames = self.paint_overlay.n_frames
        n_labeled = self._count_labeled_frames()
        cur = self.paint_overlay.current_frame
        self.frame_label.setText(
            f"Frame {cur}/{n_frames - 1}  |  "
            f"Labels on {n_labeled} frame(s)")

    def _count_labeled_frames(self):
        """Count how many frames have at least one labeled pixel."""
        if self.paint_overlay is None:
            return 0
        masks = self.paint_overlay.label_masks
        return int((masks.reshape(masks.shape[0], -1).any(axis=1)).sum())

    def _on_backend_changed(self, idx):
        is_cnn = idx == 1
        self.cnn_group.setVisible(is_cnn)

    # --- Class management ---

    def _update_class_list_widget(self):
        self.class_list_widget.clear()
        for i, cls in enumerate(self._class_list):
            item = QtWidgets.QListWidgetItem(f"  {cls['name']}")
            color = QtGui.QColor(*cls['color'])
            item.setForeground(color)
            item.setIcon(_color_icon(cls['color']))
            self.class_list_widget.addItem(item)
        if self.class_list_widget.count() > 0:
            self.class_list_widget.setCurrentRow(0)

    def _on_class_selected(self, row):
        if self.paint_overlay is not None and row >= 0:
            self.paint_overlay.active_class = row + 1

    def _add_class(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Class", "Class name:")
        if ok and name:
            idx = len(self._class_list)
            color = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
            self._class_list.append({'name': name, 'color': color})
            self._update_class_list_widget()
            self._sync_overlay_colors()

    def _remove_class(self):
        row = self.class_list_widget.currentRow()
        if row >= 0 and len(self._class_list) > 2:
            self._class_list.pop(row)
            self._update_class_list_widget()
            self._sync_overlay_colors()
            # Clear removed class from all frame masks
            if self.paint_overlay is not None:
                masks = self.paint_overlay.label_masks
                masks[masks == row + 1] = 0
                for c in range(row + 2, len(self._class_list) + 2):
                    masks[masks == c] = c - 1
                self.paint_overlay._update_display()

    def _sync_overlay_colors(self):
        if self.paint_overlay is not None:
            self.paint_overlay.class_colors = {
                i + 1: cls['color'] for i, cls in enumerate(self._class_list)
            }
            self.paint_overlay._update_display()

    # --- Brush controls ---

    def _on_brush_size_changed(self, val):
        self.brush_size_label.setText(str(val))
        if self.paint_overlay is not None:
            self.paint_overlay.brush_size = val

    def _on_opacity_changed(self, val):
        opacity = val / 100.0
        self.opacity_label.setText(f"{opacity:.2f}")
        if self.paint_overlay is not None:
            self.paint_overlay.opacity_val = opacity
            self.paint_overlay._update_display()

    # --- Feature options ---

    def _open_feature_options(self):
        """Open the feature options dialog."""
        dlg = FeatureOptionsDialog(self._feature_config, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._feature_config = dlg.get_config()
            from .features import FeatureExtractor
            n = FeatureExtractor(self._feature_config).n_features()
            self.status_label.setText(f"Feature config updated — {n} features.")

    def _make_extractor(self):
        """Create a FeatureExtractor with the current config."""
        from .features import FeatureExtractor
        return FeatureExtractor(self._feature_config)

    # --- Feature extraction ---

    def _get_current_frame_image(self):
        """Return the 2D image for the currently displayed frame."""
        if self._image_data is None:
            return None
        if self._is_stack:
            idx = self.paint_overlay.current_frame if self.paint_overlay else 0
            return self._image_data[idx]
        return self._image_data

    def _view_features(self):
        """Show feature viewer for the current frame."""
        frame_img = self._get_current_frame_image()
        if frame_img is None:
            return
        self.status_label.setText("Extracting features for current frame...")
        QtWidgets.QApplication.processEvents()
        extractor = self._make_extractor()
        features = extractor.extract(frame_img)
        names = extractor.feature_names()
        self.status_label.setText(f"Extracted {features.shape[-1]} features.")
        dlg = FeatureViewerDialog(features, names, parent=self)
        dlg.show()

    # --- Training ---

    def _build_config(self):
        from .classifier_backends import ClassifierConfig
        backend_name = 'cnn' if self.backend_combo.currentIndex() == 1 else 'random_forest'
        try:
            lr = float(self.lr_edit.text())
        except ValueError:
            lr = 1e-3
        return ClassifierConfig(
            backend=backend_name,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=lr,
            device=self.device_combo.currentText(),
        )

    def _start_training(self):
        if self.paint_overlay is None:
            g.alert("No image loaded. Select a window first.")
            return

        masks = self.paint_overlay.label_masks  # (T, H, W)
        total_labeled = (masks > 0).sum()
        if total_labeled < 10:
            g.alert(f"Need at least 10 labeled pixels (have {total_labeled}). "
                    f"Paint more labels.")
            return

        # Find which frames have labels and extract features + labels
        extractor = self._make_extractor()
        n_frames = self.paint_overlay.n_frames

        self.status_label.setText("Extracting features from labeled frames...")
        QtWidgets.QApplication.processEvents()

        all_features = []
        all_labels = []

        for t in range(n_frames):
            frame_mask = masks[t]
            labeled = frame_mask > 0
            if not labeled.any():
                continue

            # Get the image frame
            if self._is_stack:
                frame_img = self._image_data[t]
            else:
                frame_img = self._image_data

            features = extractor.extract(frame_img)  # (H, W, N)
            all_features.append(features[labeled])   # (K, N)
            all_labels.append(frame_mask[labeled])    # (K,)

        features_flat = np.concatenate(all_features, axis=0)  # (total_K, N)
        labels_flat = np.concatenate(all_labels, axis=0)       # (total_K,)

        n_labeled_frames = len(all_features)
        self.status_label.setText(
            f"Extracted features from {n_labeled_frames} frame(s), "
            f"{len(features_flat)} labeled pixels. Training...")

        config = self._build_config()

        # Build backend
        from .classifier_backends import create_backend
        self._backend = create_backend(config)
        n_classes = len(self._class_list)

        # Reset loss curve
        self._train_losses.clear()
        self.loss_curve.setData([], [])
        self.progress_bar.setValue(0)

        self._train_worker = ClassifierTrainingWorker(
            self._backend, features_flat, labels_flat, n_classes, config,
            parent=self)
        self._train_worker.sig_epoch_done.connect(self._on_epoch_done)
        self._train_worker.sig_progress.connect(self.progress_bar.setValue)
        self._train_worker.sig_status.connect(self.status_label.setText)
        self._train_worker.sig_finished.connect(self._on_train_finished)
        self._train_worker.sig_error.connect(self._on_error)
        self._train_worker.finished.connect(self._on_worker_done)

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._train_worker.start()

    def _stop_training(self):
        if self._train_worker is not None:
            self._train_worker.request_stop()
            self.status_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)

    def _on_epoch_done(self, epoch, loss):
        self._train_losses.append(loss)
        epochs = list(range(1, len(self._train_losses) + 1))
        self.loss_curve.setData(epochs, self._train_losses)

    def _on_train_finished(self):
        self.status_label.setText("Training complete. Click Predict.")
        self._update_frame_label()

    def _on_worker_done(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        g.alert(msg)

    # --- Prediction ---

    def _start_prediction(self):
        if self._backend is None or not self._backend.is_trained():
            g.alert("Train the classifier first.")
            return

        pred_img = self._image_from_combo(self.predict_window_combo)
        if pred_img is None:
            g.alert("No prediction image selected.")
            return

        # Slice to the requested frame range
        if pred_img.ndim == 3:
            if self.all_frames_check.isChecked():
                start, end = 0, pred_img.shape[0] - 1
            else:
                start = self.frame_start_spin.value()
                end = self.frame_end_spin.value()
            if start > end:
                start, end = end, start
            pred_img = pred_img[start:end + 1]
            self._predict_frame_offset = start
        else:
            self._predict_frame_offset = 0

        extractor = self._make_extractor()

        self._predict_worker = ClassifierPredictionWorker(
            self._backend, extractor, pred_img, parent=self)
        self._predict_worker.sig_result.connect(self._on_predict_result)
        self._predict_worker.sig_progress.connect(self.progress_bar.setValue)
        self._predict_worker.sig_status.connect(self.status_label.setText)
        self._predict_worker.sig_error.connect(self._on_error)
        self._predict_worker.finished.connect(
            lambda: self.predict_btn.setEnabled(True))

        self.predict_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._predict_worker.start()

    def _on_predict_result(self, labels, probs):
        from ..window import Window
        label_map = np.asarray(labels, dtype=g.settings['internal_data_type'])
        src_name = self.predict_window_combo.currentText()

        # Include frame range in the output name if it was a sub-range
        offset = getattr(self, '_predict_frame_offset', 0)
        if label_map.ndim == 3:
            end = offset + label_map.shape[0] - 1
            if offset > 0 or end < label_map.shape[0] - 1:
                suffix = f' - Classified [{offset}-{end}]'
            else:
                suffix = ' - Classified'
        else:
            suffix = ' - Classified'

        win = Window(label_map, name=src_name + suffix)
        win.metadata['class_probabilities'] = np.asarray(probs)
        n_frames = label_map.shape[0] if label_map.ndim == 3 else 1
        self.status_label.setText(
            f"Prediction complete — {n_frames} frame(s) classified.")

    # --- Export ---

    def _export_model(self):
        if self._backend is None or not self._backend.is_trained():
            g.alert("Train the classifier first.")
            return

        if isinstance(self._backend, type) and self._backend.name() == 'CNN':
            ext = "PyTorch Model (*.pt)"
        else:
            ext = "Joblib Model (*.joblib);;All Files (*)"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Model", "", ext)
        if path:
            self._backend.save(path)
            self.status_label.setText(f"Model exported to {path}")

    # --- Cleanup ---

    def closeEvent(self, event):
        for worker in (self._train_worker, self._predict_worker, self._feature_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, 'request_stop'):
                    worker.request_stop()
                worker.wait(5000)
        super().closeEvent(event)


def _color_icon(rgb, size=16):
    """Create a small colored QIcon."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtGui.QColor(*rgb))
    return QtGui.QIcon(pixmap)
