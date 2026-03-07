"""Object Detection dialog for flika.

Three-tab dialog:
  1. **Predict** — Run YOLO detection on the current window, create ROIs
  2. **Annotate** — Draw/edit bounding boxes, import/export annotations
  3. **Train** — Train or fine-tune a YOLO model from annotations

Follows the same dialog pattern as ``classifier_dialog.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Signal
import pyqtgraph as pg

from .. import global_vars as g
from ..logger import logger
from .annotations import AnnotationClass, AnnotationSet, BoundingBox
from .annotation_overlay import BoxAnnotationOverlay, InteractionMode
from .detection_backend import DetectionConfig, UltralyticsBackend, prepare_image_for_detection


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class DetectionPredictWorker(QtCore.QThread):
    """Run detection on one or more frames in a background thread."""
    sig_result = Signal(object)     # List[BoundingBox]
    sig_progress = Signal(int)
    sig_error = Signal(str)

    def __init__(self, backend, images, config, batch=False, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.images = images        # single image or list of frames
        self.config = config
        self.batch = batch
        self._stop = False

    def run(self):
        try:
            if self.batch and isinstance(self.images, np.ndarray) and self.images.ndim >= 3:
                all_boxes = []
                n = self.images.shape[0]
                for i in range(n):
                    if self._stop:
                        break
                    boxes = self.backend.predict(self.images[i], self.config)
                    for b in boxes:
                        b.frame = i
                    all_boxes.extend(boxes)
                    self.sig_progress.emit(int((i + 1) / n * 100))
                self.sig_result.emit(all_boxes)
            else:
                boxes = self.backend.predict(self.images, self.config)
                self.sig_progress.emit(100)
                self.sig_result.emit(boxes)
        except Exception as e:
            self.sig_error.emit(str(e))

    def stop(self):
        self._stop = True


class DetectionTrainWorker(QtCore.QThread):
    """Train or fine-tune a YOLO model in a background thread."""
    sig_epoch_done = Signal(int, float, float, float)  # epoch, box_loss, cls_loss, dfl_loss
    sig_progress = Signal(int)
    sig_finished = Signal(str)      # saved model path
    sig_error = Signal(str)

    def __init__(self, backend, data_yaml, config, fine_tune=False, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.data_yaml = data_yaml
        self.config = config
        self.fine_tune = fine_tune

    def run(self):
        try:
            if self.fine_tune:
                path = self.backend.fine_tune(self.data_yaml, self.config)
            else:
                path = self.backend.train(self.data_yaml, self.config)
            self.sig_finished.emit(path)
        except Exception as e:
            self.sig_error.emit(str(e))


# ---------------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128),
    (128, 0, 128), (255, 128, 0),
]


class ObjectDetectionDialog(QtWidgets.QDialog):
    """AI Object Detection dialog with Predict / Annotate / Train tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('AI Object Detection')
        self.resize(900, 700)

        self._backend = UltralyticsBackend()
        self._config = DetectionConfig()
        self._detections: list[BoundingBox] = []
        self._annotation_set: AnnotationSet | None = None
        self._overlay: BoxAnnotationOverlay | None = None
        self._predict_worker: DetectionPredictWorker | None = None
        self._train_worker: DetectionTrainWorker | None = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_predict_tab(), 'Predict')
        self.tabs.addTab(self._build_annotate_tab(), 'Annotate')
        self.tabs.addTab(self._build_train_tab(), 'Train')

    # -- Predict tab ---------------------------------------------------------

    def _build_predict_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Window selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Window:'))
        self._win_combo = QtWidgets.QComboBox()
        self._win_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Fixed)
        row.addWidget(self._win_combo)
        btn_refresh = QtWidgets.QPushButton('Refresh')
        btn_refresh.clicked.connect(self._refresh_windows)
        row.addWidget(btn_refresh)
        layout.addLayout(row)

        # Model selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Model:'))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Fixed)
        row.addWidget(self._model_combo)
        btn_browse = QtWidgets.QPushButton('Browse...')
        btn_browse.clicked.connect(self._browse_model)
        row.addWidget(btn_browse)
        layout.addLayout(row)

        # Parameters
        form = QtWidgets.QFormLayout()

        self._conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._conf_slider.setRange(1, 100)
        self._conf_slider.setValue(25)
        self._conf_label = QtWidgets.QLabel('0.25')
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_label.setText(f'{v / 100:.2f}'))
        conf_row = QtWidgets.QHBoxLayout()
        conf_row.addWidget(self._conf_slider)
        conf_row.addWidget(self._conf_label)
        form.addRow('Confidence:', conf_row)

        self._iou_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._iou_slider.setRange(1, 100)
        self._iou_slider.setValue(45)
        self._iou_label = QtWidgets.QLabel('0.45')
        self._iou_slider.valueChanged.connect(
            lambda v: self._iou_label.setText(f'{v / 100:.2f}'))
        iou_row = QtWidgets.QHBoxLayout()
        iou_row.addWidget(self._iou_slider)
        iou_row.addWidget(self._iou_label)
        form.addRow('IoU Threshold:', iou_row)

        self._imgsz_combo = QtWidgets.QComboBox()
        self._imgsz_combo.addItems(['320', '416', '640', '1280'])
        self._imgsz_combo.setCurrentText('640')
        form.addRow('Image Size:', self._imgsz_combo)

        self._device_combo = QtWidgets.QComboBox()
        self._device_combo.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        form.addRow('Device:', self._device_combo)

        self._tiling_check = QtWidgets.QCheckBox('Enable tiling for large images')
        form.addRow('Tiling:', self._tiling_check)

        self._batch_check = QtWidgets.QCheckBox('Batch (all frames)')
        form.addRow('Stack:', self._batch_check)

        layout.addLayout(form)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_detect = QtWidgets.QPushButton('Run Detection')
        self._btn_detect.clicked.connect(self._run_detection)
        btn_row.addWidget(self._btn_detect)
        self._btn_create_rois = QtWidgets.QPushButton('Create ROIs')
        self._btn_create_rois.clicked.connect(self._create_rois)
        self._btn_create_rois.setEnabled(False)
        btn_row.addWidget(self._btn_create_rois)
        self._btn_clear = QtWidgets.QPushButton('Clear Results')
        self._btn_clear.clicked.connect(self._clear_detections)
        self._btn_clear.setEnabled(False)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

        # Progress
        self._predict_progress = QtWidgets.QProgressBar()
        self._predict_progress.setVisible(False)
        layout.addWidget(self._predict_progress)

        # Results
        self._results_label = QtWidgets.QLabel('')
        layout.addWidget(self._results_label)

        layout.addStretch()

        self._refresh_windows()
        self._refresh_models()
        return tab

    # -- Annotate tab --------------------------------------------------------

    def _build_annotate_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        # Left: image view
        self._ann_view = pg.ImageView()
        self._ann_view.ui.roiBtn.hide()
        self._ann_view.ui.menuBtn.hide()
        layout.addWidget(self._ann_view, stretch=3)

        # Right: controls
        panel = QtWidgets.QVBoxLayout()

        # Load from window
        btn_load = QtWidgets.QPushButton('Load from Window')
        btn_load.clicked.connect(self._ann_load_from_window)
        panel.addWidget(btn_load)

        # Tool selector
        tool_group = QtWidgets.QGroupBox('Tool')
        tool_layout = QtWidgets.QHBoxLayout(tool_group)
        self._tool_draw = QtWidgets.QRadioButton('Draw')
        self._tool_select = QtWidgets.QRadioButton('Select')
        self._tool_delete = QtWidgets.QRadioButton('Delete')
        self._tool_draw.setChecked(True)
        self._tool_draw.toggled.connect(lambda c: c and self._set_tool(InteractionMode.DRAW))
        self._tool_select.toggled.connect(lambda c: c and self._set_tool(InteractionMode.SELECT))
        self._tool_delete.toggled.connect(lambda c: c and self._set_tool(InteractionMode.DELETE))
        tool_layout.addWidget(self._tool_draw)
        tool_layout.addWidget(self._tool_select)
        tool_layout.addWidget(self._tool_delete)
        panel.addWidget(tool_group)

        # Class management
        class_group = QtWidgets.QGroupBox('Classes')
        class_layout = QtWidgets.QVBoxLayout(class_group)
        self._class_list = QtWidgets.QListWidget()
        self._class_list.currentRowChanged.connect(self._ann_class_selected)
        class_layout.addWidget(self._class_list)

        class_btn_row = QtWidgets.QHBoxLayout()
        btn_add_cls = QtWidgets.QPushButton('+')
        btn_add_cls.setFixedWidth(30)
        btn_add_cls.clicked.connect(self._ann_add_class)
        btn_rm_cls = QtWidgets.QPushButton('-')
        btn_rm_cls.setFixedWidth(30)
        btn_rm_cls.clicked.connect(self._ann_remove_class)
        btn_rename_cls = QtWidgets.QPushButton('Rename')
        btn_rename_cls.clicked.connect(self._ann_rename_class)
        class_btn_row.addWidget(btn_add_cls)
        class_btn_row.addWidget(btn_rm_cls)
        class_btn_row.addWidget(btn_rename_cls)
        class_layout.addLayout(class_btn_row)
        panel.addWidget(class_group)

        # Stats
        self._ann_stats = QtWidgets.QLabel('')
        panel.addWidget(self._ann_stats)

        # Import / Export
        io_group = QtWidgets.QGroupBox('Import / Export')
        io_layout = QtWidgets.QVBoxLayout(io_group)
        btn_import_rois = QtWidgets.QPushButton('Import ROIs')
        btn_import_rois.clicked.connect(self._ann_import_rois)
        io_layout.addWidget(btn_import_rois)
        btn_import = QtWidgets.QPushButton('Import...')
        btn_import.clicked.connect(self._ann_import)
        io_layout.addWidget(btn_import)
        btn_export = QtWidgets.QPushButton('Export...')
        btn_export.clicked.connect(self._ann_export)
        io_layout.addWidget(btn_export)
        panel.addWidget(io_group)

        panel.addStretch()
        layout.addLayout(panel, stretch=1)
        return tab

    # -- Train tab -----------------------------------------------------------

    def _build_train_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Use a splitter: top = config, bottom = saved models
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # --- Top: Training configuration ---
        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Mode
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel('Mode:'))
        self._train_mode = QtWidgets.QComboBox()
        self._train_mode.addItems(['Train from Scratch', 'Fine-tune Pretrained'])
        mode_row.addWidget(self._train_mode)
        top_layout.addLayout(mode_row)

        # Data source
        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(QtWidgets.QLabel('Data:'))
        self._data_source = QtWidgets.QComboBox()
        self._data_source.addItems(['From Annotate Tab', 'External YOLO Dataset'])
        data_row.addWidget(self._data_source)
        self._data_dir_btn = QtWidgets.QPushButton('Browse...')
        self._data_dir_btn.clicked.connect(self._browse_data_dir)
        data_row.addWidget(self._data_dir_btn)
        top_layout.addLayout(data_row)

        self._data_yaml_path = QtWidgets.QLineEdit()
        self._data_yaml_path.setPlaceholderText('Path to data.yaml (for external dataset)')
        top_layout.addWidget(self._data_yaml_path)

        # For fine-tune: base model
        ft_row = QtWidgets.QHBoxLayout()
        ft_row.addWidget(QtWidgets.QLabel('Base Model:'))
        self._ft_model_combo = QtWidgets.QComboBox()
        ft_row.addWidget(self._ft_model_combo)
        top_layout.addLayout(ft_row)

        # Training params
        form = QtWidgets.QFormLayout()

        self._train_size = QtWidgets.QComboBox()
        self._train_size.addItems(['n', 's', 'm', 'l', 'x'])
        form.addRow('Model Size:', self._train_size)

        self._train_epochs = QtWidgets.QSpinBox()
        self._train_epochs.setRange(1, 10000)
        self._train_epochs.setValue(100)
        form.addRow('Epochs:', self._train_epochs)

        self._train_batch = QtWidgets.QSpinBox()
        self._train_batch.setRange(1, 256)
        self._train_batch.setValue(16)
        form.addRow('Batch Size:', self._train_batch)

        self._train_lr = QtWidgets.QDoubleSpinBox()
        self._train_lr.setRange(0.0001, 1.0)
        self._train_lr.setDecimals(4)
        self._train_lr.setSingleStep(0.001)
        self._train_lr.setValue(0.01)
        form.addRow('Learning Rate:', self._train_lr)

        self._train_freeze = QtWidgets.QSpinBox()
        self._train_freeze.setRange(0, 50)
        self._train_freeze.setValue(0)
        form.addRow('Freeze Layers:', self._train_freeze)

        self._train_augment = QtWidgets.QCheckBox()
        self._train_augment.setChecked(True)
        form.addRow('Augmentation:', self._train_augment)

        self._train_name = QtWidgets.QLineEdit('flika_detector')
        form.addRow('Model Name:', self._train_name)

        self._train_device = QtWidgets.QComboBox()
        self._train_device.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        form.addRow('Device:', self._train_device)

        top_layout.addLayout(form)

        # Save location info
        from .detection_backend import _models_dir
        save_info = QtWidgets.QLabel(
            f'<small>Models are saved to: <code>{_models_dir()}</code></small>')
        save_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        save_info.setWordWrap(True)
        top_layout.addWidget(save_info)

        # Progress + buttons
        self._train_progress = QtWidgets.QProgressBar()
        self._train_progress.setVisible(False)
        top_layout.addWidget(self._train_progress)

        self._train_status = QtWidgets.QLabel('')
        self._train_status.setWordWrap(True)
        top_layout.addWidget(self._train_status)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_train = QtWidgets.QPushButton('Train')
        self._btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self._btn_train)
        self._btn_stop_train = QtWidgets.QPushButton('Stop')
        self._btn_stop_train.setEnabled(False)
        btn_row.addWidget(self._btn_stop_train)
        layout.addLayout(btn_row)

        # Loss plot
        self._loss_plot = pg.PlotWidget(title='Training Loss')
        self._loss_plot.setLabel('bottom', 'Epoch')
        self._loss_plot.setLabel('left', 'Loss')
        self._loss_plot.addLegend()
        self._box_loss_curve = self._loss_plot.plot(pen='r', name='box_loss')
        self._cls_loss_curve = self._loss_plot.plot(pen='g', name='cls_loss')
        self._dfl_loss_curve = self._loss_plot.plot(pen='b', name='dfl_loss')
        self._loss_data = {'box': [], 'cls': [], 'dfl': []}
        top_layout.addWidget(self._loss_plot)

        splitter.addWidget(top_widget)

        # --- Bottom: Saved Models ---
        bottom_widget = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(QtWidgets.QLabel(
            '<b>Saved Models</b>'))
        header_row.addStretch()
        btn_open_folder = QtWidgets.QPushButton('Open Models Folder')
        btn_open_folder.clicked.connect(self._open_models_folder)
        header_row.addWidget(btn_open_folder)
        btn_refresh_saved = QtWidgets.QPushButton('Refresh')
        btn_refresh_saved.clicked.connect(self._refresh_saved_models)
        header_row.addWidget(btn_refresh_saved)
        bottom_layout.addLayout(header_row)

        self._saved_models_list = QtWidgets.QListWidget()
        bottom_layout.addWidget(self._saved_models_list)

        model_btn_row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton('Save Copy As...')
        btn_copy.setToolTip('Save a copy of the selected model to a custom location')
        btn_copy.clicked.connect(self._save_model_copy)
        model_btn_row.addWidget(btn_copy)
        btn_load_ext = QtWidgets.QPushButton('Import Model...')
        btn_load_ext.setToolTip('Import a .pt model file into the flika models folder')
        btn_load_ext.clicked.connect(self._import_model)
        model_btn_row.addWidget(btn_load_ext)
        btn_delete = QtWidgets.QPushButton('Delete')
        btn_delete.setToolTip('Delete the selected model')
        btn_delete.clicked.connect(self._delete_saved_model)
        model_btn_row.addWidget(btn_delete)
        bottom_layout.addLayout(model_btn_row)

        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self._refresh_models()
        self._refresh_saved_models()
        return tab

    # -----------------------------------------------------------------------
    # Predict tab logic
    # -----------------------------------------------------------------------

    def _refresh_windows(self):
        self._win_combo.clear()
        for w in g.windows:
            name = getattr(w, 'name', str(w))
            self._win_combo.addItem(name, w)

    def _refresh_models(self):
        try:
            models = self._backend.available_models()
        except Exception:
            models = []

        self._model_combo.clear()
        if hasattr(self, '_ft_model_combo'):
            self._ft_model_combo.clear()
        for m in models:
            self._model_combo.addItem(f"{m['name']} — {m['description']}", m['path'])
            if hasattr(self, '_ft_model_combo'):
                self._ft_model_combo.addItem(f"{m['name']}", m['path'])

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Model', '', 'PyTorch Models (*.pt);;All Files (*)')
        if path:
            self._model_combo.addItem(os.path.basename(path), path)
            self._model_combo.setCurrentIndex(self._model_combo.count() - 1)

    def _get_selected_window(self):
        idx = self._win_combo.currentIndex()
        if idx < 0:
            return None
        return self._win_combo.itemData(idx)

    def _build_config(self) -> DetectionConfig:
        model_path = self._model_combo.currentData() or ''
        return DetectionConfig(
            model_path=model_path,
            confidence=self._conf_slider.value() / 100.0,
            iou_threshold=self._iou_slider.value() / 100.0,
            image_size=int(self._imgsz_combo.currentText()),
            device=self._device_combo.currentText(),
            use_tiling=self._tiling_check.isChecked(),
        )

    def _run_detection(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            g.alert("ultralytics is not installed.\n\n"
                    "Install with:  pip install ultralytics")
            return

        win = self._get_selected_window()
        if win is None:
            g.alert('No window selected.')
            return

        config = self._build_config()
        image = win.image
        batch = self._batch_check.isChecked() and image.ndim >= 3

        self._btn_detect.setEnabled(False)
        self._predict_progress.setVisible(True)
        self._predict_progress.setValue(0)
        self._results_label.setText('Running detection...')

        self._predict_worker = DetectionPredictWorker(
            self._backend, image, config, batch=batch, parent=self)
        self._predict_worker.sig_result.connect(self._on_predict_done)
        self._predict_worker.sig_progress.connect(self._predict_progress.setValue)
        self._predict_worker.sig_error.connect(self._on_predict_error)
        self._predict_worker.start()

    def _on_predict_done(self, boxes: list):
        self._detections = boxes
        self._btn_detect.setEnabled(True)
        self._predict_progress.setVisible(False)
        self._btn_create_rois.setEnabled(bool(boxes))
        self._btn_clear.setEnabled(bool(boxes))

        # Summarize
        class_counts: dict[str, int] = {}
        for b in boxes:
            class_counts[b.class_name] = class_counts.get(b.class_name, 0) + 1
        parts = [f'{v} {k}' for k, v in sorted(class_counts.items())]
        summary = f"Found {len(boxes)} objects"
        if parts:
            summary += f" ({', '.join(parts)})"
        self._results_label.setText(summary)

        # Store in window metadata
        win = self._get_selected_window()
        if win is not None:
            win.metadata['detections'] = [
                {'class': b.class_name, 'confidence': b.confidence,
                 'bbox': [b.x, b.y, b.width, b.height], 'frame': b.frame}
                for b in boxes
            ]

    def _on_predict_error(self, msg: str):
        self._btn_detect.setEnabled(True)
        self._predict_progress.setVisible(False)
        self._results_label.setText(f'Error: {msg}')
        logger.error("Detection error: %s", msg)

    def _create_rois(self):
        win = self._get_selected_window()
        if win is None or not self._detections:
            return

        from ..roi import makeROI

        for b in self._detections:
            # Determine color from YOLO class
            color = QtGui.QColor(*_DEFAULT_COLORS[b.class_id % len(_DEFAULT_COLORS)])
            roi = makeROI('rectangle',
                          [[b.y, b.x], [b.height, b.width]],
                          window=win, color=color)
            if roi is not None:
                roi.name = f"{b.class_name} ({b.confidence:.2f})"

        self._results_label.setText(
            f'{self._results_label.text()} — ROIs created')

    def _clear_detections(self):
        self._detections.clear()
        self._btn_create_rois.setEnabled(False)
        self._btn_clear.setEnabled(False)
        self._results_label.setText('Results cleared.')

    # -----------------------------------------------------------------------
    # Annotate tab logic
    # -----------------------------------------------------------------------

    def _ann_load_from_window(self):
        win = self._get_selected_window()
        if win is None:
            # Try current window
            win = g.win
        if win is None:
            g.alert('No window available.')
            return

        image = win.image
        if image.ndim == 2:
            h, w = image.shape
            n_frames = 1
        elif image.ndim == 3:
            n_frames, h, w = image.shape
        else:
            h, w = image.shape[-2], image.shape[-1]
            n_frames = image.shape[0] if image.ndim > 2 else 1

        self._annotation_set = AnnotationSet(
            image_width=w, image_height=h, n_frames=n_frames)
        # Add default class
        self._annotation_set.add_class('object', _DEFAULT_COLORS[0])

        # Load image into viewer
        self._ann_view.setImage(image)

        # Setup overlay
        if self._overlay is not None:
            scene = self._ann_view.getView().scene()
            if scene and self._overlay.scene() is scene:
                scene.removeItem(self._overlay)

        self._overlay = BoxAnnotationOverlay(self._annotation_set)
        self._ann_view.getView().addItem(self._overlay)

        # Connect frame change
        if image.ndim >= 3:
            self._ann_view.sigTimeChanged.connect(self._ann_frame_changed)

        self._overlay.sigBoxCreated.connect(lambda _: self._ann_update_stats())
        self._overlay.sigBoxRemoved.connect(lambda _: self._ann_update_stats())

        self._ann_refresh_class_list()
        self._ann_update_stats()

    def _ann_frame_changed(self, idx, _=None):
        if self._overlay:
            self._overlay.set_frame(idx)
        self._ann_update_stats()

    def _set_tool(self, mode: InteractionMode):
        if self._overlay:
            self._overlay.mode = mode

    def _ann_refresh_class_list(self):
        self._class_list.clear()
        if self._annotation_set is None:
            return
        for cls in self._annotation_set.classes:
            item = QtWidgets.QListWidgetItem(cls.name)
            item.setForeground(QtGui.QColor(*cls.color))
            item.setData(QtCore.Qt.UserRole, cls.id)
            self._class_list.addItem(item)
        if self._class_list.count() > 0:
            self._class_list.setCurrentRow(0)

    def _ann_class_selected(self, row: int):
        if row < 0 or self._annotation_set is None:
            return
        item = self._class_list.item(row)
        if item is None:
            return
        class_id = item.data(QtCore.Qt.UserRole)
        if self._overlay:
            self._overlay.set_active_class(class_id)

    def _ann_add_class(self):
        if self._annotation_set is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, 'Add Class', 'Class name:')
        if ok and name:
            idx = len(self._annotation_set.classes)
            color = _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]
            self._annotation_set.add_class(name, color)
            self._ann_refresh_class_list()

    def _ann_remove_class(self):
        if self._annotation_set is None:
            return
        row = self._class_list.currentRow()
        if row < 0:
            return
        item = self._class_list.item(row)
        class_id = item.data(QtCore.Qt.UserRole)
        self._annotation_set.remove_class(class_id)
        self._ann_refresh_class_list()
        if self._overlay:
            self._overlay.refresh()
        self._ann_update_stats()

    def _ann_rename_class(self):
        if self._annotation_set is None:
            return
        row = self._class_list.currentRow()
        if row < 0:
            return
        item = self._class_list.item(row)
        class_id = item.data(QtCore.Qt.UserRole)
        old_name = item.text()
        name, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename Class', 'New name:', text=old_name)
        if ok and name:
            self._annotation_set.rename_class(class_id, name)
            self._ann_refresh_class_list()
            if self._overlay:
                self._overlay.refresh()

    def _ann_update_stats(self):
        if self._annotation_set is None:
            self._ann_stats.setText('')
            return
        frame = 0
        if self._overlay:
            frame = self._overlay._current_frame
        frame_boxes = self._annotation_set.get_frame_boxes(frame)
        total = len(self._annotation_set.boxes)

        class_counts = {}
        for b in frame_boxes:
            class_counts[b.class_name] = class_counts.get(b.class_name, 0) + 1

        parts = [f'{v} {k}' for k, v in sorted(class_counts.items())]
        text = (f'Frame {frame + 1}/{self._annotation_set.n_frames} — '
                f'{len(frame_boxes)} boxes')
        if parts:
            text += f' ({", ".join(parts)})'
        text += f'\nTotal: {total} boxes'
        self._ann_stats.setText(text)

    def _ann_import_rois(self):
        """Import existing flika ROIs as annotations."""
        win = g.win
        if win is None:
            g.alert('No window selected.')
            return
        if not hasattr(win, 'rois') or not win.rois:
            g.alert('No ROIs in current window.')
            return

        if self._annotation_set is None:
            self._ann_load_from_window()
        if self._annotation_set is None:
            return

        # Assign all imported ROIs to current active class
        cls_item = self._class_list.currentItem()
        if cls_item is None:
            g.alert('No class selected.')
            return
        class_id = cls_item.data(QtCore.Qt.UserRole)
        cls = self._annotation_set.get_class_by_id(class_id)
        class_name = cls.name if cls else 'object'

        count = 0
        for roi in win.rois:
            roi_type = type(roi).__name__
            if 'rect' not in roi_type.lower():
                continue
            try:
                pos = roi.pos()
                size = roi.size()
                # ROI coordinates in pyqtgraph: pos().x() is row (y), pos().y() is col (x)
                box = BoundingBox(
                    x=pos.y(), y=pos.x(),
                    width=size.y(), height=size.x(),
                    class_id=class_id, class_name=class_name,
                    confidence=1.0, frame=0,
                )
                self._annotation_set.add_box(box)
                count += 1
            except (AttributeError, TypeError):
                continue

        if self._overlay:
            self._overlay.refresh()
        self._ann_update_stats()
        self._ann_stats.setText(self._ann_stats.text() + f'\nImported {count} ROIs')

    def _ann_import(self):
        """Import annotations from file."""
        if self._annotation_set is None:
            g.alert('Load an image first (use "Load from Window").')
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Import Annotations', '',
            'COCO JSON (*.json);;YOLO data.yaml (*.yaml *.yml);;All Files (*)')
        if not path:
            return

        try:
            if path.endswith('.json'):
                with open(path) as f:
                    coco = json.load(f)
                imported = AnnotationSet.from_coco_json(coco)
            elif path.endswith(('.yaml', '.yml')):
                label_dir = os.path.join(os.path.dirname(path), 'labels')
                imported = AnnotationSet.from_yolo_format(
                    label_dir, path,
                    self._annotation_set.image_width,
                    self._annotation_set.image_height)
            else:
                g.alert('Unsupported format.')
                return

            # Merge into current set
            for cls in imported.classes:
                if not self._annotation_set.get_class_by_id(cls.id):
                    self._annotation_set.classes.append(cls)
            for box in imported.boxes:
                self._annotation_set.add_box(box)

            self._ann_refresh_class_list()
            if self._overlay:
                self._overlay.refresh()
            self._ann_update_stats()

        except Exception as e:
            g.alert(f'Import failed: {e}')
            logger.error("Annotation import error: %s", e)

    def _ann_export(self):
        """Export annotations to file."""
        if self._annotation_set is None or not self._annotation_set.boxes:
            g.alert('No annotations to export.')
            return

        fmt, ok = QtWidgets.QInputDialog.getItem(
            self, 'Export Format', 'Format:',
            ['YOLO', 'COCO JSON', 'Pascal VOC XML'], editable=False)
        if not ok:
            return

        if fmt == 'YOLO':
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Select Output Directory')
            if out_dir:
                yaml_path = self._annotation_set.to_yolo_format(out_dir)
                g.alert(f'Exported to {yaml_path}')
        elif fmt == 'COCO JSON':
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save COCO JSON', 'annotations.json', 'JSON (*.json)')
            if path:
                coco = self._annotation_set.to_coco_json()
                with open(path, 'w') as f:
                    json.dump(coco, f, indent=2)
                g.alert(f'Exported to {path}')
        elif fmt == 'Pascal VOC XML':
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Select Output Directory')
            if out_dir:
                for frame in range(self._annotation_set.n_frames):
                    xml = self._annotation_set.to_voc_xml(
                        f'frame_{frame:06d}.png', frame)
                    with open(os.path.join(out_dir, f'frame_{frame:06d}.xml'), 'w') as f:
                        f.write(xml)
                g.alert(f'Exported {self._annotation_set.n_frames} XML files to {out_dir}')

    # -----------------------------------------------------------------------
    # Train tab logic
    # -----------------------------------------------------------------------

    def _browse_data_dir(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select data.yaml', '', 'YAML (*.yaml *.yml);;All Files (*)')
        if path:
            self._data_yaml_path.setText(path)

    def _start_training(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            g.alert("ultralytics is not installed.\n\n"
                    "Install with:  pip install ultralytics")
            return

        fine_tune = self._train_mode.currentIndex() == 1

        # Get data.yaml
        if self._data_source.currentIndex() == 0:
            # From annotate tab
            if self._annotation_set is None or not self._annotation_set.boxes:
                g.alert('No annotations. Use the Annotate tab first.')
                return
            # Export to temp dir
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix='flika_det_')
            # Save annotation images
            img_dir = os.path.join(tmp_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            win = self._get_selected_window() or g.win
            if win is not None:
                image = win.image
                from PIL import Image as PILImage
                if image.ndim == 2:
                    frames = [image]
                elif image.ndim == 3:
                    frames = [image[i] for i in range(image.shape[0])]
                else:
                    frames = [image]
                paths = []
                for i, frame in enumerate(frames):
                    img_rgb = prepare_image_for_detection(frame)
                    pil = PILImage.fromarray(img_rgb)
                    p = os.path.join(img_dir, f'frame_{i:06d}.png')
                    pil.save(p)
                    paths.append(p)
                data_yaml = self._annotation_set.to_yolo_format(tmp_dir, paths)
            else:
                g.alert('No window available for image data.')
                return
        else:
            data_yaml = self._data_yaml_path.text()
            if not data_yaml or not os.path.exists(data_yaml):
                g.alert('Please select a valid data.yaml file.')
                return

        config = DetectionConfig(
            model_size=self._train_size.currentText(),
            epochs=self._train_epochs.value(),
            batch_size=self._train_batch.value(),
            learning_rate=self._train_lr.value(),
            freeze_layers=self._train_freeze.value(),
            augment=self._train_augment.isChecked(),
            experiment_name=self._train_name.text() or 'flika_detector',
            device=self._train_device.currentText(),
        )

        if fine_tune:
            config.model_path = self._ft_model_combo.currentData() or ''

        # Reset loss plot
        self._loss_data = {'box': [], 'cls': [], 'dfl': []}
        self._box_loss_curve.setData([], [])
        self._cls_loss_curve.setData([], [])
        self._dfl_loss_curve.setData([], [])

        self._btn_train.setEnabled(False)
        self._btn_stop_train.setEnabled(True)
        self._train_progress.setVisible(True)
        self._train_progress.setValue(0)
        self._train_status.setText('Training...')

        self._train_worker = DetectionTrainWorker(
            self._backend, data_yaml, config, fine_tune=fine_tune, parent=self)
        self._train_worker.sig_epoch_done.connect(self._on_epoch_done)
        self._train_worker.sig_progress.connect(self._train_progress.setValue)
        self._train_worker.sig_finished.connect(self._on_train_finished)
        self._train_worker.sig_error.connect(self._on_train_error)
        self._train_worker.start()

    def _on_epoch_done(self, epoch: int, box_loss: float, cls_loss: float, dfl_loss: float):
        self._loss_data['box'].append(box_loss)
        self._loss_data['cls'].append(cls_loss)
        self._loss_data['dfl'].append(dfl_loss)
        epochs = list(range(1, len(self._loss_data['box']) + 1))
        self._box_loss_curve.setData(epochs, self._loss_data['box'])
        self._cls_loss_curve.setData(epochs, self._loss_data['cls'])
        self._dfl_loss_curve.setData(epochs, self._loss_data['dfl'])

    def _on_train_finished(self, model_path: str):
        self._btn_train.setEnabled(True)
        self._btn_stop_train.setEnabled(False)
        self._train_progress.setVisible(False)
        self._train_status.setText(
            f'Training complete! Model saved to:\n{model_path}')
        self._refresh_models()
        self._refresh_saved_models()

    def _on_train_error(self, msg: str):
        self._btn_train.setEnabled(True)
        self._btn_stop_train.setEnabled(False)
        self._train_progress.setVisible(False)
        self._train_status.setText(f'Error: {msg}')
        logger.error("Training error: %s", msg)

    # -----------------------------------------------------------------------
    # Saved Models management
    # -----------------------------------------------------------------------

    def _refresh_saved_models(self):
        """Populate the saved models list from ~/.FLIKA/models/detectors/."""
        self._saved_models_list.clear()
        from .detection_backend import _models_dir
        models_path = _models_dir()
        if not os.path.isdir(models_path):
            return
        for fname in sorted(os.listdir(models_path)):
            if not fname.endswith('.pt'):
                continue
            full = os.path.join(models_path, fname)
            size_mb = os.path.getsize(full) / (1024 * 1024)
            import time as _time
            mtime = _time.strftime(
                '%Y-%m-%d %H:%M',
                _time.localtime(os.path.getmtime(full)))
            item = QtWidgets.QListWidgetItem(
                f'{fname}  ({size_mb:.1f} MB, {mtime})')
            item.setData(QtCore.Qt.UserRole, full)
            item.setToolTip(full)
            self._saved_models_list.addItem(item)

    def _open_models_folder(self):
        from .detection_backend import _models_dir
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(_models_dir()))

    def _save_model_copy(self):
        """Save a copy of the selected model to a user-chosen location."""
        item = self._saved_models_list.currentItem()
        if item is None:
            g.alert('Select a model from the list first.')
            return
        src = item.data(QtCore.Qt.UserRole)
        if not os.path.isfile(src):
            g.alert('Model file not found.')
            return
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Model Copy',
            os.path.basename(src),
            'PyTorch Models (*.pt);;All Files (*)')
        if dst:
            import shutil
            shutil.copy2(src, dst)
            g.alert(f'Model saved to:\n{dst}')

    def _import_model(self):
        """Import an external .pt model into the flika models folder."""
        src, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Import Model',
            '', 'PyTorch Models (*.pt);;All Files (*)')
        if not src:
            return
        from .detection_backend import _models_dir
        import shutil
        dst = os.path.join(_models_dir(), os.path.basename(src))
        if os.path.exists(dst):
            reply = QtWidgets.QMessageBox.question(
                self, 'Overwrite?',
                f'A model named "{os.path.basename(src)}" already exists.\n'
                'Overwrite it?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        shutil.copy2(src, dst)
        self._refresh_saved_models()
        self._refresh_models()
        g.alert(f'Model imported to:\n{dst}')

    def _delete_saved_model(self):
        """Delete the selected saved model."""
        item = self._saved_models_list.currentItem()
        if item is None:
            g.alert('Select a model from the list first.')
            return
        path = item.data(QtCore.Qt.UserRole)
        fname = os.path.basename(path)
        reply = QtWidgets.QMessageBox.question(
            self, 'Delete Model',
            f'Are you sure you want to delete "{fname}"?\n\n'
            'This cannot be undone.',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except OSError as e:
            g.alert(f'Failed to delete model:\n{e}')
            return
        self._refresh_saved_models()
        self._refresh_models()
