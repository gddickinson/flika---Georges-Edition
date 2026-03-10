"""Training dialog for AI denoising in flika.

Provides a full GUI for training and predicting with N2V (self-supervised)
and CARE (supervised) denoising models, with a live loss curve.
"""
from __future__ import annotations

from qtpy import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

import flika.global_vars as g
from flika.logger import logger
from flika.ai.denoiser import (
    DenoiserConfig, TrainingWorker, PredictionWorker,
    detect_denoiser_backend,
)


class DenoiserDialog(QtWidgets.QDialog):
    """Dialog for AI denoiser training and prediction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Denoiser")
        self.setMinimumSize(560, 620)
        self._train_worker = None
        self._predict_worker = None
        self._train_losses = []
        self._val_losses = []
        self._build_ui()
        self._connect_signals()
        self._on_mode_changed()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- Configuration group ---
        config_group = QtWidgets.QGroupBox("Configuration")
        config_layout = QtWidgets.QGridLayout(config_group)

        config_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([
            'N2V (Self-Supervised)',
            'CARE (Supervised)',
            'Load Pre-trained',
        ])
        config_layout.addWidget(self.mode_combo, 0, 1)

        config_layout.addWidget(QtWidgets.QLabel("Device:"), 0, 2)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        config_layout.addWidget(self.device_combo, 0, 3)

        config_layout.addWidget(QtWidgets.QLabel("Noisy Data:"), 1, 0)
        self.noisy_combo = QtWidgets.QComboBox()
        config_layout.addWidget(self.noisy_combo, 1, 1, 1, 3)

        config_layout.addWidget(QtWidgets.QLabel("Clean Data:"), 2, 0)
        self.clean_combo = QtWidgets.QComboBox()
        config_layout.addWidget(self.clean_combo, 2, 1, 1, 3)

        config_layout.addWidget(QtWidgets.QLabel("Checkpoint:"), 3, 0)
        self.checkpoint_edit = QtWidgets.QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Path to pre-trained model (optional)")
        config_layout.addWidget(self.checkpoint_edit, 3, 1, 1, 2)
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_checkpoint)
        config_layout.addWidget(self.browse_btn, 3, 3)

        layout.addWidget(config_group)

        # --- Training parameters group ---
        self.params_group = QtWidgets.QGroupBox("Training Parameters")
        params_layout = QtWidgets.QGridLayout(self.params_group)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        params_layout.addWidget(QtWidgets.QLabel("Epochs:"), 0, 0)
        params_layout.addWidget(self.epochs_spin, 0, 1)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(8)
        params_layout.addWidget(QtWidgets.QLabel("Batch Size:"), 0, 2)
        params_layout.addWidget(self.batch_spin, 0, 3)

        self.patch_spin = QtWidgets.QSpinBox()
        self.patch_spin.setRange(16, 512)
        self.patch_spin.setSingleStep(16)
        self.patch_spin.setValue(64)
        params_layout.addWidget(QtWidgets.QLabel("Patch Size:"), 1, 0)
        params_layout.addWidget(self.patch_spin, 1, 1)

        self.lr_edit = QtWidgets.QLineEdit("0.0001")
        params_layout.addWidget(QtWidgets.QLabel("Learning Rate:"), 1, 2)
        params_layout.addWidget(self.lr_edit, 1, 3)

        self.val_spin = QtWidgets.QDoubleSpinBox()
        self.val_spin.setRange(0.01, 0.5)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.10)
        params_layout.addWidget(QtWidgets.QLabel("Val Split:"), 2, 0)
        params_layout.addWidget(self.val_spin, 2, 1)

        self.name_edit = QtWidgets.QLineEdit("flika_denoiser")
        params_layout.addWidget(QtWidgets.QLabel("Name:"), 2, 2)
        params_layout.addWidget(self.name_edit, 2, 3)

        layout.addWidget(self.params_group)

        # --- Loss curve ---
        self.loss_plot = pg.PlotWidget(title="Loss Curve")
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.addLegend()
        self.train_curve = self.loss_plot.plot(pen='y', name='Train')
        self.val_curve = self.loss_plot.plot(pen='c', name='Val')
        self.loss_plot.setMinimumHeight(180)
        layout.addWidget(self.loss_plot)

        # --- Progress bar ---
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # --- Status label ---
        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Start Training")
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_btn)

        self.predict_btn = QtWidgets.QPushButton("Predict")
        self.predict_btn.clicked.connect(self._start_prediction)
        btn_layout.addWidget(self.predict_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        # Backend status
        backend = detect_denoiser_backend()
        if backend == 'none':
            self.status_label.setText(
                "No backend installed. Install: pip install careamics")
        else:
            self.status_label.setText(f"Backend: {backend}")

        self._refresh_window_combos()

    def _connect_signals(self):
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

    def _on_mode_changed(self):
        """Enable/disable controls based on selected mode."""
        mode_idx = self.mode_combo.currentIndex()
        is_n2v = mode_idx == 0
        is_care = mode_idx == 1
        is_pretrained = mode_idx == 2

        # Clean data only for CARE
        self.clean_combo.setEnabled(is_care)

        # Training params disabled for load-pretrained
        self.params_group.setEnabled(not is_pretrained)

        # Train button disabled for load-pretrained
        self.train_btn.setEnabled(not is_pretrained)

        # Noisy data always needed for predict
        self.noisy_combo.setEnabled(True)

    def _refresh_window_combos(self):
        """Populate window selectors from g.windows."""
        self.noisy_combo.clear()
        self.clean_combo.clear()
        for w in g.windows:
            name = getattr(w, 'name', str(w))
            self.noisy_combo.addItem(name)
            self.clean_combo.addItem(name)

    def _get_window_data(self, combo: QtWidgets.QComboBox):
        """Return the image array from the selected window."""
        idx = combo.currentIndex()
        if idx < 0 or idx >= len(g.windows):
            return None
        win = g.windows[idx]
        if hasattr(win, 'volume') and win.volume is not None:
            return win.volume
        return win.image

    def _browse_checkpoint(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "",
            "All Files (*);;ZIP Files (*.zip);;H5 Files (*.h5)")
        if path:
            self.checkpoint_edit.setText(path)

    def _build_config(self) -> DenoiserConfig:
        """Build a DenoiserConfig from current UI state."""
        mode_idx = self.mode_combo.currentIndex()
        mode = 'n2v' if mode_idx == 0 else 'care'

        try:
            lr = float(self.lr_edit.text())
        except ValueError:
            lr = 1e-4

        return DenoiserConfig(
            mode=mode,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            patch_size=self.patch_spin.value(),
            learning_rate=lr,
            val_percentage=self.val_spin.value(),
            device=self.device_combo.currentText(),
            experiment_name=self.name_edit.text() or 'flika_denoiser',
            checkpoint_path=self.checkpoint_edit.text(),
        )

    def _start_training(self):
        """Launch training in a background thread."""
        train_data = self._get_window_data(self.noisy_combo)
        if train_data is None:
            g.alert("No noisy data window selected.")
            return

        config = self._build_config()

        clean_data = None
        if config.mode == 'care':
            clean_data = self._get_window_data(self.clean_combo)
            if clean_data is None:
                g.alert("CARE mode requires a clean data window.")
                return

        # Reset loss curves
        self._train_losses.clear()
        self._val_losses.clear()
        self.train_curve.setData([], [])
        self.val_curve.setData([], [])
        self.progress_bar.setValue(0)

        self._train_worker = TrainingWorker(config, train_data,
                                             clean_data=clean_data, parent=self)
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
        """Request the training thread to stop."""
        if self._train_worker is not None:
            self._train_worker.request_stop()
            self.status_label.setText("Stopping after current epoch...")
            self.stop_btn.setEnabled(False)

    def _on_epoch_done(self, epoch: int, train_loss: float, val_loss: float):
        """Update loss curve on each epoch."""
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        epochs = list(range(1, len(self._train_losses) + 1))
        self.train_curve.setData(epochs, self._train_losses)
        self.val_curve.setData(epochs, self._val_losses)

    def _on_train_finished(self, model_path: str):
        """Handle training completion."""
        self.status_label.setText(f"Training complete. Model saved to: {model_path}")
        self.checkpoint_edit.setText(model_path)
        logger.info("Denoiser training complete: %s", model_path)

    def _on_worker_done(self):
        """Re-enable UI after worker finishes."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, msg: str):
        """Show error from worker thread."""
        self.status_label.setText(f"Error: {msg}")
        g.alert(msg)

    def _start_prediction(self):
        """Launch prediction in a background thread."""
        data = self._get_window_data(self.noisy_combo)
        if data is None:
            g.alert("No data window selected for prediction.")
            return

        config = self._build_config()

        # For pre-trained mode, checkpoint is required
        if self.mode_combo.currentIndex() == 2 and not config.checkpoint_path:
            g.alert("Load Pre-trained mode requires a checkpoint path.")
            return

        self._predict_worker = PredictionWorker(config, data, parent=self)
        self._predict_worker.sig_result.connect(self._on_predict_result)
        self._predict_worker.sig_progress.connect(self.progress_bar.setValue)
        self._predict_worker.sig_status.connect(self.status_label.setText)
        self._predict_worker.sig_error.connect(self._on_error)
        self._predict_worker.finished.connect(
            lambda: self.predict_btn.setEnabled(True))

        self.predict_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._predict_worker.start()

    def _on_predict_result(self, result):
        """Create a new Window with the denoised result."""
        from flika.window import Window

        result = np.asarray(result, dtype=np.float32)
        n_nan = int(np.count_nonzero(np.isnan(result)))
        if n_nan > 0:
            logger.warning("Denoised result contains %d NaN values — replacing with 0", n_nan)
            np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        result = result.astype(g.settings['internal_data_type'])
        src_name = self.noisy_combo.currentText()
        Window(result, name=src_name + ' - Denoised')
        if n_nan > 0:
            self.status_label.setText(
                f"Prediction complete (warning: {n_nan} NaN pixels replaced with 0 — "
                f"model may need more training epochs).")
        else:
            self.status_label.setText("Prediction complete.")

    def closeEvent(self, event):
        """Clean up workers on close."""
        for worker in (self._train_worker, self._predict_worker):
            if worker is not None and worker.isRunning():
                worker.request_stop()
                worker.wait(5000)
        super().closeEvent(event)
