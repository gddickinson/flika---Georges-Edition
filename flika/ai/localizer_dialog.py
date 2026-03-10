"""Particle Localizer dialog for flika.

Provides a self-contained dialog for DeepSTORM-style particle localization:
- Train from PSF simulation or real data
- Predict density maps on flika windows
- Extract sub-pixel coordinates
- Optional particle tracking via utils/tracking.py
- Export localizations as CSV
"""
from __future__ import annotations

import os

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Signal
import pyqtgraph as pg
import numpy as np

import flika.global_vars as g
from flika.logger import logger


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class LocalizerTrainingWorker(QtCore.QThread):
    """Background thread for localizer training."""

    sig_epoch_done = Signal(int, float)    # epoch, loss
    sig_progress = Signal(int)             # percent 0-100
    sig_status = Signal(str)
    sig_finished = Signal()
    sig_error = Signal(str)

    def __init__(self, backend, images, density_maps, config, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.images = images
        self.density_maps = density_maps
        self.config = config
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            total_epochs = self.config.epochs

            def callback(epoch, loss):
                self.sig_epoch_done.emit(epoch, loss)
                pct = int(100 * (epoch + 1) / max(total_epochs, 1))
                self.sig_progress.emit(min(pct, 100))

            self.sig_status.emit(f"Training {self.backend.name()}...")
            self.backend.train(self.images, self.density_maps, callback=callback)
            self.sig_progress.emit(100)
            self.sig_status.emit("Training complete.")
            self.sig_finished.emit()
        except Exception as e:
            logger.exception("Localizer training failed")
            self.sig_error.emit(str(e))


class LocalizerPredictionWorker(QtCore.QThread):
    """Background thread for localizer prediction."""

    sig_result = Signal(object, object)   # density_maps, localizations
    sig_progress = Signal(int)
    sig_status = Signal(str)
    sig_error = Signal(str)

    def __init__(self, backend, data, threshold, min_distance,
                 do_tracking=False, max_track_dist=5.0, max_gap=1,
                 parent=None):
        super().__init__(parent)
        self.backend = backend
        self.data = data
        self.threshold = threshold
        self.min_distance = min_distance
        self.do_tracking = do_tracking
        self.max_track_dist = max_track_dist
        self.max_gap = max_gap

    def run(self):
        try:
            from flika.ai.psf_simulator import PSFSimulator

            data = self.data
            is_stack = data.ndim == 3

            if is_stack:
                n_frames = data.shape[0]
                density_maps = np.zeros_like(data, dtype=np.float32)
                all_locs = []

                for i in range(n_frames):
                    self.sig_status.emit(f"Predicting frame {i+1}/{n_frames}...")
                    density = self.backend.predict(data[i])
                    density_maps[i] = density
                    coords = PSFSimulator.extract_coordinates(
                        density, self.threshold, self.min_distance)
                    # Store as (frame, y, x)
                    if len(coords) > 0:
                        frame_col = np.full((len(coords), 1), i, dtype=np.float64)
                        all_locs.append(np.hstack([frame_col, coords]))
                    pct = int(100 * (i + 1) / n_frames)
                    self.sig_progress.emit(pct)

                if all_locs:
                    localizations = np.vstack(all_locs)
                else:
                    localizations = np.empty((0, 3), dtype=np.float64)

                # Optional tracking
                if self.do_tracking and len(localizations) > 0:
                    self.sig_status.emit("Linking particles...")
                    localizations = self._run_tracking(localizations)

            else:
                self.sig_status.emit("Predicting...")
                self.sig_progress.emit(10)
                density_maps = self.backend.predict(data)
                coords = PSFSimulator.extract_coordinates(
                    density_maps, self.threshold, self.min_distance)
                if len(coords) > 0:
                    frame_col = np.zeros((len(coords), 1), dtype=np.float64)
                    localizations = np.hstack([frame_col, coords])
                else:
                    localizations = np.empty((0, 3), dtype=np.float64)
                self.sig_progress.emit(100)

            self.sig_status.emit(
                f"Found {len(localizations)} localizations.")
            self.sig_result.emit(density_maps, localizations)

        except Exception as e:
            logger.exception("Localizer prediction failed")
            self.sig_error.emit(str(e))

    def _run_tracking(self, localizations):
        """Link localizations into tracks using utils/tracking.py."""
        try:
            from flika.utils.tracking import link_particles, tracks_to_array

            # link_particles expects list of (N, 2) arrays per frame
            frames = np.unique(localizations[:, 0].astype(int))
            per_frame = []
            for f in range(int(frames.max()) + 1):
                mask = localizations[:, 0].astype(int) == f
                if mask.any():
                    per_frame.append(localizations[mask, 1:3])
                else:
                    per_frame.append(np.empty((0, 2)))

            tracks = link_particles(per_frame, max_distance=self.max_track_dist,
                                    max_gap=self.max_gap)
            result = tracks_to_array(per_frame, tracks)
            return result
        except Exception as e:
            logger.warning("Tracking failed: %s", e)
            return localizations


# ---------------------------------------------------------------------------
# Simulation data generator (background)
# ---------------------------------------------------------------------------

class SimulationWorker(QtCore.QThread):
    """Generate simulated training data in background."""

    sig_result = Signal(object, object)   # images, density_maps
    sig_status = Signal(str)
    sig_error = Signal(str)

    def __init__(self, psf_config, parent=None):
        super().__init__(parent)
        self.psf_config = psf_config

    def run(self):
        try:
            from flika.ai.psf_simulator import PSFSimulator
            self.sig_status.emit("Generating simulated data...")
            sim = PSFSimulator(self.psf_config)
            stack, all_positions = sim.generate_stack()

            # Build density maps
            h = w = self.psf_config.image_size
            density_maps = np.zeros_like(stack, dtype=np.float32)
            for i, positions in enumerate(all_positions):
                density_maps[i] = sim.positions_to_density_map(positions, (h, w))

            self.sig_status.emit(f"Generated {len(stack)} training frames.")
            self.sig_result.emit(stack, density_maps)
        except Exception as e:
            logger.exception("Simulation generation failed")
            self.sig_error.emit(str(e))


# ---------------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------------

class ParticleLocalizerDialog(QtWidgets.QDialog):
    """Self-contained dialog for DeepSTORM-style particle localization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Particle Localizer")
        self.setMinimumSize(650, 750)

        self._train_worker = None
        self._predict_worker = None
        self._sim_worker = None
        self._backend = None
        self._train_losses = []
        self._sim_images = None
        self._sim_densities = None
        self._last_localizations = None

        self._build_ui()
        self._connect_signals()
        self._refresh_window_combos()
        self._on_mode_changed()

    # ---- helpers for window combos ----

    def _refresh_window_combos(self):
        for combo in (self.train_window_combo, self.predict_window_combo):
            combo.clear()
            for w in g.windows:
                name = getattr(w, 'name', str(w))
                combo.addItem(name)

    @staticmethod
    def _get_window_data_from_combo(combo):
        idx = combo.currentIndex()
        if idx < 0 or idx >= len(g.windows):
            return None
        win = g.windows[idx]
        if hasattr(win, 'volume') and win.volume is not None:
            return win.volume
        return win.image

    def _get_predict_data(self):
        """Return the prediction data slice according to the frame range."""
        data = self._get_window_data_from_combo(self.predict_window_combo)
        if data is None:
            return None

        if data.ndim == 3 and not self.all_frames_check.isChecked():
            start = self.frame_start_spin.value()
            end = self.frame_end_spin.value()
            # Clamp
            start = max(0, min(start, data.shape[0] - 1))
            end = max(start, min(end, data.shape[0] - 1))
            data = data[start:end + 1]
        return data

    def _on_predict_window_changed(self):
        """Update frame range spinboxes when predict window changes."""
        data = self._get_window_data_from_combo(self.predict_window_combo)
        if data is not None and data.ndim == 3:
            n = data.shape[0]
            self.frame_start_spin.setRange(0, n - 1)
            self.frame_end_spin.setRange(0, n - 1)
            self.frame_end_spin.setValue(n - 1)
            self.all_frames_check.setEnabled(True)
            self.frame_start_spin.setEnabled(not self.all_frames_check.isChecked())
            self.frame_end_spin.setEnabled(not self.all_frames_check.isChecked())
        else:
            self.frame_start_spin.setRange(0, 0)
            self.frame_end_spin.setRange(0, 0)
            self.all_frames_check.setEnabled(False)
            self.frame_start_spin.setEnabled(False)
            self.frame_end_spin.setEnabled(False)

    # ---- UI build ----

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Mode and Device
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([
            'Train from Simulation',
            'Train from Data',
            'Load Pre-trained',
        ])
        top_row.addWidget(self.mode_combo)

        top_row.addWidget(QtWidgets.QLabel("Device:"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        top_row.addWidget(self.device_combo)
        layout.addLayout(top_row)

        # Train / Predict window selection
        win_group = QtWidgets.QGroupBox("Image Selection")
        win_layout = QtWidgets.QGridLayout(win_group)

        win_layout.addWidget(QtWidgets.QLabel("Train Image:"), 0, 0)
        self.train_window_combo = QtWidgets.QComboBox()
        win_layout.addWidget(self.train_window_combo, 0, 1, 1, 3)

        win_layout.addWidget(QtWidgets.QLabel("Predict Image:"), 1, 0)
        self.predict_window_combo = QtWidgets.QComboBox()
        win_layout.addWidget(self.predict_window_combo, 1, 1, 1, 3)

        # Frame range for prediction
        win_layout.addWidget(QtWidgets.QLabel("Frames:"), 2, 0)
        self.all_frames_check = QtWidgets.QCheckBox("All")
        self.all_frames_check.setChecked(True)
        win_layout.addWidget(self.all_frames_check, 2, 1)

        self.frame_start_spin = QtWidgets.QSpinBox()
        self.frame_start_spin.setPrefix("Start: ")
        self.frame_start_spin.setRange(0, 0)
        self.frame_start_spin.setEnabled(False)
        win_layout.addWidget(self.frame_start_spin, 2, 2)

        self.frame_end_spin = QtWidgets.QSpinBox()
        self.frame_end_spin.setPrefix("End: ")
        self.frame_end_spin.setRange(0, 0)
        self.frame_end_spin.setEnabled(False)
        win_layout.addWidget(self.frame_end_spin, 2, 3)

        layout.addWidget(win_group)

        # PSF parameters (simulation mode)
        self.psf_group = QtWidgets.QGroupBox("PSF Parameters (Simulation)")
        psf_layout = QtWidgets.QGridLayout(self.psf_group)

        psf_layout.addWidget(QtWidgets.QLabel("Sigma:"), 0, 0)
        self.psf_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.psf_sigma_spin.setRange(0.5, 10.0)
        self.psf_sigma_spin.setValue(1.5)
        self.psf_sigma_spin.setSingleStep(0.1)
        psf_layout.addWidget(self.psf_sigma_spin, 0, 1)

        psf_layout.addWidget(QtWidgets.QLabel("Particles:"), 0, 2)
        self.particles_spin = QtWidgets.QSpinBox()
        self.particles_spin.setRange(1, 500)
        self.particles_spin.setValue(50)
        psf_layout.addWidget(self.particles_spin, 0, 3)

        psf_layout.addWidget(QtWidgets.QLabel("Frames:"), 0, 4)
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(10, 10000)
        self.frames_spin.setValue(500)
        psf_layout.addWidget(self.frames_spin, 0, 5)

        psf_layout.addWidget(QtWidgets.QLabel("Intensity Min:"), 1, 0)
        self.intensity_min = QtWidgets.QDoubleSpinBox()
        self.intensity_min.setRange(0, 100000)
        self.intensity_min.setValue(500)
        psf_layout.addWidget(self.intensity_min, 1, 1)
        psf_layout.addWidget(QtWidgets.QLabel("Intensity Max:"), 1, 2)
        self.intensity_max = QtWidgets.QDoubleSpinBox()
        self.intensity_max.setRange(0, 100000)
        self.intensity_max.setValue(2000)
        psf_layout.addWidget(self.intensity_max, 1, 3)

        psf_layout.addWidget(QtWidgets.QLabel("Background:"), 1, 4)
        self.bg_mean_spin = QtWidgets.QDoubleSpinBox()
        self.bg_mean_spin.setRange(0, 10000)
        self.bg_mean_spin.setValue(100)
        psf_layout.addWidget(self.bg_mean_spin, 1, 5)

        psf_layout.addWidget(QtWidgets.QLabel("Noise:"), 2, 0)
        self.noise_combo = QtWidgets.QComboBox()
        self.noise_combo.addItems(['poisson', 'gaussian', 'mixed'])
        psf_layout.addWidget(self.noise_combo, 2, 1)

        self.view_sim_btn = QtWidgets.QPushButton("View Simulation")
        self.view_sim_btn.setEnabled(False)
        self.view_sim_btn.setToolTip("Open simulated images and density maps as flika Windows")
        psf_layout.addWidget(self.view_sim_btn, 2, 4, 1, 2)

        layout.addWidget(self.psf_group)

        # Training parameters
        self.train_group = QtWidgets.QGroupBox("Training Parameters")
        train_layout = QtWidgets.QGridLayout(self.train_group)

        train_layout.addWidget(QtWidgets.QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        train_layout.addWidget(self.epochs_spin, 0, 1)

        train_layout.addWidget(QtWidgets.QLabel("Batch:"), 0, 2)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        train_layout.addWidget(self.batch_spin, 0, 3)

        train_layout.addWidget(QtWidgets.QLabel("LR:"), 0, 4)
        self.lr_edit = QtWidgets.QLineEdit("0.001")
        train_layout.addWidget(self.lr_edit, 0, 5)

        train_layout.addWidget(QtWidgets.QLabel("Checkpoint:"), 1, 0)
        self.checkpoint_edit = QtWidgets.QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Path to pre-trained model (optional)")
        train_layout.addWidget(self.checkpoint_edit, 1, 1, 1, 4)
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_checkpoint)
        train_layout.addWidget(self.browse_btn, 1, 5)

        layout.addWidget(self.train_group)

        # Detection parameters
        det_group = QtWidgets.QGroupBox("Detection & Tracking")
        det_layout = QtWidgets.QGridLayout(det_group)

        det_layout.addWidget(QtWidgets.QLabel("Threshold:"), 0, 0)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 1.0)
        self.threshold_spin.setValue(0.2)
        self.threshold_spin.setSingleStep(0.05)
        det_layout.addWidget(self.threshold_spin, 0, 1)

        det_layout.addWidget(QtWidgets.QLabel("Min Distance:"), 0, 2)
        self.min_dist_spin = QtWidgets.QSpinBox()
        self.min_dist_spin.setRange(1, 50)
        self.min_dist_spin.setValue(3)
        det_layout.addWidget(self.min_dist_spin, 0, 3)

        self.track_check = QtWidgets.QCheckBox("Link particles")
        det_layout.addWidget(self.track_check, 1, 0, 1, 2)

        det_layout.addWidget(QtWidgets.QLabel("Max Dist:"), 1, 2)
        self.track_dist_spin = QtWidgets.QDoubleSpinBox()
        self.track_dist_spin.setRange(0.1, 100.0)
        self.track_dist_spin.setValue(5.0)
        det_layout.addWidget(self.track_dist_spin, 1, 3)

        det_layout.addWidget(QtWidgets.QLabel("Max Gap:"), 1, 4)
        self.track_gap_spin = QtWidgets.QSpinBox()
        self.track_gap_spin.setRange(0, 20)
        self.track_gap_spin.setValue(1)
        det_layout.addWidget(self.track_gap_spin, 1, 5)

        layout.addWidget(det_group)

        # Loss curve
        self.loss_plot = pg.PlotWidget(title="Loss")
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_curve = self.loss_plot.plot(pen='y', name='Train')
        self.loss_plot.setMinimumHeight(150)
        layout.addWidget(self.loss_plot)

        # Progress
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.train_btn = QtWidgets.QPushButton("Train")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.localize_btn = QtWidgets.QPushButton("Localize")
        self.export_btn = QtWidgets.QPushButton("Export CSV")
        self.close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.localize_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _connect_signals(self):
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn.clicked.connect(self._stop_training)
        self.localize_btn.clicked.connect(self._start_prediction)
        self.export_btn.clicked.connect(self._export_csv)
        self.close_btn.clicked.connect(self.close)
        self.view_sim_btn.clicked.connect(self._view_simulation)
        self.predict_window_combo.currentIndexChanged.connect(
            self._on_predict_window_changed)
        self.all_frames_check.toggled.connect(self._on_all_frames_toggled)

    def _on_all_frames_toggled(self, checked):
        self.frame_start_spin.setEnabled(not checked)
        self.frame_end_spin.setEnabled(not checked)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentIndex()
        self.psf_group.setVisible(mode == 0)           # Simulation
        self.train_group.setVisible(mode != 2)          # Not load
        self.train_btn.setEnabled(mode != 2)
        self.checkpoint_edit.setEnabled(mode == 2)
        self.browse_btn.setEnabled(mode == 2)

    def _browse_checkpoint(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pt);;All Files (*)")
        if path:
            self.checkpoint_edit.setText(path)

    # --- View Simulation ---

    def _view_simulation(self):
        """Open simulated images and density maps as flika Windows."""
        if self._sim_images is None or self._sim_densities is None:
            g.alert("No simulation data available. Train first.")
            return
        from flika.window import Window
        Window(self._sim_images.astype(g.settings['internal_data_type']),
               name='Simulated Images')
        Window(self._sim_densities.astype(g.settings['internal_data_type']),
               name='Simulated Density Maps')
        self.status_label.setText("Opened simulated images and density maps.")

    # --- Training ---

    def _build_config(self):
        from flika.ai.localizer_backends import LocalizerConfig
        try:
            lr = float(self.lr_edit.text())
        except ValueError:
            lr = 1e-3
        return LocalizerConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=lr,
            device=self.device_combo.currentText(),
            checkpoint_path=self.checkpoint_edit.text(),
            detection_threshold=self.threshold_spin.value(),
            min_distance=self.min_dist_spin.value(),
            psf_sigma=self.psf_sigma_spin.value(),
            n_particles=self.particles_spin.value(),
            n_training_frames=self.frames_spin.value(),
        )

    def _start_training(self):
        mode = self.mode_combo.currentIndex()
        config = self._build_config()

        if mode == 0:
            # Train from simulation — first generate data
            self._start_simulation(config)
        elif mode == 1:
            # Train from data — use selected window as images
            # User must provide density map window somehow;
            # for simplicity we use simulation with matched PSF params
            self.status_label.setText("Train from Data: generating simulation with your PSF params...")
            self._start_simulation(config)

    def _start_simulation(self, config):
        """Generate simulated training data, then train."""
        from flika.ai.psf_simulator import PSFConfig

        psf_config = PSFConfig(
            image_size=128,
            n_particles=config.n_particles,
            psf_sigma=config.psf_sigma,
            intensity_range=(self.intensity_min.value(), self.intensity_max.value()),
            background_mean=self.bg_mean_spin.value(),
            noise_type=self.noise_combo.currentText(),
            n_frames=config.n_training_frames,
        )

        self._pending_config = config
        self._sim_worker = SimulationWorker(psf_config, parent=self)
        self._sim_worker.sig_result.connect(self._on_sim_done)
        self._sim_worker.sig_status.connect(self.status_label.setText)
        self._sim_worker.sig_error.connect(self._on_error)

        self.train_btn.setEnabled(False)
        self.view_sim_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._sim_worker.start()

    def _on_sim_done(self, images, density_maps):
        """Simulation complete — now train the backend."""
        self._sim_images = images
        self._sim_densities = density_maps
        self.view_sim_btn.setEnabled(True)
        config = self._pending_config

        from flika.ai.localizer_backends import create_backend
        self._backend = create_backend(config)

        # Reset loss curve
        self._train_losses.clear()
        self.loss_curve.setData([], [])

        self._train_worker = LocalizerTrainingWorker(
            self._backend, images, density_maps, config, parent=self)
        self._train_worker.sig_epoch_done.connect(self._on_epoch_done)
        self._train_worker.sig_progress.connect(self.progress_bar.setValue)
        self._train_worker.sig_status.connect(self.status_label.setText)
        self._train_worker.sig_finished.connect(self._on_train_finished)
        self._train_worker.sig_error.connect(self._on_error)
        self._train_worker.finished.connect(self._on_worker_done)

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
        self.status_label.setText("Training complete. Click Localize.")

    def _on_worker_done(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        g.alert(msg)

    # --- Prediction ---

    def _start_prediction(self):
        # Load pre-trained if needed
        mode = self.mode_combo.currentIndex()
        if mode == 2:
            path = self.checkpoint_edit.text()
            if not path:
                g.alert("Select a checkpoint file first.")
                return
            from flika.ai.localizer_backends import create_backend
            config = self._build_config()
            self._backend = create_backend(config)
            self._backend.load(path)

        if self._backend is None or not self._backend.is_trained():
            g.alert("Train the model or load a checkpoint first.")
            return

        data = self._get_predict_data()
        if data is None:
            g.alert("No predict window selected.")
            return

        self._predict_worker = LocalizerPredictionWorker(
            self._backend, data,
            threshold=self.threshold_spin.value(),
            min_distance=self.min_dist_spin.value(),
            do_tracking=self.track_check.isChecked(),
            max_track_dist=self.track_dist_spin.value(),
            max_gap=self.track_gap_spin.value(),
            parent=self,
        )
        self._predict_worker.sig_result.connect(self._on_predict_result)
        self._predict_worker.sig_progress.connect(self.progress_bar.setValue)
        self._predict_worker.sig_status.connect(self.status_label.setText)
        self._predict_worker.sig_error.connect(self._on_error)
        self._predict_worker.finished.connect(
            lambda: self.localize_btn.setEnabled(True))

        self.localize_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._predict_worker.start()

    def _on_predict_result(self, density_maps, localizations):
        from flika.window import Window

        self._last_localizations = localizations
        src_name = self.predict_window_combo.currentText()

        # Create density map window
        density = np.asarray(density_maps, dtype=g.settings['internal_data_type'])
        win = Window(density, name=src_name + ' - Density')

        # Store localizations as metadata
        win.metadata['localizations'] = localizations

        if self.track_check.isChecked() and localizations.shape[1] > 3:
            win.metadata['tracks'] = localizations

        n_locs = len(localizations)
        self.status_label.setText(
            f"Localization complete — {n_locs} particles found.")

    # --- Export ---

    def _export_csv(self):
        if self._last_localizations is None or len(self._last_localizations) == 0:
            g.alert("No localizations to export. Run Localize first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Localizations", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        locs = self._last_localizations
        if locs.shape[1] == 3:
            header = "frame,y,x"
        elif locs.shape[1] == 4:
            header = "frame,y,x,track_id"
        else:
            header = ",".join([f"col{i}" for i in range(locs.shape[1])])

        np.savetxt(path, locs, delimiter=',', header=header, comments='',
                   fmt='%.4f')
        self.status_label.setText(f"Exported {len(locs)} localizations to {path}")

    # --- Cleanup ---

    def closeEvent(self, event):
        for worker in (self._train_worker, self._predict_worker, self._sim_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, 'request_stop'):
                    worker.request_stop()
                worker.wait(5000)
        super().closeEvent(event)
