"""Live Camera dialog for flika.

Full-featured camera control panel with:
  - Camera discovery and connection
  - Live view with auto-contrast and histogram
  - Dynamic camera property controls
  - Multiple recording modes (snap, record N, time-lapse, circular buffer)
  - Focus quality meter with live plot
  - Stream recorded frames to a flika Window
  - Frame rate and exposure controls
  - Triggered acquisition support

Follows the pattern of other flika dialogs (classifier_dialog.py,
detection_dialog.py).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Signal
import pyqtgraph as pg

import flika.global_vars as g
from flika.logger import logger
from flika.camera.backends import (CameraBackend, CameraInfo, CameraProperty,
                       OpenCVBackend, MicroManagerBackend,
                       discover_cameras, create_backend)
from flika.camera.acquisition import (AcquisitionWorker, FrameBuffer, RecordingMode,
                          FOCUS_METRICS)


# ---------------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------------

class CameraDialog(QtWidgets.QDialog):
    """Live camera feed dialog with controls and recording."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Live Camera')
        self.resize(1000, 750)

        self._backend: Optional[CameraBackend] = None
        self._cameras: list[CameraInfo] = []
        self._worker: Optional[AcquisitionWorker] = None
        self._buffer: Optional[FrameBuffer] = None
        self._display_timer = QtCore.QTimer(self)
        self._display_timer.setInterval(33)  # ~30 fps display
        self._display_timer.timeout.connect(self._update_display)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._fps = 0.0
        self._recording = False

        # Focus metric history
        self._focus_history: list[float] = []
        self._max_focus_history = 200

        # Property widgets (dynamically created)
        self._prop_widgets: dict[str, QtWidgets.QWidget] = {}

        self._build_ui()
        self._discover_cameras()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left: live view
        left = QtWidgets.QVBoxLayout()

        # Camera connection bar
        conn_row = QtWidgets.QHBoxLayout()
        conn_row.addWidget(QtWidgets.QLabel('Camera:'))
        self._camera_combo = QtWidgets.QComboBox()
        self._camera_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                         QtWidgets.QSizePolicy.Fixed)
        conn_row.addWidget(self._camera_combo)
        self._btn_refresh = QtWidgets.QPushButton('Refresh')
        self._btn_refresh.clicked.connect(self._discover_cameras)
        conn_row.addWidget(self._btn_refresh)
        self._btn_connect = QtWidgets.QPushButton('Connect')
        self._btn_connect.clicked.connect(self._connect_camera)
        conn_row.addWidget(self._btn_connect)
        self._btn_disconnect = QtWidgets.QPushButton('Disconnect')
        self._btn_disconnect.clicked.connect(self._disconnect_camera)
        self._btn_disconnect.setEnabled(False)
        conn_row.addWidget(self._btn_disconnect)
        left.addLayout(conn_row)

        # Live view
        self._live_view = pg.ImageView()
        self._live_view.ui.roiBtn.hide()
        self._live_view.ui.menuBtn.hide()
        left.addWidget(self._live_view, stretch=4)

        # Status bar
        status_row = QtWidgets.QHBoxLayout()
        self._status_fps = QtWidgets.QLabel('FPS: --')
        self._status_fps.setMinimumWidth(80)
        status_row.addWidget(self._status_fps)
        self._status_frames = QtWidgets.QLabel('Frames: 0')
        self._status_frames.setMinimumWidth(100)
        status_row.addWidget(self._status_frames)
        self._status_res = QtWidgets.QLabel('Resolution: --')
        status_row.addWidget(self._status_res)
        self._status_recording = QtWidgets.QLabel('')
        self._status_recording.setStyleSheet('color: red; font-weight: bold;')
        status_row.addWidget(self._status_recording)
        status_row.addStretch()
        left.addLayout(status_row)

        # Focus meter
        focus_group = QtWidgets.QGroupBox('Focus Quality')
        focus_layout = QtWidgets.QVBoxLayout(focus_group)
        focus_top = QtWidgets.QHBoxLayout()
        self._focus_check = QtWidgets.QCheckBox('Enable')
        self._focus_check.toggled.connect(self._toggle_focus)
        focus_top.addWidget(self._focus_check)
        self._focus_metric_combo = QtWidgets.QComboBox()
        self._focus_metric_combo.addItems(list(FOCUS_METRICS.keys()))
        focus_top.addWidget(self._focus_metric_combo)
        self._focus_value = QtWidgets.QLabel('--')
        self._focus_value.setMinimumWidth(100)
        focus_top.addWidget(self._focus_value)
        focus_layout.addLayout(focus_top)
        self._focus_plot = pg.PlotWidget()
        self._focus_plot.setMaximumHeight(100)
        self._focus_plot.setLabel('left', 'Focus')
        self._focus_plot.hideAxis('bottom')
        self._focus_curve = self._focus_plot.plot(pen='g')
        focus_layout.addWidget(self._focus_plot)
        left.addWidget(focus_group)

        main_layout.addLayout(left, stretch=3)

        # Right: controls
        right = QtWidgets.QVBoxLayout()

        # Scroll area for controls
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll_widget = QtWidgets.QWidget()
        self._controls_layout = QtWidgets.QVBoxLayout(scroll_widget)

        # Quick controls (always visible)
        quick_group = QtWidgets.QGroupBox('Quick Controls')
        quick_layout = QtWidgets.QFormLayout(quick_group)

        self._exposure_spin = QtWidgets.QDoubleSpinBox()
        self._exposure_spin.setRange(0.001, 60000)
        self._exposure_spin.setDecimals(3)
        self._exposure_spin.setValue(10.0)
        self._exposure_spin.setSuffix(' ms')
        self._exposure_spin.valueChanged.connect(
            lambda v: self._set_property('exposure', v))
        quick_layout.addRow('Exposure:', self._exposure_spin)

        self._gain_spin = QtWidgets.QDoubleSpinBox()
        self._gain_spin.setRange(0, 1000)
        self._gain_spin.setDecimals(1)
        self._gain_spin.setValue(0)
        self._gain_spin.valueChanged.connect(
            lambda v: self._set_property('gain', v))
        quick_layout.addRow('Gain:', self._gain_spin)

        self._binning_combo = QtWidgets.QComboBox()
        self._binning_combo.addItems(['1x1', '2x2', '4x4', '8x8'])
        quick_layout.addRow('Binning:', self._binning_combo)

        self._auto_contrast = QtWidgets.QCheckBox('Auto-contrast')
        self._auto_contrast.setChecked(True)
        quick_layout.addRow('', self._auto_contrast)

        self._controls_layout.addWidget(quick_group)

        # Device properties (populated dynamically)
        self._device_props_group = QtWidgets.QGroupBox('Device Properties')
        self._device_props_layout = QtWidgets.QFormLayout(self._device_props_group)
        self._controls_layout.addWidget(self._device_props_group)

        self._controls_layout.addStretch()
        scroll.setWidget(scroll_widget)
        right.addWidget(scroll, stretch=3)

        # Recording controls
        rec_group = QtWidgets.QGroupBox('Recording')
        rec_layout = QtWidgets.QVBoxLayout(rec_group)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel('Mode:'))
        self._rec_mode = QtWidgets.QComboBox()
        self._rec_mode.addItems([
            'Snap (1 frame)', 'Record N Frames',
            'Time-Lapse', 'Circular Buffer',
        ])
        self._rec_mode.currentIndexChanged.connect(self._rec_mode_changed)
        mode_row.addWidget(self._rec_mode)
        rec_layout.addLayout(mode_row)

        param_form = QtWidgets.QFormLayout()
        self._rec_frames = QtWidgets.QSpinBox()
        self._rec_frames.setRange(1, 1000000)
        self._rec_frames.setValue(100)
        param_form.addRow('Frames:', self._rec_frames)

        self._rec_interval = QtWidgets.QDoubleSpinBox()
        self._rec_interval.setRange(0, 3600000)
        self._rec_interval.setDecimals(1)
        self._rec_interval.setValue(100)
        self._rec_interval.setSuffix(' ms')
        param_form.addRow('Interval:', self._rec_interval)

        self._rec_buffer_size = QtWidgets.QSpinBox()
        self._rec_buffer_size.setRange(10, 100000)
        self._rec_buffer_size.setValue(1000)
        param_form.addRow('Buffer Size:', self._rec_buffer_size)

        rec_layout.addLayout(param_form)

        # Recording progress
        self._rec_progress = QtWidgets.QProgressBar()
        self._rec_progress.setVisible(False)
        rec_layout.addWidget(self._rec_progress)

        # Recording buttons
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_live = QtWidgets.QPushButton('Live')
        self._btn_live.clicked.connect(self._start_live)
        self._btn_live.setEnabled(False)
        btn_row.addWidget(self._btn_live)

        self._btn_record = QtWidgets.QPushButton('Record')
        self._btn_record.clicked.connect(self._start_recording)
        self._btn_record.setEnabled(False)
        btn_row.addWidget(self._btn_record)

        self._btn_stop = QtWidgets.QPushButton('Stop')
        self._btn_stop.clicked.connect(self._stop_acquisition)
        self._btn_stop.setEnabled(False)
        btn_row.addWidget(self._btn_stop)
        rec_layout.addLayout(btn_row)

        btn_row2 = QtWidgets.QHBoxLayout()
        self._btn_send = QtWidgets.QPushButton('Send to Window')
        self._btn_send.clicked.connect(self._send_to_window)
        self._btn_send.setEnabled(False)
        btn_row2.addWidget(self._btn_send)

        self._btn_save_buffer = QtWidgets.QPushButton('Save Buffer')
        self._btn_save_buffer.clicked.connect(self._save_buffer)
        self._btn_save_buffer.setEnabled(False)
        btn_row2.addWidget(self._btn_save_buffer)
        rec_layout.addLayout(btn_row2)

        right.addWidget(rec_group)
        main_layout.addLayout(right, stretch=1)

        self._rec_mode_changed(0)

    # -----------------------------------------------------------------------
    # Camera discovery & connection
    # -----------------------------------------------------------------------

    def _discover_cameras(self):
        self._camera_combo.clear()
        self._cameras = discover_cameras()
        if not self._cameras:
            self._camera_combo.addItem('No cameras found')
            return
        for cam in self._cameras:
            self._camera_combo.addItem(cam.name, cam)

    def _connect_camera(self):
        idx = self._camera_combo.currentIndex()
        if idx < 0 or idx >= len(self._cameras):
            g.alert('No camera selected.')
            return

        cam_info = self._cameras[idx]

        # Check for required library
        if cam_info.backend == 'opencv':
            try:
                import cv2  # noqa: F401
            except ImportError:
                g.alert("opencv-python is not installed.\n\n"
                        "Install with:  pip install opencv-python")
                return
        elif cam_info.backend == 'micromanager':
            try:
                import pymmcore_plus  # noqa: F401
            except ImportError:
                g.alert("pymmcore-plus is not installed.\n\n"
                        "Install with:  pip install pymmcore-plus")
                return

        try:
            self._backend = create_backend(cam_info)
            self._backend.open(cam_info.id)
        except Exception as e:
            g.alert(f'Failed to connect: {e}')
            self._backend = None
            return

        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)
        self._btn_live.setEnabled(True)
        self._btn_record.setEnabled(True)

        # Update resolution display
        w, h = self._backend.frame_size()
        self._status_res.setText(f'Resolution: {w}x{h}')

        # Create buffer
        self._create_buffer(w, h)

        # Populate device properties
        self._populate_properties()

        self.setWindowTitle(f'Live Camera — {cam_info.name}')

    def _disconnect_camera(self):
        self._stop_acquisition()
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None

        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._btn_live.setEnabled(False)
        self._btn_record.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._clear_properties()
        self.setWindowTitle('Live Camera')

    def _create_buffer(self, width: int, height: int):
        """Create or resize the frame buffer."""
        buf_size = self._rec_buffer_size.value()

        # Determine dtype and channels from a test frame
        dtype = np.uint8
        n_channels = 1
        if self._backend and self._backend.is_open():
            try:
                frame, _ = self._backend.snap()
                dtype = frame.dtype
                n_channels = frame.shape[2] if frame.ndim == 3 else 1
                height, width = frame.shape[:2]
            except Exception:
                pass

        self._buffer = FrameBuffer(
            max_frames=buf_size, height=height, width=width,
            dtype=dtype, n_channels=n_channels)

    # -----------------------------------------------------------------------
    # Property controls
    # -----------------------------------------------------------------------

    def _populate_properties(self):
        """Build property controls from the camera backend."""
        self._clear_properties()
        if self._backend is None:
            return

        props = self._backend.get_properties()
        for prop in props:
            widget = self._make_property_widget(prop)
            if widget is not None:
                self._device_props_layout.addRow(prop.name + ':', widget)
                self._prop_widgets[prop.name] = widget

    def _clear_properties(self):
        while self._device_props_layout.rowCount() > 0:
            self._device_props_layout.removeRow(0)
        self._prop_widgets.clear()

    def _make_property_widget(self, prop: CameraProperty) -> Optional[QtWidgets.QWidget]:
        """Create an appropriate widget for a camera property."""
        if prop.read_only:
            label = QtWidgets.QLabel(str(prop.value))
            label.setStyleSheet('color: gray;')
            return label

        if prop.prop_type == 'bool':
            cb = QtWidgets.QCheckBox()
            cb.setChecked(bool(prop.value))
            cb.toggled.connect(
                lambda v, n=prop.name: self._set_property(n, v))
            return cb

        elif prop.prop_type == 'enum' and prop.choices:
            combo = QtWidgets.QComboBox()
            combo.addItems(prop.choices)
            idx = combo.findText(str(prop.value))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentTextChanged.connect(
                lambda v, n=prop.name: self._set_property(n, v))
            return combo

        elif prop.prop_type == 'int':
            spin = QtWidgets.QSpinBox()
            if prop.min_val is not None:
                spin.setMinimum(int(prop.min_val))
            if prop.max_val is not None:
                spin.setMaximum(int(prop.max_val))
            if prop.step is not None:
                spin.setSingleStep(int(prop.step))
            try:
                spin.setValue(int(prop.value))
            except (ValueError, TypeError):
                pass
            spin.valueChanged.connect(
                lambda v, n=prop.name: self._set_property(n, v))
            return spin

        elif prop.prop_type == 'float':
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            if prop.min_val is not None:
                spin.setMinimum(float(prop.min_val))
            if prop.max_val is not None:
                spin.setMaximum(float(prop.max_val))
            if prop.step is not None:
                spin.setSingleStep(float(prop.step))
            try:
                spin.setValue(float(prop.value))
            except (ValueError, TypeError):
                pass
            spin.valueChanged.connect(
                lambda v, n=prop.name: self._set_property(n, v))
            return spin

        elif prop.prop_type == 'string':
            le = QtWidgets.QLineEdit(str(prop.value))
            le.editingFinished.connect(
                lambda n=prop.name, w=le: self._set_property(n, w.text()))
            return le

        return None

    def _set_property(self, name: str, value):
        """Set a camera property via the backend."""
        if self._backend is None or not self._backend.is_open():
            return
        try:
            self._backend.set_property(name, value)
        except Exception as e:
            logger.warning("Failed to set %s = %s: %s", name, value, e)

    # -----------------------------------------------------------------------
    # Live view & display
    # -----------------------------------------------------------------------

    def _start_live(self):
        if self._backend is None or not self._backend.is_open():
            return

        # Ensure buffer exists
        if self._buffer is None:
            w, h = self._backend.frame_size()
            self._create_buffer(w, h)

        self._worker = AcquisitionWorker(
            self._backend, self._buffer, parent=self)
        self._worker.sigFrameReady.connect(self._on_frame_ready)
        self._worker.sigFrameCount.connect(self._on_frame_count)
        self._worker.sigFPS.connect(self._on_fps)
        self._worker.sigFocusMetric.connect(self._on_focus_metric)
        self._worker.sigError.connect(self._on_error)
        self._worker.sigRecordingDone.connect(self._on_recording_done)

        # Configure focus
        self._toggle_focus(self._focus_check.isChecked())

        self._worker.start_preview()
        self._display_timer.start()

        self._btn_live.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status_recording.setText('')

    def _start_recording(self):
        if self._backend is None or not self._backend.is_open():
            return

        mode_idx = self._rec_mode.currentIndex()
        n_frames = self._rec_frames.value()
        interval = self._rec_interval.value()

        if mode_idx == 0:
            mode = RecordingMode.SNAP
            n_frames = 1
        elif mode_idx == 1:
            mode = RecordingMode.RECORD_N
        elif mode_idx == 2:
            mode = RecordingMode.TIME_LAPSE
        elif mode_idx == 3:
            mode = RecordingMode.CIRCULAR
            n_frames = 0  # Unlimited
        else:
            mode = RecordingMode.RECORD_N

        # Resize buffer if needed
        if mode != RecordingMode.CIRCULAR:
            w, h = self._backend.frame_size()
            buf_size = max(n_frames, 10)
            self._buffer = None
            self._create_buffer(w, h)
            self._buffer.max_frames = buf_size
            # Reallocate with correct size
            dtype = self._buffer.dtype
            nc = self._buffer.n_channels
            if nc > 1:
                self._buffer._buffer = np.zeros(
                    (buf_size, h, w, nc), dtype=dtype)
            else:
                self._buffer._buffer = np.zeros(
                    (buf_size, h, w), dtype=dtype)
            self._buffer._timestamps = np.zeros(buf_size, dtype=np.float64)

        self._recording = True
        self._rec_progress.setVisible(mode != RecordingMode.CIRCULAR)
        if mode in (RecordingMode.RECORD_N, RecordingMode.TIME_LAPSE,
                    RecordingMode.SNAP):
            self._rec_progress.setRange(0, n_frames)
            self._rec_progress.setValue(0)

        # Stop existing worker if running
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)

        self._worker = AcquisitionWorker(
            self._backend, self._buffer, parent=self)
        self._worker.sigFrameReady.connect(self._on_frame_ready)
        self._worker.sigFrameCount.connect(self._on_frame_count)
        self._worker.sigFPS.connect(self._on_fps)
        self._worker.sigFocusMetric.connect(self._on_focus_metric)
        self._worker.sigError.connect(self._on_error)
        self._worker.sigRecordingDone.connect(self._on_recording_done)
        self._toggle_focus(self._focus_check.isChecked())

        self._worker.start_recording(mode, n_frames, interval)
        self._display_timer.start()

        self._btn_live.setEnabled(False)
        self._btn_record.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_send.setEnabled(False)
        self._btn_save_buffer.setEnabled(False)
        self._status_recording.setText('RECORDING')

    def _stop_acquisition(self):
        self._display_timer.stop()
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        self._recording = False
        if self._backend and self._backend.is_open():
            self._btn_live.setEnabled(True)
            self._btn_record.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._rec_progress.setVisible(False)
        self._status_recording.setText('')

        # Enable send/save if we have frames
        has_frames = self._buffer is not None and self._buffer.frame_count > 0
        self._btn_send.setEnabled(has_frames)
        self._btn_save_buffer.setEnabled(has_frames)

    def _on_frame_ready(self, frame: np.ndarray):
        """Store latest frame for display timer."""
        self._latest_frame = frame

    def _update_display(self):
        """Called by display timer to update the live view."""
        frame = self._latest_frame
        if frame is None:
            return

        # Transpose for pyqtgraph (expects [x, y] or [x, y, c])
        if frame.ndim == 2:
            display = frame.T
        elif frame.ndim == 3:
            display = np.transpose(frame, (1, 0, 2))
        else:
            display = frame

        auto = self._auto_contrast.isChecked()
        self._live_view.setImage(display, autoLevels=auto,
                                 autoRange=False, autoHistogramRange=auto)

    def _on_frame_count(self, count: int):
        self._frame_count = count
        self._status_frames.setText(f'Frames: {count}')
        if self._recording and self._rec_progress.isVisible():
            self._rec_progress.setValue(
                min(count, self._rec_progress.maximum()))

    def _on_fps(self, fps: float):
        self._fps = fps
        self._status_fps.setText(f'FPS: {fps:.1f}')

    def _on_focus_metric(self, value: float):
        self._focus_value.setText(f'{value:.1f}')
        self._focus_history.append(value)
        if len(self._focus_history) > self._max_focus_history:
            self._focus_history = self._focus_history[-self._max_focus_history:]
        self._focus_curve.setData(self._focus_history)

    def _on_recording_done(self):
        """Called when a fixed-count recording completes."""
        self._status_recording.setText('DONE')
        self._recording = False
        self._btn_record.setEnabled(True)
        self._btn_send.setEnabled(True)
        self._btn_save_buffer.setEnabled(True)
        self._rec_progress.setVisible(False)

        # Auto-send snap to window
        if self._rec_mode.currentIndex() == 0:  # Snap
            self._send_to_window()

    def _on_error(self, msg: str):
        self._status_recording.setText(f'Error: {msg}')
        logger.error("Camera error: %s", msg)
        self._stop_acquisition()

    def _toggle_focus(self, enabled: bool):
        if self._worker is not None:
            self._worker.compute_focus = enabled
            metric_name = self._focus_metric_combo.currentText()
            self._worker.focus_metric_fn = FOCUS_METRICS.get(
                metric_name, list(FOCUS_METRICS.values())[0])
        self._focus_plot.setVisible(enabled)
        if not enabled:
            self._focus_value.setText('--')

    # -----------------------------------------------------------------------
    # Recording mode UI
    # -----------------------------------------------------------------------

    def _rec_mode_changed(self, idx: int):
        # 0=Snap, 1=Record N, 2=Time-Lapse, 3=Circular
        self._rec_frames.setVisible(idx in (1, 2))
        self._rec_interval.setVisible(idx == 2)
        self._rec_buffer_size.setVisible(idx == 3)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def _send_to_window(self):
        """Send recorded frames to a new flika Window."""
        if self._buffer is None or self._buffer.frame_count == 0:
            g.alert('No frames to send.')
            return

        frames, timestamps = self._buffer.get_recorded()
        if len(frames) == 0:
            g.alert('No frames recorded.')
            return

        from flika.window import Window

        if len(frames) == 1:
            image = frames[0]
        else:
            image = frames

        # Determine if RGB
        is_rgb = image.ndim == 3 and image.shape[-1] == 3
        if image.ndim == 4 and image.shape[-1] == 3:
            is_rgb = True

        win = Window(image, name='Camera Capture')
        win.metadata['is_rgb'] = is_rgb
        win.metadata['timestamps'] = timestamps.tolist()
        win.metadata['camera'] = {
            'backend': self._backend.backend_name() if self._backend else '',
            'fps': self._fps,
        }

        # Set framerate from timestamps
        if len(timestamps) > 1:
            dt = np.median(np.diff(timestamps))
            if dt > 0:
                win.framerate = 1.0 / dt

        self._status_recording.setText(
            f'Sent {len(frames)} frames to "{win.name}"')

    def _save_buffer(self):
        """Save the buffer contents to a file."""
        if self._buffer is None or self._buffer.frame_count == 0:
            g.alert('No frames to save.')
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Recording', 'camera_recording.tif',
            'TIFF (*.tif *.tiff);;NumPy (*.npy);;All Files (*)')
        if not path:
            return

        frames, timestamps = self._buffer.get_recorded()

        if path.endswith('.npy'):
            np.save(path, frames)
        else:
            try:
                import tifffile
                tifffile.imwrite(path, frames)
            except ImportError:
                np.save(path.rsplit('.', 1)[0] + '.npy', frames)
                g.alert('tifffile not available, saved as .npy')
                return

        self._status_recording.setText(f'Saved {len(frames)} frames to {path}')

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def closeEvent(self, event):
        self._stop_acquisition()
        self._disconnect_camera()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def show_camera_dialog(parent=None):
    """Show the live camera dialog."""
    dlg = CameraDialog(parent=parent)
    dlg.show()
    g.dialogs.append(dlg)
    return dlg
