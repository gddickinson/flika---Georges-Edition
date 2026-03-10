"""Acquisition engine for live camera feeds.

Provides:
  - ``AcquisitionWorker`` — QThread that grabs frames continuously
  - ``FrameBuffer`` — Thread-safe circular buffer with recording modes
  - ``FocusMetric`` — Real-time focus quality computation

Acquisition modes:
  - Live preview (display only, no recording)
  - Snap (single frame)
  - Record N frames
  - Time-lapse (interval between captures)
  - Triggered (external hardware trigger via backend)
  - Circular buffer (keep last N, save on demand / pre-trigger)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from qtpy import QtCore
from qtpy.QtCore import Signal

from flika.logger import logger


# ---------------------------------------------------------------------------
# Recording modes
# ---------------------------------------------------------------------------

class RecordingMode(Enum):
    PREVIEW = auto()       # Live display only
    SNAP = auto()          # Single frame capture
    RECORD_N = auto()      # Record fixed number of frames
    TIME_LAPSE = auto()    # Interval-based capture
    CIRCULAR = auto()      # Circular buffer, save on demand


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Thread-safe circular frame buffer.

    Pre-allocates a numpy array for zero-copy frame storage during
    continuous acquisition.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames to hold.
    height, width : int
        Frame dimensions.
    dtype : numpy dtype
        Frame data type (uint8, uint16, etc.).
    n_channels : int
        Number of color channels (1 for grayscale, 3 for RGB).
    """

    def __init__(self, max_frames: int, height: int, width: int,
                 dtype=np.uint8, n_channels: int = 1):
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.dtype = dtype
        self.n_channels = n_channels

        if n_channels > 1:
            self._buffer = np.zeros((max_frames, height, width, n_channels),
                                    dtype=dtype)
        else:
            self._buffer = np.zeros((max_frames, height, width), dtype=dtype)

        self._timestamps = np.zeros(max_frames, dtype=np.float64)
        self._write_idx = 0       # Next write position
        self._frame_count = 0     # Total frames written
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def filled(self) -> int:
        """Number of valid frames in buffer."""
        return min(self._frame_count, self.max_frames)

    def push(self, frame: np.ndarray, timestamp: float):
        """Add a frame to the buffer."""
        with self._lock:
            idx = self._write_idx % self.max_frames
            # Handle shape mismatch gracefully
            if frame.shape == self._buffer[idx].shape:
                self._buffer[idx] = frame
            else:
                # Resize frame to fit
                target_shape = self._buffer[idx].shape
                if frame.ndim == 2 and len(target_shape) == 2:
                    h = min(frame.shape[0], target_shape[0])
                    w = min(frame.shape[1], target_shape[1])
                    self._buffer[idx, :h, :w] = frame[:h, :w]
                elif frame.ndim == 3 and len(target_shape) == 3:
                    h = min(frame.shape[0], target_shape[0])
                    w = min(frame.shape[1], target_shape[1])
                    c = min(frame.shape[2], target_shape[2])
                    self._buffer[idx, :h, :w, :c] = frame[:h, :w, :c]

            self._timestamps[idx] = timestamp
            self._latest = self._buffer[idx]
            self._write_idx += 1
            self._frame_count += 1

    def get_latest(self) -> Optional[np.ndarray]:
        """Return the most recently pushed frame (no copy)."""
        with self._lock:
            return self._latest

    def get_latest_copy(self) -> Optional[np.ndarray]:
        """Return a copy of the most recently pushed frame."""
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def get_recorded(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frames_array, timestamps_array) for all valid frames.

        Frames are returned in chronological order.
        """
        with self._lock:
            n = self.filled
            if n == 0:
                return np.array([]), np.array([])

            if self._frame_count <= self.max_frames:
                # Buffer hasn't wrapped
                return self._buffer[:n].copy(), self._timestamps[:n].copy()
            else:
                # Buffer has wrapped — reorder chronologically
                start = self._write_idx % self.max_frames
                indices = np.roll(np.arange(self.max_frames), -start)
                return self._buffer[indices].copy(), self._timestamps[indices].copy()

    def clear(self):
        """Reset the buffer."""
        with self._lock:
            self._buffer[:] = 0
            self._timestamps[:] = 0
            self._write_idx = 0
            self._frame_count = 0
            self._latest = None


# ---------------------------------------------------------------------------
# Focus quality metrics
# ---------------------------------------------------------------------------

def focus_laplacian_variance(frame: np.ndarray) -> float:
    """Laplacian variance — the most reliable focus metric.

    Higher values = sharper image = better focus.
    """
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
    frame = frame.astype(np.float64)
    # Laplacian kernel
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.ndimage import convolve
    laplacian = convolve(frame, lap)
    return float(np.var(laplacian))


def focus_normalized_variance(frame: np.ndarray) -> float:
    """Normalized variance — fast focus metric."""
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
    mean_val = frame.mean()
    if mean_val == 0:
        return 0.0
    return float(np.var(frame.astype(np.float64)) / mean_val)


def focus_tenenbaum_gradient(frame: np.ndarray) -> float:
    """Tenenbaum gradient — sum of squared Sobel gradients."""
    if frame.ndim == 3:
        frame = frame.mean(axis=-1)
    frame = frame.astype(np.float64)
    from scipy.ndimage import sobel
    gx = sobel(frame, axis=0)
    gy = sobel(frame, axis=1)
    return float(np.mean(gx**2 + gy**2))


FOCUS_METRICS = {
    'Laplacian Variance': focus_laplacian_variance,
    'Normalized Variance': focus_normalized_variance,
    'Tenenbaum Gradient': focus_tenenbaum_gradient,
}


# ---------------------------------------------------------------------------
# Acquisition worker
# ---------------------------------------------------------------------------

class AcquisitionWorker(QtCore.QThread):
    """Background thread for continuous frame acquisition.

    Signals
    -------
    sigFrameReady(numpy.ndarray)
        Emitted when a new frame is available (for display update).
    sigFrameCount(int)
        Current frame count.
    sigFPS(float)
        Measured frames per second.
    sigFocusMetric(float)
        Focus quality value (computed every N frames).
    sigRecordingDone()
        Emitted when a fixed-count recording completes.
    sigError(str)
        Error message.
    """

    sigFrameReady = Signal(object)
    sigFrameCount = Signal(int)
    sigFPS = Signal(float)
    sigFocusMetric = Signal(float)
    sigRecordingDone = Signal()
    sigError = Signal(str)

    def __init__(self, backend, buffer: FrameBuffer, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.buffer = buffer

        self.mode = RecordingMode.PREVIEW
        self.target_frames = 0        # for RECORD_N
        self.interval_ms = 0          # for TIME_LAPSE
        self.compute_focus = False
        self.focus_metric_fn = focus_laplacian_variance
        self._focus_interval = 5      # compute every N frames

        self._running = False
        self._recording = False

    def start_preview(self):
        """Start live preview (display only)."""
        self.mode = RecordingMode.PREVIEW
        self._recording = False
        if not self.isRunning():
            self._running = True
            self.start()

    def start_recording(self, mode: RecordingMode, n_frames: int = 0,
                        interval_ms: float = 0):
        """Start recording frames to the buffer."""
        self.mode = mode
        self.target_frames = n_frames
        self.interval_ms = interval_ms
        self._recording = True
        self.buffer.clear()
        if not self.isRunning():
            self._running = True
            self.start()

    def stop(self):
        """Stop acquisition."""
        self._running = False
        self._recording = False

    def run(self):
        try:
            self.backend.start_continuous()
        except Exception as e:
            self.sigError.emit(f"Failed to start acquisition: {e}")
            return

        fps_counter = 0
        fps_time = time.time()
        frame_count = 0

        try:
            while self._running:
                # Time-lapse interval
                if (self._recording and
                        self.mode == RecordingMode.TIME_LAPSE and
                        self.interval_ms > 0 and frame_count > 0):
                    time.sleep(self.interval_ms / 1000.0)

                result = self.backend.grab_frame()
                if result is None:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue

                frame, timestamp = result
                frame_count += 1

                # Always emit for display
                self.sigFrameReady.emit(frame)
                self.sigFrameCount.emit(frame_count)

                # Record to buffer if recording
                if self._recording:
                    self.buffer.push(frame, timestamp)

                    # Check if done (RECORD_N or TIME_LAPSE)
                    if (self.mode in (RecordingMode.RECORD_N,
                                      RecordingMode.TIME_LAPSE,
                                      RecordingMode.SNAP) and
                            self.target_frames > 0 and
                            self.buffer.frame_count >= self.target_frames):
                        self._recording = False
                        self.sigRecordingDone.emit()
                        if self.mode == RecordingMode.SNAP:
                            self._running = False
                            break

                # FPS calculation (every second)
                fps_counter += 1
                now = time.time()
                elapsed = now - fps_time
                if elapsed >= 1.0:
                    self.sigFPS.emit(fps_counter / elapsed)
                    fps_counter = 0
                    fps_time = now

                # Focus metric
                if self.compute_focus and frame_count % self._focus_interval == 0:
                    try:
                        val = self.focus_metric_fn(frame)
                        self.sigFocusMetric.emit(val)
                    except Exception:
                        pass

        except Exception as e:
            self.sigError.emit(str(e))
        finally:
            try:
                self.backend.stop_continuous()
            except Exception:
                pass
