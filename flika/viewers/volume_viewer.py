"""3D Volume Viewer using pyqtgraph's OpenGL widgets.

Renders the current time-slice of a 4D volume using ``GLVolumeItem``
with user-controlled opacity/threshold and three movable slicer planes.
"""
from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtWidgets

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except ImportError:
    _HAS_GL = False


def _normalize_volume(vol3d, threshold_pct=0, opacity=0.5):
    """Convert a 3D array to RGBA uint8 suitable for GLVolumeItem.

    Parameters
    ----------
    vol3d : ndarray (X, Y, Z)
    threshold_pct : float 0-100, voxels below this percentile are transparent
    opacity : float 0-1, global alpha multiplier

    Returns
    -------
    rgba : ndarray (X, Y, Z, 4) uint8
    """
    v = vol3d.astype(np.float64)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmax - vmin == 0:
        v = np.zeros_like(v)
    else:
        v = (v - vmin) / (vmax - vmin)

    thresh = np.percentile(v, threshold_pct) if threshold_pct > 0 else 0
    alpha = np.where(v > thresh, v * opacity, 0.0)

    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (v * 255).astype(np.uint8)
    rgba[..., 1] = (v * 255).astype(np.uint8)
    rgba[..., 2] = (v * 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return rgba


class VolumeViewer(QtWidgets.QWidget):
    """Interactive 3D volume renderer for 4D flika windows.

    Uses ``pyqtgraph.opengl.GLViewWidget`` and ``GLVolumeItem``.
    """

    def __init__(self, parent_window):
        super().__init__()
        if not _HAS_GL:
            raise ImportError("pyqtgraph.opengl is required for the 3D volume viewer")

        self.parent_window = parent_window
        self.setWindowTitle(f'3D Volume: {parent_window.name}')
        self.resize(700, 600)

        # --- GL view ---
        self.glview = gl.GLViewWidget()
        self.glview.setCameraPosition(distance=200)

        # Grid + axes
        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)
        self.glview.addItem(grid)

        ax = gl.GLAxisItem()
        ax.setSize(50, 50, 50)
        self.glview.addItem(ax)

        # Volume item (filled on rebuild)
        self._vol_item = None

        # Slicer planes
        self._xy_plane = None
        self._xz_plane = None
        self._yz_plane = None

        # --- Controls ---
        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._opacity_slider.setRange(1, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(self._rebuild_volume)

        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._threshold_slider.setRange(0, 99)
        self._threshold_slider.setValue(20)
        self._threshold_slider.valueChanged.connect(self._rebuild_volume)

        self._reset_view_btn = QtWidgets.QPushButton('Reset View (Top-Down XY)')
        self._reset_view_btn.clicked.connect(self._reset_camera)

        ctrl_layout = QtWidgets.QFormLayout()
        ctrl_layout.addRow('Opacity', self._opacity_slider)
        ctrl_layout.addRow('Threshold %', self._threshold_slider)
        ctrl_layout.addRow(self._reset_view_btn)

        hint = QtWidgets.QLabel("Hold C in the image window to position the slicer planes")
        hint.setStyleSheet("color: #888; font-style: italic; padding: 2px;")
        hint.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.glview, stretch=1)
        layout.addLayout(ctrl_layout)

        # --- Signals ---
        parent_window.sigTimeChanged.connect(lambda _: self._rebuild_volume())
        for dim_idx, (slider, _label) in parent_window._dim_sliders.items():
            slider.valueChanged.connect(lambda _: self._rebuild_volume())

        self._crosshair_x = parent_window.mx // 2
        self._crosshair_y = parent_window.my // 2

        self._gl_ready = False

    # ------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        if not self._gl_ready:
            self._gl_ready = True
            # Defer first build so the OpenGL context is fully initialised
            QtCore.QTimer.singleShot(0, self._rebuild_volume)

    # ------------------------------------------------------------------
    def set_crosshair(self, x, y):
        """Called from the main window's mouseMoved to update slicer planes."""
        self._crosshair_x = int(np.clip(x, 0, self.parent_window.mx - 1))
        self._crosshair_y = int(np.clip(y, 0, self.parent_window.my - 1))
        self._update_slicer_planes()

    # ------------------------------------------------------------------
    def _rebuild_volume(self):
        if not self._gl_ready:
            return
        pw = self.parent_window
        vol = pw.volume
        if vol is None or vol.ndim != 4:
            return

        t = min(pw.currentIndex, vol.shape[0] - 1)
        sub = vol[t]  # shape (X, Y, Z)

        opacity = self._opacity_slider.value() / 100.0
        threshold = self._threshold_slider.value()

        rgba = _normalize_volume(sub, threshold, opacity)

        # Remove old volume
        if self._vol_item is not None:
            self.glview.removeItem(self._vol_item)
        self._vol_item = gl.GLVolumeItem(rgba, sliceDensity=1, smooth=True)
        self.glview.addItem(self._vol_item)

        self._update_slicer_planes()

    # ------------------------------------------------------------------
    def _update_slicer_planes(self):
        """Move the three coloured slicer planes to current positions."""
        if not self._gl_ready:
            return
        pw = self.parent_window
        vol = pw.volume
        if vol is None or vol.ndim != 4:
            return

        mx, my, mz = vol.shape[1], vol.shape[2], vol.shape[3]

        # Current Z from dimension slider
        z_pos = 0
        if 3 in pw._dim_sliders:
            z_pos = pw._dim_sliders[3][0].value()

        # --- XY plane (red) at current Z ---
        self._replace_plane('_xy_plane', self._make_quad(
            [[0, 0, z_pos], [mx, 0, z_pos], [mx, my, z_pos], [0, my, z_pos]],
            color=(1, 0, 0, 0.3)
        ))

        # --- XZ plane (green) at current Y ---
        y = min(self._crosshair_y, my - 1)
        self._replace_plane('_xz_plane', self._make_quad(
            [[0, y, 0], [mx, y, 0], [mx, y, mz], [0, y, mz]],
            color=(0, 1, 0, 0.3)
        ))

        # --- YZ plane (blue) at current X ---
        x = min(self._crosshair_x, mx - 1)
        self._replace_plane('_yz_plane', self._make_quad(
            [[x, 0, 0], [x, my, 0], [x, my, mz], [x, 0, mz]],
            color=(0, 0, 1, 0.3)
        ))

    def _replace_plane(self, attr, new_item):
        old = getattr(self, attr)
        if old is not None:
            self.glview.removeItem(old)
        setattr(self, attr, new_item)
        if new_item is not None:
            self.glview.addItem(new_item)

    @staticmethod
    def _make_quad(corners, color):
        """Create a GLMeshItem quad from 4 corner points."""
        verts = np.array(corners, dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        colors = np.array([color, color], dtype=np.float32)
        return gl.GLMeshItem(
            vertexes=verts, faces=faces, faceColors=colors,
            smooth=False, drawEdges=False,
        )

    # ------------------------------------------------------------------
    def _reset_camera(self):
        """Reset the camera to a top-down XY orientation matching the main window."""
        pw = self.parent_window
        vol = pw.volume
        if vol is not None and vol.ndim == 4:
            max_dim = max(vol.shape[1], vol.shape[2], vol.shape[3])
        else:
            max_dim = max(pw.mx, pw.my)
        self.glview.setCameraPosition(distance=max_dim * 2, elevation=90, azimuth=0)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.parent_window._volume_viewer = None
        event.accept()
