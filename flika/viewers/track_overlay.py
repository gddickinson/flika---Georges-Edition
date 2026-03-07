# -*- coding: utf-8 -*-
"""Track visualization overlay for particle tracking results.

Draws particle tracks on top of flika image windows, with configurable
tail length, color modes, and filtering by track length.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import QObject, Signal

from ..logger import logger


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def _hsv_cycle(n):
    """Return *n* distinct QColors by cycling through HSV hue."""
    colors = []
    for i in range(max(n, 1)):
        hue = int(255 * (i / max(n, 1))) % 256
        c = QtGui.QColor.fromHsv(hue, 220, 230)
        colors.append(c)
    return colors


def _map_values_to_colors(values, cmap_name='viridis'):
    """Map an array of floats to QColors using a pyqtgraph colormap.

    Parameters
    ----------
    values : array-like
        Values to map.  Will be normalized to [0, 1].
    cmap_name : str
        Name passed to ``pg.colormap.get``.  Falls back to a simple
        blue-green-yellow-red gradient if the name is unavailable.

    Returns
    -------
    list of QColor
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax - vmin < 1e-12:
        normed = np.zeros_like(values)
    else:
        normed = (values - vmin) / (vmax - vmin)

    try:
        cmap = pg.colormap.get(cmap_name)
    except Exception:
        # Fallback gradient: blue -> cyan -> yellow -> red
        positions = [0.0, 0.33, 0.66, 1.0]
        colors_rgb = [
            [50, 50, 200],
            [50, 200, 200],
            [220, 220, 50],
            [200, 50, 50],
        ]
        cmap = pg.ColorMap(positions, np.array(colors_rgb, dtype=np.ubyte))

    lut = cmap.mapToQColor(normed)
    return list(lut)


# ---------------------------------------------------------------------------
# TrackOverlay
# ---------------------------------------------------------------------------

class TrackOverlay(QObject):
    """Manages track visualization on a :class:`~flika.window.Window`.

    Tracks are stored as a dict mapping ``track_id`` to an ``(N, 3)`` numpy
    array whose columns are ``[frame, x, y]``.  On every frame change the
    overlay redraws the visible tail segments and current-position markers.
    """

    sigUpdated = Signal()

    def __init__(self, window):
        super().__init__()
        self.window = window

        # Track data  {track_id: ndarray (N, 3)  columns [frame, x, y]}
        self.tracks = {}

        # Display items currently in the ViewBox
        self._plot_items = []   # PlotCurveItem (trail lines)
        self._point_items = []  # ScatterPlotItem (current positions)

        # Display parameters
        self.tail_length = 10
        self.color_mode = 'track_id'
        self.visible = True
        self.min_track_length = 3

        # Pre-computed colors  {track_id: QColor}
        self._colors = {}

        # Column-based color mode state
        self._color_column = ''

        # Whether to display unlinked localizations (track_id == -1)
        self._show_unlinked = False

        # Connect to the window's time-change signal
        self._time_conn = self.window.sigTimeChanged.connect(self._on_time_changed)
        self._close_conn = self.window.closeSignal.connect(self.cleanup)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_tracks_from_array(self, data):
        """Load tracks from an array with columns ``[frame, x, y, track_id]``.

        Parameters
        ----------
        data : ndarray, shape (M, 4+)
            Each row is ``[frame, x, y, track_id, ...]``.
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError("Expected array with at least 4 columns "
                             "[frame, x, y, track_id]")
        tracks_dict = {}
        ids = np.unique(data[:, 3])
        for tid in ids:
            mask = data[:, 3] == tid
            segment = data[mask][:, :3].copy()  # [frame, x, y]
            # Sort by frame
            order = np.argsort(segment[:, 0])
            tracks_dict[int(tid)] = segment[order]
        self.load_tracks_from_dict(tracks_dict)

    def load_tracks_from_csv(self, path):
        """Load tracks from a CSV file.

        The CSV must have columns ``frame, x, y, track_id`` (header optional).

        Parameters
        ----------
        path : str
            Path to the CSV file.
        """
        try:
            data = np.genfromtxt(path, delimiter=',', skip_header=0,
                                 dtype=float)
            # If header row produced NaNs, skip it
            if np.any(np.isnan(data[0])):
                data = data[1:]
        except Exception as exc:
            logger.error("Failed to load tracks from %s: %s", path, exc)
            raise
        self.load_tracks_from_array(data)

    def load_tracks_from_dict(self, tracks_dict):
        """Load tracks directly from a dict.

        Parameters
        ----------
        tracks_dict : dict
            Mapping ``{track_id: ndarray (N, 3)}`` with columns
            ``[frame, x, y]``.
        """
        self.tracks = {}
        for tid, arr in tracks_dict.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                logger.warning("Skipping track %s: unexpected shape %s",
                               tid, arr.shape)
                continue
            self.tracks[int(tid)] = arr[:, :3].copy()

        logger.info("Loaded %d tracks", len(self.tracks))
        self.set_color_mode(self.color_mode)
        self.update_display()

    # ------------------------------------------------------------------
    # Color modes
    # ------------------------------------------------------------------

    def set_color_mode(self, mode):
        """Pre-compute colors for all tracks.

        Parameters
        ----------
        mode : str
            One of ``'track_id'``, ``'velocity'``, ``'displacement'``,
            ``'time'``, ``'classification'``, ``'column'``.
        """
        self.color_mode = mode
        tids = sorted(self.tracks.keys())

        if not tids:
            self._colors = {}
            return

        if mode == 'track_id':
            palette = _hsv_cycle(len(tids))
            self._colors = {tid: palette[i] for i, tid in enumerate(tids)}

        elif mode == 'velocity':
            values = []
            for tid in tids:
                pts = self.tracks[tid]
                if len(pts) < 2:
                    values.append(0.0)
                else:
                    dx = np.diff(pts[:, 1])
                    dy = np.diff(pts[:, 2])
                    dt = np.diff(pts[:, 0])
                    dt[dt == 0] = 1.0
                    speeds = np.sqrt(dx ** 2 + dy ** 2) / dt
                    values.append(float(np.nanmean(speeds)))
            qcolors = _map_values_to_colors(values)
            self._colors = {tid: qcolors[i] for i, tid in enumerate(tids)}

        elif mode == 'displacement':
            values = []
            for tid in tids:
                pts = self.tracks[tid]
                if len(pts) < 2:
                    values.append(0.0)
                else:
                    disp = np.sqrt((pts[-1, 1] - pts[0, 1]) ** 2 +
                                   (pts[-1, 2] - pts[0, 2]) ** 2)
                    values.append(float(disp))
            qcolors = _map_values_to_colors(values)
            self._colors = {tid: qcolors[i] for i, tid in enumerate(tids)}

        elif mode == 'time':
            values = []
            for tid in tids:
                pts = self.tracks[tid]
                values.append(float(np.mean(pts[:, 0])))
            qcolors = _map_values_to_colors(values)
            self._colors = {tid: qcolors[i] for i, tid in enumerate(tids)}

        elif mode == 'classification':
            # Color by SVM classification label from SPT metadata.
            # 1=Mobile (green), 2=Confined (yellow), 3=Trapped (red)
            _class_colors = {
                1: QtGui.QColor('#00CC00'),   # Mobile
                2: QtGui.QColor('#CCCC00'),   # Confined
                3: QtGui.QColor('#CC0000'),   # Trapped
            }
            default_color = QtGui.QColor(180, 180, 180)
            classification = (self.window.metadata.get('spt', {})
                              .get('classification', {}))
            self._colors = {}
            for tid in tids:
                label = classification.get(tid, None)
                self._colors[tid] = _class_colors.get(label, default_color)

        elif mode == 'column':
            # Color by an arbitrary feature column value via viridis colormap.
            features_by_track = (self.window.metadata.get('spt', {})
                                 .get('features_by_track', {}))
            col = self._color_column
            values = []
            valid_tids = []
            for tid in tids:
                feat = features_by_track.get(tid, {})
                val = feat.get(col, None)
                if val is not None:
                    values.append(float(val))
                    valid_tids.append(tid)
            if values:
                qcolors = _map_values_to_colors(values)
                self._colors = {tid: qcolors[i]
                                for i, tid in enumerate(valid_tids)}
            else:
                # No valid data — fall back to gray
                default_color = QtGui.QColor(180, 180, 180)
                self._colors = {tid: default_color for tid in tids}
            # Fill in any tids that had no feature value
            default_color = QtGui.QColor(180, 180, 180)
            for tid in tids:
                if tid not in self._colors:
                    self._colors[tid] = default_color

        else:
            logger.warning("Unknown color mode '%s', falling back to "
                           "'track_id'", mode)
            self.set_color_mode('track_id')
            return

        self.update_display()

    # ------------------------------------------------------------------
    # Display parameter setters
    # ------------------------------------------------------------------

    def set_tail_length(self, n):
        """Set the number of past frames shown as trailing lines."""
        self.tail_length = max(1, int(n))
        self.update_display()

    def set_min_track_length(self, n):
        """Set the minimum number of detections a track must have to be drawn."""
        self.min_track_length = max(1, int(n))
        self.update_display()

    def set_color_by_column(self, column_name):
        """Set color mode to map a feature column through a viridis colormap.

        Parameters
        ----------
        column_name : str
            Name of the feature column in the per-track feature dict
            (``window.metadata['spt']['features_by_track'][tid][column_name]``).
        """
        self.color_mode = 'column'
        self._color_column = column_name
        self.set_color_mode('column')

    def set_show_unlinked(self, show):
        """Toggle display of unlinked localizations (track_id == -1).

        Parameters
        ----------
        show : bool
        """
        self._show_unlinked = bool(show)
        self.update_display()

    def toggle_visibility(self):
        """Toggle visibility of all track overlay items."""
        self.visible = not self.visible
        for item in self._plot_items + self._point_items:
            item.setVisible(self.visible)

    # ------------------------------------------------------------------
    # Core drawing
    # ------------------------------------------------------------------

    def _on_time_changed(self, frame_idx):
        self.update_display(frame_idx)

    def update_display(self, frame_idx=None):
        """Redraw tracks for the given frame (or the window's current frame).

        This clears all previous overlay items and creates new ones for each
        track that falls within the current tail window.
        """
        view = self.window.imageview.view

        # Remove old items
        for item in self._plot_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        for item in self._point_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._plot_items.clear()
        self._point_items.clear()

        if not self.tracks or not self.visible:
            return

        if frame_idx is None:
            frame_idx = self.window.currentIndex

        f_lo = frame_idx - self.tail_length
        f_hi = frame_idx

        # Collect scatter points for a single batch item
        scatter_xs = []
        scatter_ys = []
        scatter_brushes = []

        for tid, pts in self.tracks.items():
            # Filter by minimum track length
            if len(pts) < self.min_track_length:
                continue

            # Find points in the tail window [f_lo, f_hi]
            mask = (pts[:, 0] >= f_lo) & (pts[:, 0] <= f_hi)
            if not np.any(mask):
                continue

            seg = pts[mask]
            color = self._colors.get(tid, QtGui.QColor(200, 200, 200))

            # Draw trail line if there are at least 2 points in the window
            # Tracks store [frame, x, y] where x=dim1 (pyqtgraph x)
            # and y=dim2 (pyqtgraph y) — no swap needed.
            if len(seg) >= 2:
                pen = pg.mkPen(color, width=2)
                curve = pg.PlotCurveItem(
                    x=seg[:, 1], y=seg[:, 2], pen=pen,
                )
                curve.setZValue(10)
                view.addItem(curve)
                self._plot_items.append(curve)

            # Mark current-frame position with a scatter dot
            at_current = seg[seg[:, 0] == frame_idx]
            if len(at_current) > 0:
                scatter_xs.append(at_current[0, 1])
                scatter_ys.append(at_current[0, 2])
                scatter_brushes.append(pg.mkBrush(color))

        # Add a single ScatterPlotItem with all current-frame markers
        if scatter_xs:
            scatter = pg.ScatterPlotItem(
                x=np.array(scatter_xs),
                y=np.array(scatter_ys),
                size=8,
                pen=pg.mkPen('w', width=1),
                brush=scatter_brushes,
            )
            scatter.setZValue(20)
            view.addItem(scatter)
            self._point_items.append(scatter)

        # Optionally show unlinked localizations as small gray dots
        if self._show_unlinked:
            localizations = (self.window.metadata.get('spt', {})
                             .get('localizations', None))
            if localizations is not None:
                localizations = np.asarray(localizations, dtype=float)
                # Expected columns: [frame, x, y, track_id, ...]
                if localizations.ndim == 2 and localizations.shape[1] >= 4:
                    mask = ((localizations[:, 3] == -1) &
                            (localizations[:, 0] == frame_idx))
                    unlinked = localizations[mask]
                    if len(unlinked) > 0:
                        gray_brush = pg.mkBrush(150, 150, 150, 120)
                        unlinked_scatter = pg.ScatterPlotItem(
                            x=unlinked[:, 1],
                            y=unlinked[:, 2],
                            size=4,
                            pen=pg.mkPen(None),
                            brush=gray_brush,
                        )
                        unlinked_scatter.setZValue(5)
                        view.addItem(unlinked_scatter)
                        self._point_items.append(unlinked_scatter)

        self.sigUpdated.emit()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove all overlay items and disconnect signals."""
        view = self.window.imageview.view
        for item in self._plot_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        for item in self._point_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._plot_items.clear()
        self._point_items.clear()

        try:
            self.window.sigTimeChanged.disconnect(self._on_time_changed)
        except (TypeError, RuntimeError):
            pass
        try:
            self.window.closeSignal.disconnect(self.cleanup)
        except (TypeError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# TrackControlPanel
# ---------------------------------------------------------------------------

class TrackControlPanel(QtWidgets.QDockWidget):
    """Dockable control panel for :class:`TrackOverlay` settings.

    Provides widgets for loading track data, choosing color mode, adjusting
    tail length, filtering short tracks, toggling visibility, and removing
    the overlay entirely.
    """

    def __init__(self, track_overlay, parent=None):
        super().__init__("Track Overlay", parent)
        self.overlay = track_overlay
        self.setObjectName("TrackOverlayDock")
        self.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Load Tracks button ---
        self.load_btn = QtWidgets.QPushButton("Load Tracks")
        self.load_btn.setToolTip("Load tracks from a CSV file "
                                 "(columns: frame, x, y, track_id)")
        self.load_btn.clicked.connect(self._on_load)
        layout.addWidget(self.load_btn)

        # --- Color mode ---
        color_label = QtWidgets.QLabel("Color mode:")
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(['track_id', 'velocity',
                                   'displacement', 'time',
                                   'classification', 'column'])
        self.color_combo.setCurrentText(self.overlay.color_mode)
        self.color_combo.currentTextChanged.connect(self._on_color_mode)
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(color_label)
        color_row.addWidget(self.color_combo)
        layout.addLayout(color_row)

        # --- Color column (used when color mode is 'column') ---
        col_label = QtWidgets.QLabel("Color column:")
        self.column_combo = QtWidgets.QComboBox()
        self.column_combo.setToolTip(
            "Feature column used for coloring when mode is 'column'.\n"
            "Populated automatically when track data with features is loaded.")
        self.column_combo.setEnabled(False)
        self.column_combo.currentTextChanged.connect(self._on_color_column)
        col_row = QtWidgets.QHBoxLayout()
        col_row.addWidget(col_label)
        col_row.addWidget(self.column_combo)
        layout.addLayout(col_row)

        # --- Tail length ---
        tail_label = QtWidgets.QLabel("Tail length:")
        self.tail_spin = QtWidgets.QSpinBox()
        self.tail_spin.setRange(1, 100)
        self.tail_spin.setValue(self.overlay.tail_length)
        self.tail_spin.setToolTip("Number of past frames to display as trail")
        self.tail_spin.valueChanged.connect(self._on_tail_length)
        tail_row = QtWidgets.QHBoxLayout()
        tail_row.addWidget(tail_label)
        tail_row.addWidget(self.tail_spin)
        layout.addLayout(tail_row)

        # --- Min track length ---
        minlen_label = QtWidgets.QLabel("Min track length:")
        self.minlen_spin = QtWidgets.QSpinBox()
        self.minlen_spin.setRange(1, 1000)
        self.minlen_spin.setValue(self.overlay.min_track_length)
        self.minlen_spin.setToolTip("Hide tracks shorter than this many "
                                    "detections")
        self.minlen_spin.valueChanged.connect(self._on_min_length)
        minlen_row = QtWidgets.QHBoxLayout()
        minlen_row.addWidget(minlen_label)
        minlen_row.addWidget(self.minlen_spin)
        layout.addLayout(minlen_row)

        # --- Visibility checkbox ---
        self.vis_check = QtWidgets.QCheckBox("Visible")
        self.vis_check.setChecked(self.overlay.visible)
        self.vis_check.toggled.connect(self._on_visibility)
        layout.addWidget(self.vis_check)

        # --- Show unlinked localizations ---
        self.unlinked_check = QtWidgets.QCheckBox("Show Unlinked")
        self.unlinked_check.setChecked(self.overlay._show_unlinked)
        self.unlinked_check.setToolTip(
            "Display unlinked localizations (track_id == -1) as small "
            "gray dots on the current frame")
        self.unlinked_check.toggled.connect(self._on_show_unlinked)
        layout.addWidget(self.unlinked_check)

        # --- Remove Overlay button ---
        self.remove_btn = QtWidgets.QPushButton("Remove Overlay")
        self.remove_btn.setToolTip("Remove all tracks and close this panel")
        self.remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(self.remove_btn)

        layout.addStretch()
        self.setWidget(container)

        # Refresh column combo when tracks are updated
        self.overlay.sigUpdated.connect(self._try_refresh_columns)

    def _try_refresh_columns(self):
        """Refresh the column combo if there are feature columns available."""
        features_by_track = (self.overlay.window.metadata.get('spt', {})
                             .get('features_by_track', {}))
        if features_by_track and self.column_combo.count() == 0:
            self._refresh_column_combo()
            self.column_combo.setEnabled(
                self.overlay.color_mode == 'column')

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load(self):
        from .. import global_vars as g
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Track CSV", "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        if path:
            try:
                self.overlay.load_tracks_from_csv(path)
                logger.info("Loaded tracks from %s", path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Load Error",
                    f"Failed to load tracks:\n{exc}")

    def _on_color_mode(self, mode):
        # Enable/disable the column combo based on selected mode
        self.column_combo.setEnabled(mode == 'column')
        if mode == 'column':
            self._refresh_column_combo()
            col = self.column_combo.currentText()
            if col:
                self.overlay.set_color_by_column(col)
            else:
                self.overlay.set_color_mode(mode)
        else:
            self.overlay.set_color_mode(mode)

    def _on_color_column(self, column_name):
        if column_name and self.overlay.color_mode == 'column':
            self.overlay.set_color_by_column(column_name)

    def _on_show_unlinked(self, checked):
        self.overlay.set_show_unlinked(checked)

    def _refresh_column_combo(self):
        """Populate the color column combo from SPT feature data."""
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        features_by_track = (self.overlay.window.metadata.get('spt', {})
                             .get('features_by_track', {}))
        columns = set()
        for feat_dict in features_by_track.values():
            if isinstance(feat_dict, dict):
                columns.update(feat_dict.keys())
        for col in sorted(columns):
            self.column_combo.addItem(col)
        # Restore previous selection if still valid
        prev = self.overlay._color_column
        idx = self.column_combo.findText(prev)
        if idx >= 0:
            self.column_combo.setCurrentIndex(idx)
        self.column_combo.blockSignals(False)

    def _on_tail_length(self, value):
        self.overlay.set_tail_length(value)

    def _on_min_length(self, value):
        self.overlay.set_min_track_length(value)

    def _on_visibility(self, checked):
        if checked != self.overlay.visible:
            self.overlay.toggle_visibility()

    def _on_remove(self):
        self.overlay.cleanup()
        from .. import global_vars as g
        if g.m is not None:
            g.m.removeDockWidget(self)
        self.close()
        self.deleteLater()

    def closeEvent(self, event):
        """Ensure overlay is cleaned up when the panel is closed."""
        self.overlay.cleanup()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_track_overlay(window=None):
    """Create a :class:`TrackOverlay` on *window* and show the control panel.

    Parameters
    ----------
    window : :class:`~flika.window.Window`, optional
        The image window to overlay tracks on.  Defaults to ``g.win``.

    Returns
    -------
    tuple
        ``(TrackOverlay, TrackControlPanel)``
    """
    from .. import global_vars as g

    if window is None:
        window = g.win
    if window is None:
        logger.error("No active window for track overlay")
        raise RuntimeError("No active window.  Open an image first.")

    overlay = TrackOverlay(window)
    panel = TrackControlPanel(overlay, parent=g.m)

    if g.m is not None:
        g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
    panel.show()

    return overlay, panel
