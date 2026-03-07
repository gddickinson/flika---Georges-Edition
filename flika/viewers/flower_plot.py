# -*- coding: utf-8 -*-
"""Origin-centered track display ("flower plot") for SPT visualization.

All tracks are plotted with their start position translated to the origin
(0, 0), revealing the spatial distribution, extent, and directionality
of particle motion independent of absolute position.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from ..logger import logger
from .track_overlay import _hsv_cycle, _map_values_to_colors


# ---------------------------------------------------------------------------
# FlowerPlotWindow
# ---------------------------------------------------------------------------

class FlowerPlotWindow(QtWidgets.QWidget):
    """All tracks displayed with start positions at origin.

    Shows track spatial distribution patterns -- useful for visualizing
    the extent and directionality of particle motion.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flower Plot")
        self.resize(700, 600)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Data
        self._tracks = {}            # {track_id: (N,3) [frame, x, y]}
        self._classification = {}    # {track_id: int/str label}
        self._features = {}          # {track_id: {col: value, ...}}
        self._centered_tracks = {}   # {track_id: (N,2) [dx, dy]}

        # Build UI
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # --- Plot widget (left, stretches) ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'dx', units='px')
        self.plot_widget.setLabel('left', 'dy', units='px')
        self.plot_widget.addLine(x=0, pen=pg.mkPen('w', width=0.5, style=QtCore.Qt.DashLine))
        self.plot_widget.addLine(y=0, pen=pg.mkPen('w', width=0.5, style=QtCore.Qt.DashLine))
        main_layout.addWidget(self.plot_widget, stretch=1)

        # --- Controls panel (right) ---
        ctrl_widget = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(6)

        # Color mode
        ctrl_layout.addWidget(QtWidgets.QLabel("Color mode:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems([
            'track_id', 'displacement', 'classification', 'velocity',
        ])
        self.color_combo.currentTextChanged.connect(self._on_settings_changed)
        ctrl_layout.addWidget(self.color_combo)

        # Min track length
        ctrl_layout.addWidget(QtWidgets.QLabel("Min track length:"))
        self.min_length_spin = QtWidgets.QSpinBox()
        self.min_length_spin.setRange(1, 100000)
        self.min_length_spin.setValue(3)
        self.min_length_spin.setToolTip(
            "Only display tracks with at least this many detections")
        self.min_length_spin.valueChanged.connect(self._on_settings_changed)
        ctrl_layout.addWidget(self.min_length_spin)

        # Show endpoints checkbox
        self.show_endpoints_check = QtWidgets.QCheckBox("Show endpoints")
        self.show_endpoints_check.setChecked(False)
        self.show_endpoints_check.setToolTip(
            "Draw a marker at the last point of each centered track")
        self.show_endpoints_check.toggled.connect(self._on_settings_changed)
        ctrl_layout.addWidget(self.show_endpoints_check)

        # Show mean endpoint checkbox
        self.show_mean_endpoint_check = QtWidgets.QCheckBox("Show mean endpoint")
        self.show_mean_endpoint_check.setChecked(False)
        self.show_mean_endpoint_check.setToolTip(
            "Draw an X marker at the mean of all track endpoints")
        self.show_mean_endpoint_check.toggled.connect(self._on_settings_changed)
        ctrl_layout.addWidget(self.show_mean_endpoint_check)

        # Background sigma circles
        self.sigma_circles_check = QtWidgets.QCheckBox("Sigma circles")
        self.sigma_circles_check.setChecked(False)
        self.sigma_circles_check.setToolTip(
            "Draw concentric circles at 1, 2, 3 sigma of endpoint distances")
        self.sigma_circles_check.toggled.connect(self._on_settings_changed)
        ctrl_layout.addWidget(self.sigma_circles_check)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl_layout.addWidget(line)

        # Export button
        self.export_btn = QtWidgets.QPushButton("Export PNG")
        self.export_btn.setToolTip("Save the current plot as a PNG image")
        self.export_btn.clicked.connect(self._on_export)
        ctrl_layout.addWidget(self.export_btn)

        ctrl_layout.addStretch()

        ctrl_widget.setFixedWidth(180)
        main_layout.addWidget(ctrl_widget)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_tracks(self, tracks_dict, classification=None, features=None):
        """Load track data for display.

        Parameters
        ----------
        tracks_dict : dict
            Mapping ``{track_id: ndarray (N, 3)}`` with columns
            ``[frame, x, y]``.
        classification : dict, optional
            Mapping ``{track_id: label}`` for classification coloring.
        features : dict, optional
            Mapping ``{track_id: {column: value, ...}}`` for feature-based
            coloring (e.g. velocity, MSD slope, etc.).
        """
        self._tracks = {}
        for tid, arr in tracks_dict.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                logger.warning("Flower plot: skipping track %s (shape %s)",
                               tid, arr.shape)
                continue
            self._tracks[int(tid)] = arr[:, :3].copy()

        self._classification = dict(classification) if classification else {}
        self._features = dict(features) if features else {}

        # Pre-compute centered tracks (translate start to origin)
        self._centered_tracks = {}
        for tid, arr in self._tracks.items():
            if arr.shape[0] < 1:
                continue
            x0, y0 = arr[0, 1], arr[0, 2]
            centered = np.column_stack([
                arr[:, 1] - x0,
                arr[:, 2] - y0,
            ])
            self._centered_tracks[tid] = centered

        logger.info("Flower plot: loaded %d tracks (%d after centering)",
                     len(self._tracks), len(self._centered_tracks))
        self._update_plot()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filtered_track_ids(self):
        """Return sorted list of track IDs passing the min-length filter."""
        min_len = self.min_length_spin.value()
        return sorted(
            tid for tid, arr in self._tracks.items()
            if arr.shape[0] >= min_len and tid in self._centered_tracks
        )

    def _compute_colors(self, tids):
        """Compute per-track QColors based on the current color mode.

        Returns
        -------
        dict
            ``{track_id: QColor}``
        """
        mode = self.color_combo.currentText()

        if not tids:
            return {}

        if mode == 'track_id':
            palette = _hsv_cycle(len(tids))
            return {tid: palette[i] for i, tid in enumerate(tids)}

        elif mode == 'displacement':
            values = []
            for tid in tids:
                pts = self._centered_tracks[tid]
                # Displacement is distance from origin to last point
                disp = np.sqrt(pts[-1, 0] ** 2 + pts[-1, 1] ** 2)
                values.append(float(disp))
            qcolors = _map_values_to_colors(values)
            return {tid: qcolors[i] for i, tid in enumerate(tids)}

        elif mode == 'velocity':
            values = []
            for tid in tids:
                arr = self._tracks[tid]
                if arr.shape[0] < 2:
                    values.append(0.0)
                    continue
                dx = np.diff(arr[:, 1])
                dy = np.diff(arr[:, 2])
                dt = np.diff(arr[:, 0])
                dt[dt == 0] = 1.0
                speeds = np.sqrt(dx ** 2 + dy ** 2) / dt
                values.append(float(np.nanmean(speeds)))
            qcolors = _map_values_to_colors(values)
            return {tid: qcolors[i] for i, tid in enumerate(tids)}

        elif mode == 'classification':
            _class_colors = {
                1: QtGui.QColor('#00CC00'),   # Mobile
                2: QtGui.QColor('#CCCC00'),   # Confined
                3: QtGui.QColor('#CC0000'),   # Trapped
            }
            # Also support string labels
            _class_colors_str = {
                'mobile': QtGui.QColor('#00CC00'),
                'confined': QtGui.QColor('#CCCC00'),
                'trapped': QtGui.QColor('#CC0000'),
            }
            default_color = QtGui.QColor(180, 180, 180)
            colors = {}
            for tid in tids:
                label = self._classification.get(tid, None)
                if label is None:
                    colors[tid] = default_color
                elif isinstance(label, str):
                    colors[tid] = _class_colors_str.get(
                        label.lower(), default_color)
                else:
                    colors[tid] = _class_colors.get(label, default_color)
            return colors

        else:
            # Fallback: HSV cycle
            palette = _hsv_cycle(len(tids))
            return {tid: palette[i] for i, tid in enumerate(tids)}

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _update_plot(self):
        """Clear and redraw all centered tracks with current settings."""
        self.plot_widget.clear()

        # Re-add cross-hair lines at origin
        self.plot_widget.addLine(
            x=0, pen=pg.mkPen('w', width=0.5, style=QtCore.Qt.DashLine))
        self.plot_widget.addLine(
            y=0, pen=pg.mkPen('w', width=0.5, style=QtCore.Qt.DashLine))

        tids = self._filtered_track_ids()
        if not tids:
            return

        colors = self._compute_colors(tids)

        # Draw each centered track as a line from origin
        for tid in tids:
            pts = self._centered_tracks[tid]
            if pts.shape[0] < 2:
                continue
            color = colors.get(tid, QtGui.QColor(200, 200, 200))
            pen = pg.mkPen(color, width=1.5)
            self.plot_widget.plot(pts[:, 0], pts[:, 1], pen=pen)

        # Optional: endpoint markers
        if self.show_endpoints_check.isChecked():
            self._draw_endpoints(tids, colors)

        # Optional: mean endpoint marker
        if self.show_mean_endpoint_check.isChecked():
            self._draw_mean_endpoint(tids)

        # Optional: sigma circles
        if self.sigma_circles_check.isChecked():
            self._draw_sigma_circles(tids)

    def _draw_endpoints(self, tids, colors):
        """Draw scatter markers at the last point of each centered track."""
        xs, ys, brushes = [], [], []
        for tid in tids:
            pts = self._centered_tracks[tid]
            if pts.shape[0] < 1:
                continue
            xs.append(pts[-1, 0])
            ys.append(pts[-1, 1])
            color = colors.get(tid, QtGui.QColor(200, 200, 200))
            brushes.append(pg.mkBrush(color))

        if xs:
            scatter = pg.ScatterPlotItem(
                x=np.array(xs), y=np.array(ys),
                size=6,
                pen=pg.mkPen('w', width=0.8),
                brush=brushes,
            )
            scatter.setZValue(10)
            self.plot_widget.addItem(scatter)

    def _draw_mean_endpoint(self, tids):
        """Draw an X marker at the mean endpoint position."""
        endpoints = []
        for tid in tids:
            pts = self._centered_tracks[tid]
            if pts.shape[0] < 1:
                continue
            endpoints.append([pts[-1, 0], pts[-1, 1]])

        if not endpoints:
            return

        endpoints = np.array(endpoints)
        mean_x = np.mean(endpoints[:, 0])
        mean_y = np.mean(endpoints[:, 1])

        scatter = pg.ScatterPlotItem(
            x=[mean_x], y=[mean_y],
            size=16,
            symbol='x',
            pen=pg.mkPen('#FF4444', width=2.5),
            brush=pg.mkBrush(None),
        )
        scatter.setZValue(20)
        self.plot_widget.addItem(scatter)

    def _draw_sigma_circles(self, tids):
        """Draw concentric circles at 1, 2, 3 sigma of endpoint distances.

        Sigma is computed as the standard deviation of the Euclidean distance
        from the origin to each track's endpoint.
        """
        distances = []
        for tid in tids:
            pts = self._centered_tracks[tid]
            if pts.shape[0] < 1:
                continue
            d = np.sqrt(pts[-1, 0] ** 2 + pts[-1, 1] ** 2)
            distances.append(d)

        if len(distances) < 2:
            return

        sigma = float(np.std(distances))
        if sigma < 1e-9:
            return

        # Draw circles at 1x, 2x, 3x sigma
        n_points = 120
        theta = np.linspace(0, 2 * np.pi, n_points)

        sigma_labels = [1, 2, 3]
        alphas = [120, 80, 50]

        for mult, alpha in zip(sigma_labels, alphas):
            r = sigma * mult
            cx = r * np.cos(theta)
            cy = r * np.sin(theta)
            pen = pg.mkPen(QtGui.QColor(180, 180, 255, alpha), width=1.2,
                           style=QtCore.Qt.DotLine)
            curve = pg.PlotCurveItem(x=cx, y=cy, pen=pen)
            curve.setZValue(-5)
            self.plot_widget.addItem(curve)

            # Add label at top of circle
            label = pg.TextItem(
                text=f"{mult}\u03c3",
                color=QtGui.QColor(180, 180, 255, alpha + 40),
                anchor=(0.5, 1.0),
            )
            label.setPos(0, r)
            label.setZValue(-4)
            self.plot_widget.addItem(label)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_settings_changed(self, *args):
        """Redraw when any control widget changes."""
        self._update_plot()

    def _on_export(self):
        """Export the current plot as a PNG file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Flower Plot", "flower_plot.png",
            "PNG Images (*.png);;All Files (*)")
        if not path:
            return

        try:
            exporter = pg.exporters.ImageExporter(
                self.plot_widget.plotItem)
            exporter.parameters()['width'] = 1200
            exporter.export(path)
            logger.info("Flower plot exported to %s", path)
        except Exception as exc:
            logger.error("Failed to export flower plot: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Export Error",
                f"Failed to export plot:\n{exc}")


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_flower_plot(window=None):
    """Create and show a :class:`FlowerPlotWindow` populated from a window's
    track overlay data.

    Parameters
    ----------
    window : :class:`~flika.window.Window`, optional
        The image window whose track overlay data should be used.
        Defaults to ``g.win``.

    Returns
    -------
    FlowerPlotWindow
        The flower plot window instance.
    """
    from .. import global_vars as g

    if window is None:
        window = g.win
    if window is None:
        logger.error("No active window for flower plot")
        raise RuntimeError("No active window.  Open an image first.")

    # Gather track data from the window's SPT metadata or track overlay
    tracks_dict = None
    classification = None
    features = None

    # Check for track overlay attached to the window
    spt_meta = getattr(window, 'metadata', {}).get('spt', {})
    if spt_meta:
        # Build tracks dict from stored localizations if available
        tracks_raw = spt_meta.get('tracks', None)
        if isinstance(tracks_raw, dict):
            tracks_dict = tracks_raw
        classification = spt_meta.get('classification', None)
        features = spt_meta.get('features_by_track', None)

    # Alternatively, look for a TrackOverlay object
    if tracks_dict is None:
        overlay = getattr(window, '_track_overlay', None)
        if overlay is not None and hasattr(overlay, 'tracks'):
            tracks_dict = overlay.tracks

    if not tracks_dict:
        logger.error("No track data found on the active window")
        raise RuntimeError(
            "No track data found.  Load tracks onto the window first.")

    fp = FlowerPlotWindow(parent=None)
    fp.set_tracks(tracks_dict, classification=classification,
                  features=features)
    fp.show()

    return fp
