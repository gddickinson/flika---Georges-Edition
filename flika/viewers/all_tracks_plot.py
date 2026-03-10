# -*- coding: utf-8 -*-
"""Multi-track intensity extraction and visualization window.

Ported from locsAndTracksPlotter's ``allTracksPlotter.py``.  Extracts
intensity values at tracked-particle positions from the source image and
presents three synchronized views:

1. All individual intensity traces (overlaid or vertically stacked).
2. Mean intensity trace with standard-deviation envelope.
3. Heatmap of all traces (rows = tracks, columns = frames).
"""
import csv
import os

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

from flika.logger import logger
from .track_overlay import _hsv_cycle


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_EXTRACTION_METHODS = ['Point', 'Mean 3x3', 'Mean 5x5', 'Max 3x3']

_SORT_KEYS = ['Track ID', 'Start Frame', 'Mean Intensity', 'Track Length']


def _neighborhood_value(image_2d, x, y, method):
    """Extract a single intensity value from *image_2d* at (*x*, *y*).

    Parameters
    ----------
    image_2d : ndarray (H, W)
        Single frame of the image.
    x, y : float
        Sub-pixel coordinates.  Rounded to nearest integer.
    method : str
        One of ``'Point'``, ``'Mean 3x3'``, ``'Mean 5x5'``, ``'Max 3x3'``.

    Returns
    -------
    float
        The extracted intensity value, or ``np.nan`` if the coordinate
        falls outside the image.
    """
    h, w = image_2d.shape
    ix = int(round(x))
    iy = int(round(y))

    if method == 'Point':
        if 0 <= iy < h and 0 <= ix < w:
            return float(image_2d[iy, ix])
        return np.nan

    # Determine half-size for neighbourhood extraction
    if method in ('Mean 3x3', 'Max 3x3'):
        half = 1
    elif method == 'Mean 5x5':
        half = 2
    else:
        # Unknown method — fall back to single pixel
        if 0 <= iy < h and 0 <= ix < w:
            return float(image_2d[iy, ix])
        return np.nan

    y0 = max(iy - half, 0)
    y1 = min(iy + half + 1, h)
    x0 = max(ix - half, 0)
    x1 = min(ix + half + 1, w)

    if y0 >= y1 or x0 >= x1:
        return np.nan

    patch = image_2d[y0:y1, x0:x1]

    if method.startswith('Mean'):
        return float(np.mean(patch))
    elif method == 'Max 3x3':
        return float(np.max(patch))
    return np.nan


# ---------------------------------------------------------------------------
# AllTracksPlotWindow
# ---------------------------------------------------------------------------

class AllTracksPlotWindow(QtWidgets.QWidget):
    """Intensity extraction and visualization for all tracks.

    Extracts intensity at track positions from the source image and displays:

    - **Plot 1** — All intensity traces stacked/overlaid.
    - **Plot 2** — Mean intensity trace with std envelope.
    - **Plot 3** — Heatmap of all tracks' intensities (rows=tracks, cols=frames).

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("All Tracks – Intensity Plot")
        self.setMinimumSize(820, 600)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Data ---------------------------------------------------------
        self._window = None          # source flika Window
        self._tracks = {}            # {track_id: (N,3) array}
        self._intensities = {}       # {track_id: (N,) array}
        self._sorted_ids = []        # sorted list of track IDs

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # --- Plots (left side) ----------------------------------------
        plot_splitter = QtWidgets.QSplitter(Qt.Vertical)

        self.plot_traces = pg.PlotWidget(title="Individual Traces")
        self.plot_traces.setLabel('bottom', 'Frame')
        self.plot_traces.setLabel('left', 'Intensity')
        self.plot_traces.addLegend(offset=(30, 10))

        self.plot_mean = pg.PlotWidget(title="Mean Intensity")
        self.plot_mean.setLabel('bottom', 'Relative Frame')
        self.plot_mean.setLabel('left', 'Intensity')

        self.plot_heatmap = pg.PlotWidget(title="Intensity Heatmap")
        self.plot_heatmap.setLabel('bottom', 'Relative Frame')
        self.plot_heatmap.setLabel('left', 'Track Index')

        plot_splitter.addWidget(self.plot_traces)
        plot_splitter.addWidget(self.plot_mean)
        plot_splitter.addWidget(self.plot_heatmap)
        plot_splitter.setSizes([200, 200, 200])

        main_layout.addWidget(plot_splitter, stretch=3)

        # --- Controls (right side) ------------------------------------
        ctrl = QtWidgets.QWidget()
        ctrl.setFixedWidth(200)
        cl = QtWidgets.QVBoxLayout(ctrl)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(6)

        # Extraction method
        cl.addWidget(QtWidgets.QLabel("Extraction method:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(_EXTRACTION_METHODS)
        self.method_combo.setCurrentText('Point')
        self.method_combo.setToolTip(
            "How intensity is sampled at each track position.\n"
            "'Point' uses the single nearest pixel.\n"
            "'Mean NxN' averages an NxN neighborhood.\n"
            "'Max 3x3' takes the max of a 3x3 neighborhood."
        )
        cl.addWidget(self.method_combo)

        # Sort by
        cl.addWidget(QtWidgets.QLabel("Sort by:"))
        self.sort_combo = QtWidgets.QComboBox()
        self.sort_combo.addItems(_SORT_KEYS)
        self.sort_combo.setCurrentText('Track ID')
        self.sort_combo.setToolTip("Ordering for individual traces and heatmap rows.")
        cl.addWidget(self.sort_combo)

        # Show individual traces
        self.show_individual_cb = QtWidgets.QCheckBox("Show individual traces")
        self.show_individual_cb.setChecked(True)
        self.show_individual_cb.setToolTip(
            "Toggle visibility of individual traces in the top plot.")
        cl.addWidget(self.show_individual_cb)

        # Highlight track
        hl_row = QtWidgets.QHBoxLayout()
        hl_row.addWidget(QtWidgets.QLabel("Highlight track:"))
        self.highlight_spin = QtWidgets.QSpinBox()
        self.highlight_spin.setRange(-1, 999999)
        self.highlight_spin.setValue(-1)
        self.highlight_spin.setSpecialValueText("None")
        self.highlight_spin.setToolTip(
            "Highlight one specific track in red on the traces plot.\n"
            "Set to -1 (None) to disable highlighting.")
        hl_row.addWidget(self.highlight_spin)
        cl.addLayout(hl_row)

        # Normalize
        self.normalize_cb = QtWidgets.QCheckBox("Normalize [0, 1]")
        self.normalize_cb.setChecked(False)
        self.normalize_cb.setToolTip(
            "Normalize each intensity trace to [0, 1] before plotting.")
        cl.addWidget(self.normalize_cb)

        cl.addSpacing(12)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        cl.addWidget(separator)
        cl.addSpacing(6)

        # Export CSV
        self.export_csv_btn = QtWidgets.QPushButton("Export CSV")
        self.export_csv_btn.setToolTip("Export all intensity traces to a CSV file.")
        cl.addWidget(self.export_csv_btn)

        # Export Plot
        self.export_plot_btn = QtWidgets.QPushButton("Export Plot")
        self.export_plot_btn.setToolTip("Save the current plots as a PNG image.")
        cl.addWidget(self.export_plot_btn)

        cl.addStretch()
        main_layout.addWidget(ctrl, stretch=0)

    def _connect_signals(self):
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self.show_individual_cb.toggled.connect(self._on_display_changed)
        self.highlight_spin.valueChanged.connect(self._on_display_changed)
        self.normalize_cb.toggled.connect(self._on_display_changed)
        self.export_csv_btn.clicked.connect(self._export_csv)
        self.export_plot_btn.clicked.connect(self._export_plot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, window, tracks_dict):
        """Load track data and extract intensities from the source window.

        Parameters
        ----------
        window : :class:`~flika.window.Window`
            Source flika window whose ``.image`` attribute provides the
            intensity data (shape ``(T, H, W)`` or ``(H, W)``).
        tracks_dict : dict
            Mapping ``{track_id: ndarray (N, 3)}`` with columns
            ``[frame, x, y]``.
        """
        self._window = window
        self._tracks = {}
        for tid, arr in tracks_dict.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                logger.warning("AllTracksPlot: skipping track %s, shape %s",
                               tid, arr.shape)
                continue
            self._tracks[int(tid)] = arr[:, :3].copy()

        if not self._tracks:
            logger.warning("AllTracksPlot: no valid tracks provided")
            return

        self._intensities = self._extract_intensities(
            window, self._tracks)
        self._compute_sort_order()
        self._update_all()

    # ------------------------------------------------------------------
    # Intensity extraction
    # ------------------------------------------------------------------

    def _extract_intensities(self, window, tracks_dict):
        """Extract intensities from *window* at track positions.

        Parameters
        ----------
        window : flika Window
            Must have an ``.image`` attribute with shape ``(T, H, W)``
            or ``(H, W)``.
        tracks_dict : dict
            ``{track_id: (N, 3) ndarray}`` with columns ``[frame, x, y]``.

        Returns
        -------
        dict
            ``{track_id: (N,) ndarray}`` of extracted intensity values.
        """
        method = self.method_combo.currentText()
        image = np.asarray(window.image, dtype=float)
        is_3d = image.ndim >= 3  # (T, H, W) or more

        result = {}
        n_tracks = len(tracks_dict)
        for idx, (tid, pts) in enumerate(tracks_dict.items()):
            vals = np.empty(len(pts), dtype=float)
            for j, row in enumerate(pts):
                frame = int(round(row[0]))
                x = row[1]
                y = row[2]
                if is_3d:
                    if 0 <= frame < image.shape[0]:
                        frame_img = image[frame]
                    else:
                        vals[j] = np.nan
                        continue
                else:
                    # Single 2-D frame — ignore frame index
                    frame_img = image

                vals[j] = _neighborhood_value(frame_img, x, y, method)
            result[tid] = vals

            # Progress logging for large datasets
            if n_tracks > 100 and (idx + 1) % 100 == 0:
                logger.info("AllTracksPlot: extracted %d / %d tracks",
                            idx + 1, n_tracks)

        logger.info("AllTracksPlot: extracted intensities for %d tracks "
                    "using '%s'", len(result), method)
        return result

    # ------------------------------------------------------------------
    # Sort order
    # ------------------------------------------------------------------

    def _compute_sort_order(self):
        """Recompute ``self._sorted_ids`` based on the current sort key."""
        key = self.sort_combo.currentText()
        tids = list(self._tracks.keys())

        if key == 'Track ID':
            self._sorted_ids = sorted(tids)

        elif key == 'Start Frame':
            self._sorted_ids = sorted(
                tids, key=lambda t: float(self._tracks[t][0, 0]))

        elif key == 'Mean Intensity':
            self._sorted_ids = sorted(
                tids,
                key=lambda t: float(np.nanmean(self._intensities.get(t, [0]))))

        elif key == 'Track Length':
            self._sorted_ids = sorted(
                tids, key=lambda t: len(self._tracks[t]))

        else:
            self._sorted_ids = sorted(tids)

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _update_all(self):
        """Refresh all three plots."""
        self._update_traces_plot()
        self._update_mean_plot()
        self._update_heatmap()

    def _get_display_intensities(self):
        """Return intensities dict, optionally normalized to [0, 1].

        Returns
        -------
        dict
            ``{track_id: (N,) ndarray}``
        """
        if not self.normalize_cb.isChecked():
            return self._intensities

        normed = {}
        for tid, vals in self._intensities.items():
            v = vals.copy()
            vmin = np.nanmin(v)
            vmax = np.nanmax(v)
            rng = vmax - vmin
            if rng > 1e-12:
                v = (v - vmin) / rng
            else:
                v = np.zeros_like(v)
            normed[tid] = v
        return normed

    # --- Plot 1: individual traces ------------------------------------

    def _update_traces_plot(self):
        """Redraw the individual-traces plot (Plot 1)."""
        self.plot_traces.clear()
        if not self._sorted_ids:
            return

        show = self.show_individual_cb.isChecked()
        highlight_id = self.highlight_spin.value()
        intens = self._get_display_intensities()

        n = len(self._sorted_ids)
        colors = _hsv_cycle(n)

        for i, tid in enumerate(self._sorted_ids):
            if tid not in self._tracks or tid not in intens:
                continue
            frames = self._tracks[tid][:, 0]
            vals = intens[tid]

            is_highlight = (tid == highlight_id)
            if is_highlight:
                pen = pg.mkPen(color='r', width=3)
                z_val = 100
            else:
                pen = pg.mkPen(color=colors[i], width=1)
                z_val = 10

            if show or is_highlight:
                curve = self.plot_traces.plot(
                    frames, vals, pen=pen)
                curve.setZValue(z_val)

    # --- Plot 2: mean +/- std ----------------------------------------

    def _update_mean_plot(self):
        """Redraw the mean-intensity plot with std envelope (Plot 2).

        All traces are aligned to relative time (frame - start_frame).
        """
        self.plot_mean.clear()
        if not self._sorted_ids:
            return

        intens = self._get_display_intensities()

        # Align traces by relative frame
        aligned = []
        for tid in self._sorted_ids:
            if tid not in self._tracks or tid not in intens:
                continue
            frames = self._tracks[tid][:, 0]
            vals = intens[tid]
            start = frames[0]
            rel = frames - start
            aligned.append((rel, vals))

        if not aligned:
            return

        # Find the maximum relative frame across all traces
        max_rel = int(max(np.nanmax(rel) for rel, _ in aligned))

        # Build a 2-D array padded with NaN for computing statistics
        n_traces = len(aligned)
        matrix = np.full((n_traces, max_rel + 1), np.nan, dtype=float)
        for i, (rel, vals) in enumerate(aligned):
            for j in range(len(rel)):
                col = int(round(rel[j]))
                if 0 <= col <= max_rel:
                    matrix[i, col] = vals[j]

        # Compute mean and std (ignoring NaN)
        with np.errstate(all='ignore'):
            mean_trace = np.nanmean(matrix, axis=0)
            std_trace = np.nanstd(matrix, axis=0)
            count_trace = np.sum(~np.isnan(matrix), axis=0)

        # Only plot frames with at least 2 contributing traces
        valid_mask = count_trace >= 2
        x_all = np.arange(max_rel + 1, dtype=float)

        if not np.any(valid_mask):
            # Fall back: plot whatever we have
            valid_mask = count_trace >= 1

        x = x_all[valid_mask]
        mu = mean_trace[valid_mask]
        sd = std_trace[valid_mask]
        sd = np.nan_to_num(sd)

        # Mean line
        mean_pen = pg.mkPen(color='w', width=2)
        self.plot_mean.plot(x, mu, pen=mean_pen, name='Mean')

        # Std envelope
        upper = mu + sd
        lower = mu - sd

        upper_curve = pg.PlotCurveItem(x, upper, pen=pg.mkPen(None))
        lower_curve = pg.PlotCurveItem(x, lower, pen=pg.mkPen(None))
        fill = pg.FillBetweenItem(upper_curve, lower_curve,
                                  brush=pg.mkBrush(255, 255, 255, 50))
        self.plot_mean.addItem(upper_curve)
        self.plot_mean.addItem(lower_curve)
        self.plot_mean.addItem(fill)

    # --- Plot 3: heatmap ----------------------------------------------

    def _update_heatmap(self):
        """Redraw the intensity heatmap (Plot 3).

        Rows correspond to tracks (in sort order), columns to relative
        frames.  Cells outside a track's lifetime are transparent / NaN.
        """
        self.plot_heatmap.clear()
        if not self._sorted_ids:
            return

        intens = self._get_display_intensities()

        # Determine max track length (relative frames)
        max_len = 0
        entries = []
        for tid in self._sorted_ids:
            if tid not in self._tracks or tid not in intens:
                continue
            frames = self._tracks[tid][:, 0]
            vals = intens[tid]
            start = frames[0]
            rel = frames - start
            length = int(np.nanmax(rel)) + 1
            if length > max_len:
                max_len = length
            entries.append((tid, rel, vals, length))

        if not entries or max_len == 0:
            return

        n_tracks = len(entries)
        matrix = np.full((n_tracks, max_len), np.nan, dtype=float)
        for i, (tid, rel, vals, _) in enumerate(entries):
            for j in range(len(rel)):
                col = int(round(rel[j]))
                if 0 <= col < max_len:
                    matrix[i, col] = vals[j]

        # Replace NaN with a value below the data range for visual clarity
        display_matrix = matrix.copy()
        valid = ~np.isnan(display_matrix)
        if np.any(valid):
            fill_val = np.nanmin(display_matrix) - 1.0
        else:
            fill_val = 0.0
        display_matrix[~valid] = fill_val

        # Create image item
        img_item = pg.ImageItem()
        img_item.setImage(display_matrix.T, autoLevels=True)

        # Set the transform so that x-axis = relative frame, y-axis = track index
        tr = QtGui.QTransform()
        # ImageItem draws with (col, row) → we transposed, so
        # image shape is (max_len, n_tracks).  We want:
        #   x-axis (horizontal) = relative frame  → image column → already col
        #   y-axis (vertical)   = track index     → image row    → already row
        img_item.setTransform(tr)

        self.plot_heatmap.addItem(img_item)
        self.plot_heatmap.setRange(
            xRange=[0, max_len],
            yRange=[0, n_tracks],
            padding=0.02)

        # Colorbar
        try:
            cmap = pg.colormap.get('viridis')
        except Exception:
            cmap = pg.colormap.get('CET-L1', source='colorcet')

        bar = pg.ColorBarItem(
            values=(np.nanmin(matrix[valid]) if np.any(valid) else 0,
                    np.nanmax(matrix[valid]) if np.any(valid) else 1),
            colorMap=cmap,
            interactive=False,
            width=15,
        )
        bar.setImageItem(img_item, insert_in=self.plot_heatmap.getPlotItem())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_method_changed(self, _text):
        """Re-extract intensities when the extraction method changes."""
        if self._window is None or not self._tracks:
            return
        self._intensities = self._extract_intensities(
            self._window, self._tracks)
        self._compute_sort_order()
        self._update_all()

    def _on_sort_changed(self, _text):
        """Reorder traces when the sort key changes."""
        self._compute_sort_order()
        self._update_all()

    def _on_display_changed(self, _value=None):
        """Refresh plots when a display option changes."""
        self._update_all()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self):
        """Write all intensity traces to a CSV file.

        Columns: ``track_id, frame, x, y, intensity``
        """
        if not self._intensities:
            logger.warning("AllTracksPlot: no data to export")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Intensity Traces", "",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['track_id', 'frame', 'x', 'y', 'intensity'])

                for tid in self._sorted_ids:
                    if tid not in self._tracks or tid not in self._intensities:
                        continue
                    pts = self._tracks[tid]
                    vals = self._intensities[tid]
                    for j in range(len(pts)):
                        writer.writerow([
                            tid,
                            pts[j, 0],
                            pts[j, 1],
                            pts[j, 2],
                            vals[j],
                        ])
            logger.info("AllTracksPlot: exported %d tracks to %s",
                        len(self._sorted_ids), path)
        except Exception as exc:
            logger.error("AllTracksPlot: CSV export failed: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Export Error",
                f"Failed to write CSV:\n{exc}")

    def _export_plot(self):
        """Save the three-plot layout as a PNG image."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Plot Image", "",
            "PNG Images (*.png);;All Files (*)")
        if not path:
            return

        try:
            # Grab pixmaps from each plot widget
            p1 = self.plot_traces.grab()
            p2 = self.plot_mean.grab()
            p3 = self.plot_heatmap.grab()

            total_h = p1.height() + p2.height() + p3.height()
            max_w = max(p1.width(), p2.width(), p3.width())

            combined = QtGui.QPixmap(max_w, total_h)
            combined.fill(QtGui.QColor(0, 0, 0))

            painter = QtGui.QPainter(combined)
            y_off = 0
            painter.drawPixmap(0, y_off, p1)
            y_off += p1.height()
            painter.drawPixmap(0, y_off, p2)
            y_off += p2.height()
            painter.drawPixmap(0, y_off, p3)
            painter.end()

            combined.save(path, 'PNG')
            logger.info("AllTracksPlot: exported plot to %s", path)
        except Exception as exc:
            logger.error("AllTracksPlot: plot export failed: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Export Error",
                f"Failed to save plot image:\n{exc}")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Clean up references on close."""
        self._window = None
        self._tracks.clear()
        self._intensities.clear()
        self._sorted_ids.clear()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_all_tracks_plot(window=None):
    """Create and show an :class:`AllTracksPlotWindow`.

    If the *window* has a :class:`~flika.viewers.track_overlay.TrackOverlay`
    attached (stored in ``window.metadata['spt']['overlay']``), its tracks
    are used automatically.  Otherwise the window's
    ``metadata['spt']['tracks']`` dict is tried.

    Parameters
    ----------
    window : :class:`~flika.window.Window`, optional
        Source image window.  Defaults to ``g.win``.

    Returns
    -------
    AllTracksPlotWindow
        The newly created plot window.

    Raises
    ------
    RuntimeError
        If no active window is available or no track data can be found.
    """
    import flika.global_vars as g

    if window is None:
        window = g.win
    if window is None:
        logger.error("AllTracksPlot: no active window")
        raise RuntimeError("No active window.  Open an image first.")

    # Try to find track data
    spt = window.metadata.get('spt', {}) if hasattr(window, 'metadata') else {}

    tracks_dict = None

    # 1. From an attached TrackOverlay
    overlay = spt.get('overlay', None)
    if overlay is not None and hasattr(overlay, 'tracks') and overlay.tracks:
        tracks_dict = overlay.tracks

    # 2. From metadata dict
    if tracks_dict is None:
        tracks_dict = spt.get('tracks', None)

    if tracks_dict is None or len(tracks_dict) == 0:
        logger.error("AllTracksPlot: no track data found on window")
        raise RuntimeError(
            "No track data found on the active window.\n"
            "Load tracks via the Track Overlay panel first.")

    plot_win = AllTracksPlotWindow(parent=None)
    plot_win.set_data(window, tracks_dict)
    plot_win.show()
    plot_win.raise_()

    logger.info("AllTracksPlot: opened with %d tracks from window '%s'",
                len(tracks_dict),
                getattr(window, 'name', '?'))
    return plot_win
