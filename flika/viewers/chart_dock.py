# -*- coding: utf-8 -*-
"""General-purpose scatter / histogram plotting dock for SPT feature data.

Provides interactive column-based plotting from pandas DataFrames, supporting
scatter, histogram, cumulative distribution, and line plot types.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui

from ..logger import logger

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# ChartDock
# ---------------------------------------------------------------------------

class ChartDock(QtWidgets.QDockWidget):
    """General-purpose scatter/histogram plotting from track feature data.

    X/Y axis column selectors, multiple plot types, data source toggle.
    """

    def __init__(self, parent=None):
        super().__init__("Chart", parent)
        self.setObjectName("ChartDock")
        self.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)

        # Data storage
        self._per_track_df = None   # pandas DataFrame or None
        self._per_point_df = None   # pandas DataFrame or None

        # Build UI
        container = QtWidgets.QWidget()
        self._build_ui(container)
        self.setWidget(container)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, container):
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Data source ---
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(QtWidgets.QLabel("Data source:"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(['Per-track', 'Per-point'])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        src_row.addWidget(self.source_combo)
        layout.addLayout(src_row)

        # --- X axis column ---
        x_row = QtWidgets.QHBoxLayout()
        x_row.addWidget(QtWidgets.QLabel("X column:"))
        self.x_combo = QtWidgets.QComboBox()
        self.x_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Fixed)
        x_row.addWidget(self.x_combo)
        layout.addLayout(x_row)

        # --- Y axis column ---
        y_row = QtWidgets.QHBoxLayout()
        y_row.addWidget(QtWidgets.QLabel("Y column:"))
        self.y_combo = QtWidgets.QComboBox()
        self.y_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Fixed)
        y_row.addWidget(self.y_combo)
        layout.addLayout(y_row)

        # --- Plot type ---
        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel("Plot type:"))
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(['Scatter', 'Histogram', 'Cumulative', 'Line'])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_row.addWidget(self.type_combo)
        layout.addLayout(type_row)

        # --- Bin count (for histograms) ---
        bin_row = QtWidgets.QHBoxLayout()
        self.bin_label = QtWidgets.QLabel("Bins:")
        bin_row.addWidget(self.bin_label)
        self.bin_spin = QtWidgets.QSpinBox()
        self.bin_spin.setRange(5, 5000)
        self.bin_spin.setValue(50)
        self.bin_spin.setToolTip("Number of bins for histogram / cumulative")
        bin_row.addWidget(self.bin_spin)
        layout.addLayout(bin_row)

        # --- Plot button ---
        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.plot_btn.setToolTip("Draw the selected plot")
        self.plot_btn.clicked.connect(self._update_plot)
        layout.addWidget(self.plot_btn)

        # --- Separator ---
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)

        # --- Plot widget ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMinimumHeight(250)
        layout.addWidget(self.plot_widget, stretch=1)

        # --- Export CSV button ---
        self.export_btn = QtWidgets.QPushButton("Export CSV")
        self.export_btn.setToolTip(
            "Export the current data source DataFrame to a CSV file")
        self.export_btn.clicked.connect(self._on_export_csv)
        layout.addWidget(self.export_btn)

        # Initial visibility of bin controls
        self._on_type_changed(self.type_combo.currentText())

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_data(self, per_track_df, per_point_df=None):
        """Set data from DataFrames.

        Parameters
        ----------
        per_track_df : pandas.DataFrame
            Feature data with one row per track.  Column names populate the
            axis selectors.
        per_point_df : pandas.DataFrame, optional
            Feature data with one row per localization / detection.
        """
        if not _HAS_PANDAS:
            # Convert dicts / arrays to a lightweight wrapper if pandas
            # is unavailable.  In practice flika always has pandas, but
            # guard anyway.
            logger.warning("ChartDock: pandas not available; data ignored")
            return

        if per_track_df is not None:
            self._per_track_df = per_track_df.copy()
        else:
            self._per_track_df = None

        if per_point_df is not None:
            self._per_point_df = per_point_df.copy()
        else:
            self._per_point_df = None

        # Refresh column combos for the current data source
        self._refresh_columns()
        logger.info("ChartDock: loaded data (per-track=%s, per-point=%s)",
                     self._per_track_df.shape if self._per_track_df is not None
                     else None,
                     self._per_point_df.shape if self._per_point_df is not None
                     else None)

    # ------------------------------------------------------------------
    # Column combo helpers
    # ------------------------------------------------------------------

    def _active_df(self):
        """Return the DataFrame for the currently selected data source."""
        if self.source_combo.currentText() == 'Per-point':
            return self._per_point_df
        return self._per_track_df

    def _refresh_columns(self):
        """Populate X / Y combos from the active DataFrame's columns."""
        df = self._active_df()

        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)

        prev_x = self.x_combo.currentText()
        prev_y = self.y_combo.currentText()

        self.x_combo.clear()
        self.y_combo.clear()

        if df is not None and hasattr(df, 'columns'):
            cols = [str(c) for c in df.columns]
            self.x_combo.addItems(cols)
            self.y_combo.addItems(cols)

            # Restore previous selections if they are still valid
            idx_x = self.x_combo.findText(prev_x)
            if idx_x >= 0:
                self.x_combo.setCurrentIndex(idx_x)
            idx_y = self.y_combo.findText(prev_y)
            if idx_y >= 0:
                self.y_combo.setCurrentIndex(idx_y)

        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _update_plot(self):
        """Clear the plot widget and redraw based on the current selections."""
        self.plot_widget.clear()

        df = self._active_df()
        if df is None or len(df) == 0:
            logger.warning("ChartDock: no data to plot")
            return

        plot_type = self.type_combo.currentText()
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()

        if not x_col:
            logger.warning("ChartDock: no X column selected")
            return

        try:
            x_data = np.asarray(df[x_col], dtype=float)
        except (KeyError, ValueError) as exc:
            logger.error("ChartDock: cannot read X column '%s': %s",
                         x_col, exc)
            return

        # Remove NaN values from x_data
        valid_x = np.isfinite(x_data)

        if plot_type == 'Scatter':
            self._plot_scatter(df, x_col, y_col, x_data, valid_x)
        elif plot_type == 'Histogram':
            self._plot_histogram(x_col, x_data, valid_x)
        elif plot_type == 'Cumulative':
            self._plot_cumulative(x_col, x_data, valid_x)
        elif plot_type == 'Line':
            self._plot_line(df, x_col, y_col, x_data, valid_x)
        else:
            logger.warning("ChartDock: unknown plot type '%s'", plot_type)

    def _plot_scatter(self, df, x_col, y_col, x_data, valid_x):
        """Scatter plot of X vs Y columns."""
        if not y_col:
            logger.warning("ChartDock: no Y column selected for scatter")
            return

        try:
            y_data = np.asarray(df[y_col], dtype=float)
        except (KeyError, ValueError) as exc:
            logger.error("ChartDock: cannot read Y column '%s': %s",
                         y_col, exc)
            return

        valid = valid_x & np.isfinite(y_data)
        xv = x_data[valid]
        yv = y_data[valid]

        if len(xv) == 0:
            logger.warning("ChartDock: no valid data for scatter")
            return

        scatter = pg.ScatterPlotItem(
            x=xv, y=yv,
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 180, 255, 160),
            hoverable=True,
            hoverPen=pg.mkPen('w', width=1.5),
            hoverBrush=pg.mkBrush(255, 255, 100, 200),
            hoverSize=8,
        )
        # Hover tooltip showing coordinates
        scatter.setToolTip("")
        scatter.sigHovered.connect(self._on_scatter_hover)
        self._scatter_x_col = x_col
        self._scatter_y_col = y_col

        self.plot_widget.addItem(scatter)
        self.plot_widget.setLabel('bottom', x_col)
        self.plot_widget.setLabel('left', y_col)

    def _on_scatter_hover(self, item, points, ev):
        """Update tooltip when hovering over scatter points."""
        if points:
            pt = points[0]
            pos = pt.pos()
            x_col = getattr(self, '_scatter_x_col', 'x')
            y_col = getattr(self, '_scatter_y_col', 'y')
            tip = f"{x_col}: {pos.x():.4g}\n{y_col}: {pos.y():.4g}"
            item.setToolTip(tip)

    def _plot_histogram(self, x_col, x_data, valid_x):
        """Histogram of the selected X column."""
        xv = x_data[valid_x]
        if len(xv) == 0:
            logger.warning("ChartDock: no valid data for histogram")
            return

        n_bins = self.bin_spin.value()
        counts, edges = np.histogram(xv, bins=n_bins)

        # Use BarGraphItem for clean histogram bars
        width = np.diff(edges)
        centers = edges[:-1] + width / 2
        bar = pg.BarGraphItem(
            x=centers, height=counts, width=width * 0.9,
            brush=pg.mkBrush(100, 180, 255, 180),
            pen=pg.mkPen('w', width=0.5),
        )
        self.plot_widget.addItem(bar)
        self.plot_widget.setLabel('bottom', x_col)
        self.plot_widget.setLabel('left', 'Count')

    def _plot_cumulative(self, x_col, x_data, valid_x):
        """Empirical cumulative distribution function of the selected column."""
        xv = x_data[valid_x]
        if len(xv) == 0:
            logger.warning("ChartDock: no valid data for cumulative plot")
            return

        sorted_vals = np.sort(xv)
        n = len(sorted_vals)
        ecdf = np.arange(1, n + 1) / n

        pen = pg.mkPen(color=(100, 180, 255), width=2)
        self.plot_widget.plot(sorted_vals, ecdf, pen=pen)
        self.plot_widget.setLabel('bottom', x_col)
        self.plot_widget.setLabel('left', 'Cumulative Probability')
        self.plot_widget.setYRange(0, 1.05)

    def _plot_line(self, df, x_col, y_col, x_data, valid_x):
        """Line plot sorted by X column values."""
        if not y_col:
            logger.warning("ChartDock: no Y column selected for line plot")
            return

        try:
            y_data = np.asarray(df[y_col], dtype=float)
        except (KeyError, ValueError) as exc:
            logger.error("ChartDock: cannot read Y column '%s': %s",
                         y_col, exc)
            return

        valid = valid_x & np.isfinite(y_data)
        xv = x_data[valid]
        yv = y_data[valid]

        if len(xv) == 0:
            logger.warning("ChartDock: no valid data for line plot")
            return

        # Sort by X for a clean line
        order = np.argsort(xv)
        xv = xv[order]
        yv = yv[order]

        pen = pg.mkPen(color=(100, 180, 255), width=2)
        self.plot_widget.plot(xv, yv, pen=pen)
        self.plot_widget.setLabel('bottom', x_col)
        self.plot_widget.setLabel('left', y_col)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_source_changed(self, source_text):
        """Refresh column combos when data source changes."""
        self._refresh_columns()

    def _on_type_changed(self, plot_type):
        """Show / hide the bin count control based on plot type."""
        show_bins = plot_type in ('Histogram', 'Cumulative')
        self.bin_label.setVisible(show_bins)
        self.bin_spin.setVisible(show_bins)

        # Y column is irrelevant for Histogram / Cumulative
        y_needed = plot_type in ('Scatter', 'Line')
        self.y_combo.setEnabled(y_needed)

    def _on_export_csv(self):
        """Export the active DataFrame to a CSV file."""
        df = self._active_df()
        if df is None or len(df) == 0:
            QtWidgets.QMessageBox.information(
                self, "Export", "No data to export.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", "chart_data.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            df.to_csv(path, index=False)
            logger.info("ChartDock: exported data to %s", path)
        except Exception as exc:
            logger.error("ChartDock: export failed: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Export Error",
                f"Failed to export:\n{exc}")


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_chart_dock(window=None):
    """Create and show a :class:`ChartDock`, optionally populated from
    a window's SPT feature data.

    Parameters
    ----------
    window : :class:`~flika.window.Window`, optional
        The image window whose SPT feature DataFrames should be loaded.
        Defaults to ``g.win``.

    Returns
    -------
    ChartDock
        The dockable chart widget instance.
    """
    from .. import global_vars as g

    if window is None:
        window = g.win

    dock = ChartDock(parent=g.m)

    # Try to populate from window SPT metadata
    if window is not None and _HAS_PANDAS:
        spt_meta = getattr(window, 'metadata', {}).get('spt', {})
        per_track_df = spt_meta.get('per_track_df', None)
        per_point_df = spt_meta.get('per_point_df', None)

        # Build a per-track DataFrame from features_by_track dict if needed
        if per_track_df is None:
            features_by_track = spt_meta.get('features_by_track', None)
            if features_by_track and isinstance(features_by_track, dict):
                try:
                    per_track_df = pd.DataFrame.from_dict(
                        features_by_track, orient='index')
                    per_track_df.index.name = 'track_id'
                    per_track_df.reset_index(inplace=True)
                except Exception as exc:
                    logger.warning("ChartDock: could not build per-track "
                                   "DataFrame from features: %s", exc)

        if per_track_df is not None:
            dock.set_data(per_track_df, per_point_df)

    if g.m is not None:
        g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    dock.show()

    return dock
