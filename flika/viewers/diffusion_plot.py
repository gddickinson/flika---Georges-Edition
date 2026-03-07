# -*- coding: utf-8 -*-
"""Diffusion analysis window for single-particle tracking data.

Provides MSD analysis, step length distributions, and CDF fitting with
1-, 2-, or 3-component exponential models to extract diffusion coefficients.

Ported from locsAndTracksPlotter's diffusionPlot.py for integration with
the flika SPT workflow.
"""
import json

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets
from scipy.optimize import curve_fit

from ..logger import logger


# ---------------------------------------------------------------------------
# CDF model functions
# ---------------------------------------------------------------------------

def _cdf_1comp(r2, D, dt):
    """1-component cumulative distribution of squared displacements.

    CDF(r^2) = 1 - exp(-r^2 / (4 * D * dt))
    """
    return 1.0 - np.exp(-r2 / (4.0 * D * dt))


def _cdf_2comp(r2, D1, D2, f1, dt):
    """2-component CDF of squared displacements.

    CDF(r^2) = 1 - f1 * exp(-r^2/(4*D1*dt)) - (1-f1) * exp(-r^2/(4*D2*dt))
    """
    return 1.0 - f1 * np.exp(-r2 / (4.0 * D1 * dt)) \
               - (1.0 - f1) * np.exp(-r2 / (4.0 * D2 * dt))


def _cdf_3comp(r2, D1, D2, D3, f1, f2, dt):
    """3-component CDF of squared displacements.

    CDF(r^2) = 1 - f1*exp(-r^2/(4*D1*dt)) - f2*exp(-r^2/(4*D2*dt))
                 - (1-f1-f2)*exp(-r^2/(4*D3*dt))
    """
    f3 = 1.0 - f1 - f2
    return 1.0 - (f1 * np.exp(-r2 / (4.0 * D1 * dt))
                  + f2 * np.exp(-r2 / (4.0 * D2 * dt))
                  + f3 * np.exp(-r2 / (4.0 * D3 * dt)))


# ---------------------------------------------------------------------------
# MSD computation helpers
# ---------------------------------------------------------------------------

def _msd_single_track(positions, max_lag):
    """Compute MSD for a single track.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        XY positions in physical units.
    max_lag : int
        Maximum lag in frames.

    Returns
    -------
    lags : ndarray
        Lag values (1 .. min(max_lag, N-1)).
    msd : ndarray
        Mean squared displacement for each lag.
    counts : ndarray
        Number of displacement pairs contributing to each lag.
    """
    n = len(positions)
    if n < 2:
        return np.array([]), np.array([]), np.array([])

    actual_max = min(max_lag, n - 1)
    lags = np.arange(1, actual_max + 1)
    msd = np.empty(len(lags))
    counts = np.empty(len(lags), dtype=int)

    for i, lag in enumerate(lags):
        displacements = positions[lag:] - positions[:-lag]
        sq_disp = np.sum(displacements ** 2, axis=1)
        msd[i] = np.mean(sq_disp)
        counts[i] = len(sq_disp)

    return lags, msd, counts


# ---------------------------------------------------------------------------
# DiffusionAnalysisWindow
# ---------------------------------------------------------------------------

class DiffusionAnalysisWindow(QtWidgets.QWidget):
    """MSD, step length, and CDF analysis for diffusion characterization.

    Plot 1: MSD vs lag time (per-track or aggregate)
    Plot 2: Step length distribution histogram
    Plot 3: CDF of squared step lengths with exponential fitting
      - 1, 2, or 3-component fits
      - D coefficient extraction

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diffusion Analysis")
        self.resize(1200, 500)

        # Data storage
        self._tracks = {}           # {track_id: (N, 3) array [frame, x, y]}
        self._pixel_size = 108.0    # nm
        self._frame_interval = 0.05 # seconds
        self._positions = {}        # {track_id: (N, 2) array [x_um, y_um]}
        self._step_lengths = np.array([])
        self._sq_step_lengths = np.array([])

        # Fit results
        self._fit_params = {}
        self._fit_r2_sorted = None
        self._fit_cdf_empirical = None
        self._fit_cdf_model = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Construct the full layout: 3 plots + control panel."""
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # --- Plot area (3 plots side by side) ---
        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Plot 1: MSD vs lag time
        self.msd_plot = pg.PlotWidget(title="MSD vs Lag Time")
        self.msd_plot.setLabel('bottom', 'Lag Time', units='s')
        self.msd_plot.setLabel('left', 'MSD', units='um^2')
        self.msd_plot.addLegend(offset=(10, 10))
        self.msd_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_splitter.addWidget(self.msd_plot)

        # Plot 2: Step length histogram
        self.step_plot = pg.PlotWidget(title="Step Length Distribution")
        self.step_plot.setLabel('bottom', 'Step Length', units='um')
        self.step_plot.setLabel('left', 'Count')
        self.step_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_splitter.addWidget(self.step_plot)

        # Plot 3: CDF of squared step lengths
        self.cdf_plot = pg.PlotWidget(title="CDF of Squared Displacements")
        self.cdf_plot.setLabel('bottom', 'r^2', units='um^2')
        self.cdf_plot.setLabel('left', 'CDF')
        self.cdf_plot.addLegend(offset=(10, 10))
        self.cdf_plot.showGrid(x=True, y=True, alpha=0.3)
        plot_splitter.addWidget(self.cdf_plot)

        main_layout.addWidget(plot_splitter, stretch=1)

        # --- Control panel ---
        ctrl_widget = QtWidgets.QWidget()
        ctrl_widget.setFixedWidth(260)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(6)

        ctrl_layout.addWidget(QtWidgets.QLabel(
            "<b>Acquisition Parameters</b>"))

        # Pixel size
        px_row = QtWidgets.QHBoxLayout()
        px_row.addWidget(QtWidgets.QLabel("Pixel size (nm):"))
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin.setRange(1.0, 10000.0)
        self.pixel_size_spin.setDecimals(1)
        self.pixel_size_spin.setValue(self._pixel_size)
        self.pixel_size_spin.setSingleStep(1.0)
        self.pixel_size_spin.setToolTip("Camera pixel size in nanometers")
        px_row.addWidget(self.pixel_size_spin)
        ctrl_layout.addLayout(px_row)

        # Frame interval
        dt_row = QtWidgets.QHBoxLayout()
        dt_row.addWidget(QtWidgets.QLabel("Frame interval (s):"))
        self.dt_spin = QtWidgets.QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 100.0)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setValue(self._frame_interval)
        self.dt_spin.setSingleStep(0.001)
        self.dt_spin.setToolTip("Time between successive frames in seconds")
        dt_row.addWidget(self.dt_spin)
        ctrl_layout.addLayout(dt_row)

        # Max lag
        lag_row = QtWidgets.QHBoxLayout()
        lag_row.addWidget(QtWidgets.QLabel("Max lag (frames):"))
        self.max_lag_spin = QtWidgets.QSpinBox()
        self.max_lag_spin.setRange(1, 1000)
        self.max_lag_spin.setValue(10)
        self.max_lag_spin.setToolTip(
            "Maximum lag time in frames for MSD calculation")
        lag_row.addWidget(self.max_lag_spin)
        ctrl_layout.addLayout(lag_row)

        ctrl_layout.addWidget(_h_separator())

        ctrl_layout.addWidget(QtWidgets.QLabel("<b>MSD Options</b>"))

        # MSD mode
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("MSD mode:"))
        self.msd_mode_combo = QtWidgets.QComboBox()
        self.msd_mode_combo.addItems([
            "Individual tracks",
            "Aggregate (mean)",
            "Both",
        ])
        self.msd_mode_combo.setCurrentIndex(1)
        self.msd_mode_combo.setToolTip(
            "Individual: one MSD curve per track\n"
            "Aggregate: weighted mean across all tracks\n"
            "Both: overlay individual curves with aggregate mean")
        mode_row.addWidget(self.msd_mode_combo)
        ctrl_layout.addLayout(mode_row)

        ctrl_layout.addWidget(_h_separator())

        ctrl_layout.addWidget(QtWidgets.QLabel("<b>CDF Fitting</b>"))

        # CDF components
        comp_row = QtWidgets.QHBoxLayout()
        comp_row.addWidget(QtWidgets.QLabel("Components:"))
        self.cdf_comp_combo = QtWidgets.QComboBox()
        self.cdf_comp_combo.addItems([
            "1-component",
            "2-component",
            "3-component",
        ])
        self.cdf_comp_combo.setToolTip(
            "Number of diffusion populations to fit in the CDF")
        comp_row.addWidget(self.cdf_comp_combo)
        ctrl_layout.addLayout(comp_row)

        # Fit button
        self.fit_btn = QtWidgets.QPushButton("Fit CDF")
        self.fit_btn.setToolTip(
            "Fit the cumulative distribution of squared step lengths\n"
            "with the selected number of exponential components")
        self.fit_btn.clicked.connect(self._on_fit_cdf)
        ctrl_layout.addWidget(self.fit_btn)

        # Update plots button
        self.update_btn = QtWidgets.QPushButton("Update Plots")
        self.update_btn.setToolTip("Recompute and refresh all plots")
        self.update_btn.clicked.connect(self._on_update_all)
        ctrl_layout.addWidget(self.update_btn)

        ctrl_layout.addWidget(_h_separator())

        ctrl_layout.addWidget(QtWidgets.QLabel("<b>Results</b>"))

        # Results text
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(140)
        self.results_text.setStyleSheet(
            "QTextEdit { font-family: monospace; font-size: 11px; }")
        ctrl_layout.addWidget(self.results_text)

        # Export button
        self.export_btn = QtWidgets.QPushButton("Export Results")
        self.export_btn.setToolTip("Export fit results to a JSON file")
        self.export_btn.clicked.connect(self._on_export)
        ctrl_layout.addWidget(self.export_btn)

        ctrl_layout.addStretch()

        main_layout.addWidget(ctrl_widget)

        # Connect parameter changes to auto-update
        self.pixel_size_spin.valueChanged.connect(self._on_params_changed)
        self.dt_spin.valueChanged.connect(self._on_params_changed)
        self.max_lag_spin.valueChanged.connect(self._on_params_changed)
        self.msd_mode_combo.currentIndexChanged.connect(self._on_params_changed)

    # ------------------------------------------------------------------
    # Public data interface
    # ------------------------------------------------------------------

    def set_data(self, tracks_dict, pixel_size=108.0, frame_interval=0.05):
        """Load track data and update all plots.

        Parameters
        ----------
        tracks_dict : dict
            Mapping ``{track_id: ndarray (N, 3)}`` with columns
            ``[frame, x, y]`` in pixel coordinates.
        pixel_size : float
            Pixel size in nanometers (default 108.0 nm).
        frame_interval : float
            Time between frames in seconds (default 0.05 s).
        """
        self._tracks = {}
        for tid, arr in tracks_dict.items():
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                logger.warning("Skipping track %s: unexpected shape %s",
                               tid, arr.shape)
                continue
            self._tracks[int(tid)] = arr[:, :3].copy()

        self._pixel_size = pixel_size
        self._frame_interval = frame_interval

        # Update spinboxes (block signals to avoid redundant recompute)
        self.pixel_size_spin.blockSignals(True)
        self.pixel_size_spin.setValue(pixel_size)
        self.pixel_size_spin.blockSignals(False)

        self.dt_spin.blockSignals(True)
        self.dt_spin.setValue(frame_interval)
        self.dt_spin.blockSignals(False)

        logger.info("Diffusion analysis: loaded %d tracks", len(self._tracks))

        self._recompute_all()

    # ------------------------------------------------------------------
    # Internal recomputation
    # ------------------------------------------------------------------

    def _read_params(self):
        """Read current parameter values from the control spinboxes."""
        self._pixel_size = self.pixel_size_spin.value()
        self._frame_interval = self.dt_spin.value()

    def _convert_to_physical(self):
        """Convert pixel coordinates to microns and store in _positions."""
        scale = self._pixel_size / 1000.0  # nm -> um
        self._positions = {}
        for tid, arr in self._tracks.items():
            xy_um = arr[:, 1:3] * scale
            self._positions[tid] = xy_um

    def _recompute_all(self):
        """Recompute all derived data and refresh all plots."""
        self._read_params()
        self._convert_to_physical()
        self._compute_step_lengths()
        self._update_msd_plot()
        self._update_step_histogram()
        self._update_cdf_plot()

    # ------------------------------------------------------------------
    # MSD computation
    # ------------------------------------------------------------------

    def _compute_msd_aggregate(self):
        """Compute the weighted-mean MSD across all tracks.

        Returns
        -------
        lag_times : ndarray
            Lag times in seconds.
        msd_mean : ndarray
            Weighted mean MSD in um^2.
        msd_sem : ndarray
            Standard error of the mean for each lag.
        """
        max_lag = self.max_lag_spin.value()
        dt = self._frame_interval

        # Collect per-track MSD arrays, keyed by lag
        lag_msd_map = {}   # lag_index -> list of (msd_value, count)
        for tid, pos in self._positions.items():
            if len(pos) < 2:
                continue
            lags, msd, counts = _msd_single_track(pos, max_lag)
            for i, lag in enumerate(lags):
                lag_msd_map.setdefault(int(lag), []).append(
                    (msd[i], counts[i]))

        if not lag_msd_map:
            return np.array([]), np.array([]), np.array([])

        sorted_lags = sorted(lag_msd_map.keys())
        lag_times = np.array(sorted_lags) * dt
        msd_mean = np.empty(len(sorted_lags))
        msd_sem = np.empty(len(sorted_lags))

        for i, lag in enumerate(sorted_lags):
            entries = lag_msd_map[lag]
            values = np.array([e[0] for e in entries])
            weights = np.array([e[1] for e in entries], dtype=float)
            total_weight = np.sum(weights)
            if total_weight > 0:
                wm = np.sum(values * weights) / total_weight
                # Weighted standard error
                if len(values) > 1:
                    var = np.sum(weights * (values - wm) ** 2) / total_weight
                    se = np.sqrt(var / len(values))
                else:
                    se = 0.0
                msd_mean[i] = wm
                msd_sem[i] = se
            else:
                msd_mean[i] = 0.0
                msd_sem[i] = 0.0

        return lag_times, msd_mean, msd_sem

    def _compute_msd_individual(self):
        """Compute per-track MSD curves.

        Returns
        -------
        curves : list of (lag_times, msd) tuples
            Each entry corresponds to one track.
        """
        max_lag = self.max_lag_spin.value()
        dt = self._frame_interval
        curves = []

        for tid in sorted(self._positions.keys()):
            pos = self._positions[tid]
            if len(pos) < 2:
                continue
            lags, msd, _ = _msd_single_track(pos, max_lag)
            if len(lags) > 0:
                curves.append((lags * dt, msd))

        return curves

    def _update_msd_plot(self):
        """Refresh Plot 1 (MSD vs lag time)."""
        self.msd_plot.clear()

        if not self._positions:
            return

        mode_text = self.msd_mode_combo.currentText()
        show_individual = mode_text in ("Individual tracks", "Both")
        show_aggregate = mode_text in ("Aggregate (mean)", "Both")

        # Individual track curves
        if show_individual:
            curves = self._compute_msd_individual()
            # Use semi-transparent lines for individual tracks
            n_curves = max(len(curves), 1)
            alpha = max(30, min(180, int(255 / (1 + n_curves / 20))))
            for lag_t, msd_vals in curves:
                pen = pg.mkPen(color=(100, 150, 255, alpha), width=1)
                self.msd_plot.plot(lag_t, msd_vals, pen=pen)

        # Aggregate MSD with error bars
        if show_aggregate:
            lag_times, msd_mean, msd_sem = self._compute_msd_aggregate()
            if len(lag_times) > 0:
                pen = pg.mkPen(color='r', width=3)
                self.msd_plot.plot(lag_times, msd_mean, pen=pen,
                                  name="Aggregate MSD")
                # Error bars
                err = pg.ErrorBarItem(
                    x=lag_times, y=msd_mean,
                    top=msd_sem, bottom=msd_sem,
                    pen=pg.mkPen(color=(255, 100, 100, 150), width=1),
                )
                self.msd_plot.addItem(err)

    # ------------------------------------------------------------------
    # Step length computation
    # ------------------------------------------------------------------

    def _compute_step_lengths(self):
        """Compute all frame-to-frame step lengths in physical units.

        Stores step lengths and squared step lengths as instance attributes.
        """
        all_steps = []
        for tid, pos in self._positions.items():
            if len(pos) < 2:
                continue
            displacements = pos[1:] - pos[:-1]
            step_len = np.sqrt(np.sum(displacements ** 2, axis=1))
            all_steps.append(step_len)

        if all_steps:
            self._step_lengths = np.concatenate(all_steps)
        else:
            self._step_lengths = np.array([])

        self._sq_step_lengths = self._step_lengths ** 2

    def _update_step_histogram(self):
        """Refresh Plot 2 (step length distribution histogram)."""
        self.step_plot.clear()

        if len(self._step_lengths) == 0:
            return

        nbins = 50
        counts, edges = np.histogram(self._step_lengths, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = edges[1] - edges[0] if len(edges) > 1 else 0.01

        bar = pg.BarGraphItem(
            x=centers, height=counts, width=width * 0.9,
            brush=pg.mkBrush(100, 150, 255, 180),
            pen=pg.mkPen(color=(60, 90, 180), width=1),
        )
        self.step_plot.addItem(bar)

        # Add summary statistics as a text item
        mean_sl = np.mean(self._step_lengths)
        median_sl = np.median(self._step_lengths)
        text = pg.TextItem(
            f"mean={mean_sl:.4f} um\nmedian={median_sl:.4f} um",
            anchor=(0, 0), color='w',
        )
        text.setPos(centers[len(centers) // 2], counts.max() * 0.9)
        self.step_plot.addItem(text)

    # ------------------------------------------------------------------
    # CDF fitting
    # ------------------------------------------------------------------

    def _fit_cdf(self, n_components):
        """Fit the empirical CDF of squared step lengths.

        Parameters
        ----------
        n_components : int
            Number of exponential components (1, 2, or 3).

        Returns
        -------
        params : dict
            Fitted parameters with keys like 'D1', 'D2', 'f1', etc.
            Also contains 'n_components', 'dt', 'n_steps', and
            'residual_sum_sq'.
        r2_sorted : ndarray
            Sorted squared step lengths.
        cdf_empirical : ndarray
            Empirical CDF values.
        cdf_fit : ndarray
            Fitted CDF values.
        """
        if len(self._sq_step_lengths) < 5:
            logger.warning("Not enough step lengths for CDF fitting "
                           "(need >= 5, have %d)", len(self._sq_step_lengths))
            return None, None, None, None

        dt = self._frame_interval

        # Sort squared step lengths and compute empirical CDF
        r2_sorted = np.sort(self._sq_step_lengths)
        n = len(r2_sorted)
        cdf_empirical = np.arange(1, n + 1) / float(n)

        # Data-driven initial guesses
        # Estimate D from the mean squared displacement: <r^2> = 4*D*dt (2D)
        mean_r2 = np.mean(r2_sorted)
        d_est = mean_r2 / (4.0 * dt) if dt > 0 else 1.0

        # Use quantile-based estimates for multi-component fits
        q25_r2 = np.percentile(r2_sorted, 25)
        q75_r2 = np.percentile(r2_sorted, 75)
        d_slow = q25_r2 / (4.0 * dt) if dt > 0 else 0.1
        d_fast = q75_r2 / (4.0 * dt) if dt > 0 else 10.0
        # Ensure minimum positive values
        d_est = max(d_est, 1e-8)
        d_slow = max(d_slow, 1e-8)
        d_fast = max(d_fast, d_slow * 2)

        params = {'n_components': n_components, 'dt': dt, 'n_steps': n}

        try:
            if n_components == 1:
                popt, pcov = curve_fit(
                    lambda r2, D: _cdf_1comp(r2, D, dt),
                    r2_sorted, cdf_empirical,
                    p0=[d_est],
                    bounds=([1e-10], [np.inf]),
                    maxfev=10000,
                )
                params['D1'] = float(popt[0])
                cdf_fit = _cdf_1comp(r2_sorted, popt[0], dt)

                # Compute standard errors from covariance
                perr = np.sqrt(np.diag(pcov))
                params['D1_err'] = float(perr[0])

            elif n_components == 2:
                # 2-component: fit D1, D2, f1
                # Initial guess: slow and fast populations
                p0 = [d_slow, d_fast, 0.5]
                bounds_lo = [1e-10, 1e-10, 0.01]
                bounds_hi = [np.inf, np.inf, 0.99]

                popt, pcov = curve_fit(
                    lambda r2, D1, D2, f1: _cdf_2comp(r2, D1, D2, f1, dt),
                    r2_sorted, cdf_empirical,
                    p0=p0,
                    bounds=(bounds_lo, bounds_hi),
                    maxfev=20000,
                )

                # Ensure D1 <= D2 (sort by diffusion coefficient)
                D1, D2, f1 = popt[0], popt[1], popt[2]
                if D1 > D2:
                    D1, D2 = D2, D1
                    f1 = 1.0 - f1

                params['D1'] = float(D1)
                params['D2'] = float(D2)
                params['f1'] = float(f1)
                params['f2'] = float(1.0 - f1)
                cdf_fit = _cdf_2comp(r2_sorted, D1, D2, f1, dt)

                perr = np.sqrt(np.diag(pcov))
                params['D1_err'] = float(perr[0])
                params['D2_err'] = float(perr[1])
                params['f1_err'] = float(perr[2])

            elif n_components == 3:
                # 3-component: fit D1, D2, D3, f1, f2
                q33_r2 = np.percentile(r2_sorted, 33)
                q66_r2 = np.percentile(r2_sorted, 66)
                d_mid = q33_r2 / (4.0 * dt) if dt > 0 else 1.0
                d_mid = max(d_mid, d_slow * 1.5)
                d_fast_3 = q66_r2 / (4.0 * dt) if dt > 0 else 10.0
                d_fast_3 = max(d_fast_3, d_mid * 1.5)

                p0 = [d_slow, d_mid, d_fast_3, 0.33, 0.34]
                bounds_lo = [1e-10, 1e-10, 1e-10, 0.01, 0.01]
                bounds_hi = [np.inf, np.inf, np.inf, 0.98, 0.98]

                popt, pcov = curve_fit(
                    lambda r2, D1, D2, D3, f1, f2: _cdf_3comp(
                        r2, D1, D2, D3, f1, f2, dt),
                    r2_sorted, cdf_empirical,
                    p0=p0,
                    bounds=(bounds_lo, bounds_hi),
                    maxfev=50000,
                )

                # Sort populations by D
                D_vals = np.array([popt[0], popt[1], popt[2]])
                f_vals = np.array([popt[3], popt[4],
                                   1.0 - popt[3] - popt[4]])
                sort_idx = np.argsort(D_vals)
                D_vals = D_vals[sort_idx]
                f_vals = f_vals[sort_idx]

                params['D1'] = float(D_vals[0])
                params['D2'] = float(D_vals[1])
                params['D3'] = float(D_vals[2])
                params['f1'] = float(f_vals[0])
                params['f2'] = float(f_vals[1])
                params['f3'] = float(f_vals[2])
                cdf_fit = _cdf_3comp(
                    r2_sorted, D_vals[0], D_vals[1], D_vals[2],
                    f_vals[0], f_vals[1], dt)

                perr = np.sqrt(np.diag(pcov))
                params['D1_err'] = float(perr[0])
                params['D2_err'] = float(perr[1])
                params['D3_err'] = float(perr[2])
                params['f1_err'] = float(perr[3])
                params['f2_err'] = float(perr[4])
            else:
                logger.error("Invalid n_components=%d", n_components)
                return None, None, None, None

            # Goodness of fit: residual sum of squares
            residuals = cdf_empirical - cdf_fit
            params['residual_sum_sq'] = float(np.sum(residuals ** 2))

            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((cdf_empirical - np.mean(cdf_empirical)) ** 2)
            params['r_squared'] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        except (RuntimeError, ValueError) as exc:
            logger.error("CDF fitting failed (%d-component): %s",
                         n_components, exc)
            self.results_text.setPlainText(
                f"Fitting failed ({n_components}-component):\n{exc}\n\n"
                "Try a different number of components or check that\n"
                "the data contains enough steps.")
            return None, None, None, None

        return params, r2_sorted, cdf_empirical, cdf_fit

    def _update_cdf_plot(self):
        """Refresh Plot 3 (CDF of squared displacements).

        Shows the empirical CDF. If a fit has been performed, also shows
        the fitted curve and individual component curves.
        """
        self.cdf_plot.clear()

        if len(self._sq_step_lengths) == 0:
            return

        # Empirical CDF (always shown)
        r2_sorted = np.sort(self._sq_step_lengths)
        n = len(r2_sorted)
        cdf_emp = np.arange(1, n + 1) / float(n)

        self.cdf_plot.plot(r2_sorted, cdf_emp,
                          pen=pg.mkPen(color=(100, 150, 255), width=2),
                          name="Empirical CDF")

        # Fitted curve (if available)
        if (self._fit_r2_sorted is not None
                and self._fit_cdf_model is not None):
            self.cdf_plot.plot(
                self._fit_r2_sorted, self._fit_cdf_model,
                pen=pg.mkPen(color='r', width=2, style=QtCore.Qt.DashLine),
                name="Fit")

            # Draw individual component contributions
            dt = self._fit_params.get('dt', self._frame_interval)
            nc = self._fit_params.get('n_components', 0)
            r2 = self._fit_r2_sorted

            component_colors = [
                (255, 200, 50, 150),   # yellow
                (50, 255, 100, 150),   # green
                (255, 100, 255, 150),  # magenta
            ]

            if nc >= 2:
                for k in range(1, nc + 1):
                    D_k = self._fit_params.get(f'D{k}')
                    f_k = self._fit_params.get(f'f{k}')
                    if D_k is not None and f_k is not None:
                        # Component contribution to survival function
                        comp_cdf = f_k * _cdf_1comp(r2, D_k, dt)
                        color = component_colors[(k - 1) % len(
                            component_colors)]
                        pen = pg.mkPen(color=color, width=1,
                                       style=QtCore.Qt.DotLine)
                        self.cdf_plot.plot(
                            r2, comp_cdf, pen=pen,
                            name=f"D{k}={D_k:.4f} um^2/s ({f_k:.0%})")

    # ------------------------------------------------------------------
    # Formatting results
    # ------------------------------------------------------------------

    def _format_results(self, params):
        """Format fit results as human-readable text.

        Parameters
        ----------
        params : dict
            Fit parameters from ``_fit_cdf``.

        Returns
        -------
        str
            Formatted multi-line text.
        """
        if params is None:
            return "No fit results."

        nc = params.get('n_components', 0)
        dt = params.get('dt', 0)
        n_steps = params.get('n_steps', 0)
        lines = [
            f"CDF Fit: {nc}-component",
            f"dt = {dt:.4f} s, N = {n_steps} steps",
            "",
        ]

        for k in range(1, nc + 1):
            D = params.get(f'D{k}', 0)
            D_err = params.get(f'D{k}_err', 0)
            line = f"  D{k} = {D:.6f} +/- {D_err:.6f} um^2/s"
            if nc > 1:
                f_k = params.get(f'f{k}', 0)
                f_err = params.get(f'f{k}_err', 0)
                line += f"\n  f{k} = {f_k:.4f}"
                if f_err > 0:
                    line += f" +/- {f_err:.4f}"
                line += f" ({f_k:.1%})"
            lines.append(line)

        lines.append("")
        r2 = params.get('r_squared', 0)
        rss = params.get('residual_sum_sq', 0)
        lines.append(f"R^2 = {r2:.6f}")
        lines.append(f"RSS = {rss:.6f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_params_changed(self, *_args):
        """Called when any acquisition parameter or MSD mode changes."""
        if self._tracks:
            self._recompute_all()

    def _on_update_all(self):
        """Explicit update triggered by the Update Plots button."""
        if not self._tracks:
            self.results_text.setPlainText(
                "No track data loaded.\n"
                "Use set_data() or open from an SPT window.")
            return
        self._recompute_all()

    def _on_fit_cdf(self):
        """Triggered by the Fit CDF button."""
        if len(self._sq_step_lengths) == 0:
            self.results_text.setPlainText(
                "No step length data available.\nLoad tracks first.")
            return

        comp_text = self.cdf_comp_combo.currentText()
        if comp_text.startswith("1"):
            n_comp = 1
        elif comp_text.startswith("2"):
            n_comp = 2
        else:
            n_comp = 3

        params, r2_sorted, cdf_emp, cdf_fit = self._fit_cdf(n_comp)

        if params is not None:
            self._fit_params = params
            self._fit_r2_sorted = r2_sorted
            self._fit_cdf_empirical = cdf_emp
            self._fit_cdf_model = cdf_fit
            self.results_text.setPlainText(self._format_results(params))
            self._update_cdf_plot()
            logger.info("CDF fit complete (%d-component): %s",
                        n_comp, params)

    def _on_export(self):
        """Open a save dialog and export fit results to JSON."""
        if not self._fit_params:
            self.results_text.setPlainText(
                "No fit results to export.\nRun 'Fit CDF' first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Fit Results", "diffusion_results.json",
            "JSON Files (*.json);;All Files (*)")
        if path:
            self.export_fit_results(path)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_fit_results(self, path):
        """Export fit results and summary statistics to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        output = {
            'fit_parameters': self._fit_params,
            'acquisition': {
                'pixel_size_nm': self._pixel_size,
                'frame_interval_s': self._frame_interval,
            },
            'data_summary': {
                'n_tracks': len(self._tracks),
                'n_steps': len(self._step_lengths),
                'mean_step_length_um': (
                    float(np.mean(self._step_lengths))
                    if len(self._step_lengths) > 0 else None),
                'median_step_length_um': (
                    float(np.median(self._step_lengths))
                    if len(self._step_lengths) > 0 else None),
                'mean_r2_um2': (
                    float(np.mean(self._sq_step_lengths))
                    if len(self._sq_step_lengths) > 0 else None),
                'track_lengths': {
                    str(tid): len(arr) for tid, arr in self._tracks.items()
                },
            },
        }

        # Include MSD aggregate data if available
        lag_times, msd_mean, msd_sem = self._compute_msd_aggregate()
        if len(lag_times) > 0:
            output['msd_aggregate'] = {
                'lag_times_s': lag_times.tolist(),
                'msd_um2': msd_mean.tolist(),
                'msd_sem_um2': msd_sem.tolist(),
            }

        try:
            with open(path, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info("Exported diffusion results to %s", path)
            self.results_text.append(f"\nExported to: {path}")
        except Exception as exc:
            logger.error("Failed to export results: %s", exc)
            self.results_text.append(f"\nExport failed: {exc}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Clean up resources on window close."""
        self._tracks.clear()
        self._positions.clear()
        self._fit_params.clear()
        event.accept()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _h_separator():
    """Return a horizontal line separator widget."""
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_diffusion_analysis(window=None):
    """Open a diffusion analysis window for SPT tracks.

    Expects track data in ``window.metadata['spt']['tracks_dict']``, where
    each entry is ``{track_id: (N, 3) array [frame, x, y]}`` in pixel coords.

    Parameters
    ----------
    window : :class:`~flika.window.Window`, optional
        The image window containing SPT data.  Defaults to ``g.win``.

    Returns
    -------
    DiffusionAnalysisWindow
        The newly created analysis window.
    """
    from .. import global_vars as g

    if window is None:
        window = g.win
    if window is None:
        logger.error("No active window for diffusion analysis")
        raise RuntimeError("No active window.  Open an image first.")

    spt = window.metadata.get('spt', {})
    tracks_dict = spt.get('tracks_dict', None)
    if tracks_dict is None or len(tracks_dict) == 0:
        logger.error("No SPT tracks found in window metadata")
        raise ValueError(
            "No track data found in window.metadata['spt']['tracks_dict'].\n"
            "Run particle tracking first.")

    # Read acquisition parameters if available in metadata
    pixel_size = spt.get('pixel_size_nm', 108.0)
    frame_interval = spt.get('frame_interval_s', 0.05)

    win = DiffusionAnalysisWindow(parent=None)
    win.set_data(tracks_dict,
                 pixel_size=pixel_size,
                 frame_interval=frame_interval)
    win.show()
    win.raise_()

    logger.info("Opened diffusion analysis: %d tracks, px=%.1f nm, dt=%.4f s",
                len(tracks_dict), pixel_size, frame_interval)

    return win
