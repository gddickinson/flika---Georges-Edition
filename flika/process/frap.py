# -*- coding: utf-8 -*-
"""FRAP (Fluorescence Recovery After Photobleaching) analysis.

Provides single- and double-exponential recovery fitting, mobile/immobile
fraction estimation, half-time computation, and Soumpasis diffusion
coefficient estimation.
"""
from flika.logger import logger
logger.debug("Started 'reading process/frap.py'")
import numpy as np
from scipy.optimize import curve_fit
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
import flika.global_vars as g
from flika.utils.BaseProcess import BaseProcess
from flika.utils.custom_widgets import WindowSelector, SliderLabel, CheckBox, ComboBox, MissingWindowError

__all__ = ['frap_analysis']


# ---------------------------------------------------------------------------
# Inlined fitting utilities (from flika.utils.fitting, not in this repo)
# ---------------------------------------------------------------------------

def _r_squared(y_true, y_predicted):
    """Compute R-squared (coefficient of determination)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_predicted = np.asarray(y_predicted, dtype=np.float64)
    ss_res = np.sum((y_true - y_predicted) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def exp_recovery(t, amplitude, tau, offset):
    """Single-exponential recovery: amplitude * (1 - exp(-t/tau)) + offset."""
    return amplitude * (1.0 - np.exp(-np.asarray(t, dtype=np.float64) / tau)) + offset


def double_exp_recovery(t, A1, tau1, A2, tau2, offset):
    """Double-exponential recovery model."""
    t = np.asarray(t, dtype=np.float64)
    return A1 * (1.0 - np.exp(-t / tau1)) + A2 * (1.0 - np.exp(-t / tau2)) + offset


def fit_exponential_recovery(time, data, p0=None):
    """Fit single-exponential recovery. Returns (params, r_squared)."""
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if p0 is None:
        amp0 = float(np.max(data) - np.min(data))
        t_span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
        tau0 = max(t_span / 2.0, 1.0)
        off0 = float(np.min(data))
        p0 = (amp0, tau0, off0)
    popt, _ = curve_fit(
        exp_recovery, time, data, p0=p0,
        bounds=([0, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=10000)
    fitted = exp_recovery(time, *popt)
    r2 = _r_squared(data, fitted)
    return tuple(float(v) for v in popt), r2


def fit_double_exponential_recovery(time, data, p0=None):
    """Fit double-exponential recovery. Returns (params, r_squared)."""
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    if p0 is None:
        A_total = float(np.max(data) - np.min(data))
        t_span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
        if t_span <= 0:
            t_span = 1.0
        off0 = float(np.min(data))
        p0 = (A_total * 0.6, t_span / 4.0, A_total * 0.4, t_span, off0)
    popt, _ = curve_fit(
        double_exp_recovery, time, data, p0=p0,
        bounds=([0, 1e-12, 0, 1e-12, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf]),
        maxfev=20000)
    A1, tau1, A2, tau2, offset = popt
    if tau1 > tau2:
        A1, A2 = A2, A1
        tau1, tau2 = tau2, tau1
    fitted = double_exp_recovery(time, A1, tau1, A2, tau2, offset)
    r2 = _r_squared(data, fitted)
    return (float(A1), float(tau1), float(A2), float(tau2), float(offset)), r2


# ---------------------------------------------------------------------------
# Pure analysis functions (no Qt dependencies)
# ---------------------------------------------------------------------------

def normalize_frap(intensities, pre_bleach_frames, bleach_frame):
    """Double-normalize a FRAP intensity trace.

    Normalization formula::

        normalized = (I - post_min) / (pre_mean - post_min)

    so that the pre-bleach baseline is ~1.0 and the post-bleach minimum
    is 0.0.

    Parameters
    ----------
    intensities : 1-D array-like
        Raw mean intensity per frame.
    pre_bleach_frames : int
        Number of frames before *bleach_frame* to average for the baseline.
    bleach_frame : int
        Index of the bleach event frame.

    Returns
    -------
    numpy.ndarray
        Normalized recovery curve (same length as *intensities*).
    """
    intensities = np.asarray(intensities, dtype=np.float64)

    pre_start = max(0, bleach_frame - pre_bleach_frames)
    pre_region = intensities[pre_start:bleach_frame]
    if pre_region.size == 0:
        pre_mean = intensities[0]
    else:
        pre_mean = np.mean(pre_region)

    post_region = intensities[bleach_frame:]
    if post_region.size == 0:
        post_min = 0.0
    else:
        post_min = np.min(post_region)

    denom = pre_mean - post_min
    if abs(denom) < 1e-12:
        return np.zeros_like(intensities)

    return (intensities - post_min) / denom


def fit_single_exponential(time, intensities):
    """Fit a single-exponential recovery model to normalized FRAP data.

    Model::

        y = A * (1 - exp(-t / tau)) + offset

    Parameters
    ----------
    time : 1-D array
        Time values (e.g. frame indices or seconds).
    intensities : 1-D array
        Normalized intensity values.

    Returns
    -------
    dict
        Keys: tau, mobile_fraction, immobile_fraction, r_squared, fit_curve.
        Returns NaN values on fit failure.
    """
    time = np.asarray(time, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    result = {
        'tau': np.nan,
        'mobile_fraction': np.nan,
        'immobile_fraction': np.nan,
        'r_squared': np.nan,
        'fit_curve': np.full_like(intensities, np.nan),
    }

    if time.size < 3:
        return result

    try:
        (A, tau, offset), r2 = fit_exponential_recovery(time, intensities)
    except (RuntimeError, ValueError):
        return result

    fit_curve = exp_recovery(time, A, tau, offset)

    mobile_fraction = float(A + offset)
    immobile_fraction = 1.0 - mobile_fraction

    result['tau'] = float(tau)
    result['mobile_fraction'] = float(mobile_fraction)
    result['immobile_fraction'] = float(immobile_fraction)
    result['r_squared'] = float(r2)
    result['fit_curve'] = fit_curve

    return result


def fit_double_exponential(time, intensities):
    """Fit a two-component exponential recovery model to normalized FRAP data.

    Model::

        y = A1*(1 - exp(-t/tau1)) + A2*(1 - exp(-t/tau2)) + offset

    Parameters
    ----------
    time : 1-D array
        Time values.
    intensities : 1-D array
        Normalized intensity values.

    Returns
    -------
    dict
        Keys: tau1, tau2, fraction1, fraction2, mobile_fraction,
        immobile_fraction, r_squared, fit_curve.
        Returns NaN values on fit failure.
    """
    time = np.asarray(time, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    result = {
        'tau1': np.nan,
        'tau2': np.nan,
        'fraction1': np.nan,
        'fraction2': np.nan,
        'mobile_fraction': np.nan,
        'immobile_fraction': np.nan,
        'r_squared': np.nan,
        'fit_curve': np.full_like(intensities, np.nan),
    }

    if time.size < 5:
        return result

    try:
        (A1, tau1, A2, tau2, offset), r2 = fit_double_exponential_recovery(
            time, intensities)
    except (RuntimeError, ValueError):
        return result

    fit_curve = double_exp_recovery(time, A1, tau1, A2, tau2, offset)

    A_sum = A1 + A2
    if A_sum > 0:
        frac1 = float(A1 / A_sum)
        frac2 = float(A2 / A_sum)
    else:
        frac1, frac2 = 0.5, 0.5

    mobile_fraction = float(A1 + A2 + offset)
    immobile_fraction = 1.0 - mobile_fraction

    result['tau1'] = float(tau1)
    result['tau2'] = float(tau2)
    result['fraction1'] = frac1
    result['fraction2'] = frac2
    result['mobile_fraction'] = float(mobile_fraction)
    result['immobile_fraction'] = float(immobile_fraction)
    result['r_squared'] = float(r2)
    result['fit_curve'] = fit_curve

    return result


def compute_half_time(tau):
    """Compute the half-time of recovery from an exponential time constant.

    Parameters
    ----------
    tau : float
        Exponential time constant.

    Returns
    -------
    float
        t_half = tau * ln(2).
    """
    return float(tau * np.log(2))


def compute_diffusion_coefficient(half_time, bleach_radius, dimensionality=2):
    """Estimate the effective diffusion coefficient using the Soumpasis equation.

    For a 2-D circular bleach spot::

        D = 0.224 * r^2 / t_half

    Parameters
    ----------
    half_time : float
        Recovery half-time (same time units as desired for D).
    bleach_radius : float
        Radius of the bleach spot (same length units as desired for D).
    dimensionality : int
        Spatial dimensionality (only 2 is currently supported).

    Returns
    -------
    float
        Diffusion coefficient in units of [length^2 / time].

    Raises
    ------
    ValueError
        If *dimensionality* is not 2 or *half_time* is non-positive.
    """
    if dimensionality != 2:
        raise ValueError(
            "Soumpasis equation is only valid for 2-D; got dimensionality={}".format(dimensionality)
        )
    if half_time <= 0:
        raise ValueError("half_time must be positive; got {}".format(half_time))

    return 0.224 * bleach_radius ** 2 / half_time


# ---------------------------------------------------------------------------
# BaseProcess subclass
# ---------------------------------------------------------------------------

class FRAPAnalysis(BaseProcess):
    """frap_analysis(window, pre_bleach_frames=10, bleach_frame=10, bleach_radius_um=1.0, model='single_exponential', use_roi=False, keepSourceWindow=True)

    Fluorescence Recovery After Photobleaching (FRAP) analysis.

    Extracts the mean intensity from the active ROI (or full image) across
    all frames, normalises the recovery curve using double normalisation,
    fits a single- or double-exponential recovery model, computes the
    mobile/immobile fraction, recovery half-time, and (optionally) the
    diffusion coefficient via the Soumpasis equation.

    Results are printed to the console, shown in a pyqtgraph plot, and
    stored in ``window.metadata['frap']``.

    Parameters:
        window (Window): Image stack window containing FRAP data.
        pre_bleach_frames (int): Number of pre-bleach frames to average for baseline.
        bleach_frame (int): Frame index where photobleaching occurs.
        bleach_radius_um (float): Radius of the bleach spot in micrometres.
        model (str): 'single_exponential' or 'double_exponential'.
        use_roi (bool): If True, extract intensity from the active ROI.
    Returns:
        None
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window = WindowSelector()
        pre_bleach_frames = SliderLabel(decimals=0)
        pre_bleach_frames.setRange(1, 500)
        pre_bleach_frames.setValue(10)
        bleach_frame = SliderLabel(decimals=0)
        bleach_frame.setRange(0, 10000)
        bleach_frame.setValue(10)
        bleach_radius_um = SliderLabel(decimals=2)
        bleach_radius_um.setRange(0.01, 100.0)
        bleach_radius_um.setValue(1.0)
        model = ComboBox()
        model.addItem('single_exponential')
        model.addItem('double_exponential')
        use_roi = CheckBox()
        use_roi.setChecked(True)
        self.items.append({'name': 'window', 'string': 'Window', 'object': window})
        self.items.append({'name': 'pre_bleach_frames', 'string': 'Pre-bleach frames', 'object': pre_bleach_frames})
        self.items.append({'name': 'bleach_frame', 'string': 'Bleach frame', 'object': bleach_frame})
        self.items.append({'name': 'bleach_radius_um', 'string': 'Bleach radius (um)', 'object': bleach_radius_um})
        self.items.append({'name': 'model', 'string': 'Model', 'object': model})
        self.items.append({'name': 'use_roi', 'string': 'Use ROI', 'object': use_roi})
        super().gui()

    def __call__(self, window, pre_bleach_frames=10, bleach_frame=10,
                 bleach_radius_um=1.0, model='single_exponential',
                 use_roi=False, keepSourceWindow=True):
        if window is None:
            raise MissingWindowError("A window must be selected for FRAP analysis.")

        img = window.image
        if img.ndim < 3:
            g.alert("FRAP analysis requires a time-series (3-D) image stack.")
            return

        n_frames = img.shape[0]
        if bleach_frame >= n_frames:
            g.alert("Bleach frame ({}) exceeds stack length ({}).".format(bleach_frame, n_frames))
            return

        # --- Build ROI mask ------------------------------------------------
        mask = None
        if use_roi:
            if hasattr(window, 'currentROI') and window.currentROI is not None:
                roi = window.currentROI
                yy, xx = roi.getMask()
                if yy.size > 0:
                    mask = np.zeros(img.shape[1:3], dtype=bool)
                    valid = (
                        (yy >= 0) & (yy < img.shape[1]) &
                        (xx >= 0) & (xx < img.shape[2])
                    )
                    mask[yy[valid], xx[valid]] = True
                else:
                    if g.m is not None:
                        g.m.statusBar().showMessage('ROI mask is empty; analysing full image.')
            else:
                if g.m is not None:
                    g.m.statusBar().showMessage('No ROI found; analysing full image.')

        # --- Extract mean intensity per frame ------------------------------
        if mask is not None:
            intensities = np.array([
                np.mean(img[i][mask]) for i in range(n_frames)
            ])
        else:
            intensities = np.array([
                np.mean(img[i]) for i in range(n_frames)
            ])

        # Guard against constant intensity
        if np.std(intensities) < 1e-12:
            g.alert("Intensity trace is constant; cannot perform FRAP analysis.")
            return

        # --- Normalise -----------------------------------------------------
        norm = normalize_frap(intensities, pre_bleach_frames, bleach_frame)

        # Use only post-bleach data for fitting
        post_bleach_norm = norm[bleach_frame:]
        time = np.arange(len(post_bleach_norm), dtype=np.float64)

        # --- Fit recovery model -------------------------------------------
        if model == 'double_exponential':
            fit_result = fit_double_exponential(time, post_bleach_norm)
        else:
            fit_result = fit_single_exponential(time, post_bleach_norm)

        # --- Derived quantities -------------------------------------------
        if model == 'double_exponential':
            tau_eff = np.nan
            if not np.isnan(fit_result['tau1']) and not np.isnan(fit_result['tau2']):
                f1 = fit_result['fraction1']
                f2 = fit_result['fraction2']
                tau_eff = f1 * fit_result['tau1'] + f2 * fit_result['tau2']
        else:
            tau_eff = fit_result.get('tau', np.nan)

        if not np.isnan(tau_eff) and tau_eff > 0:
            t_half = compute_half_time(tau_eff)
        else:
            t_half = np.nan

        D = np.nan
        if not np.isnan(t_half) and t_half > 0 and bleach_radius_um > 0:
            try:
                D = compute_diffusion_coefficient(t_half, bleach_radius_um, dimensionality=2)
            except ValueError:
                D = np.nan

        # --- Build results dict -------------------------------------------
        results = {
            'model': model,
            'pre_bleach_frames': int(pre_bleach_frames),
            'bleach_frame': int(bleach_frame),
            'bleach_radius_um': float(bleach_radius_um),
            'raw_intensities': intensities,
            'normalized': norm,
            'mobile_fraction': fit_result['mobile_fraction'],
            'immobile_fraction': fit_result['immobile_fraction'],
            'r_squared': fit_result['r_squared'],
            't_half': t_half,
            'diffusion_coefficient': D,
        }

        if model == 'double_exponential':
            results['tau1'] = fit_result['tau1']
            results['tau2'] = fit_result['tau2']
            results['fraction1'] = fit_result['fraction1']
            results['fraction2'] = fit_result['fraction2']
            results['tau_effective'] = tau_eff
        else:
            results['tau'] = fit_result.get('tau', np.nan)

        # --- Print results -------------------------------------------------
        name = window.name
        print("=" * 55)
        print("FRAP Analysis: {}".format(name))
        print("=" * 55)
        print("  Model              = {}".format(model))
        print("  Pre-bleach frames  = {}".format(pre_bleach_frames))
        print("  Bleach frame       = {}".format(bleach_frame))
        print("  Bleach radius (um) = {:.2f}".format(bleach_radius_um))

        if model == 'double_exponential':
            print("  tau1 (frames)      = {:.4f}".format(fit_result['tau1']))
            print("  tau2 (frames)      = {:.4f}".format(fit_result['tau2']))
            print("  Fraction 1         = {:.4f}".format(fit_result['fraction1']))
            print("  Fraction 2         = {:.4f}".format(fit_result['fraction2']))
            print("  tau_eff (frames)   = {:.4f}".format(tau_eff))
        else:
            print("  tau (frames)       = {:.4f}".format(fit_result.get('tau', np.nan)))

        print("  Mobile fraction    = {:.4f}".format(fit_result['mobile_fraction']))
        print("  Immobile fraction  = {:.4f}".format(fit_result['immobile_fraction']))
        print("  R-squared          = {:.4f}".format(fit_result['r_squared']))
        print("  t_half (frames)    = {:.4f}".format(t_half))
        print("  D (um^2/frame)     = {:.6f}".format(D))
        print("=" * 55)

        # --- Store in metadata --------------------------------------------
        window.metadata['frap'] = results

        # --- Status bar ----------------------------------------------------
        if g.m is not None:
            g.m.statusBar().showMessage(
                'FRAP: mobile={:.3f}, t_half={:.2f}, R^2={:.3f}'.format(
                    fit_result['mobile_fraction'], t_half, fit_result['r_squared']))

        # --- Plot ----------------------------------------------------------
        self._plot_widget = QtWidgets.QWidget()
        self._plot_widget.setWindowTitle('FRAP Recovery: {}'.format(name))
        self._plot_widget.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self._plot_widget)

        pw = pg.PlotWidget()
        pw.setLabel('bottom', 'Frame')
        pw.setLabel('left', 'Normalized Intensity')
        pw.setTitle('FRAP Recovery: {}'.format(name))
        layout.addWidget(pw)

        # Full normalised trace
        frames_all = np.arange(n_frames)
        pw.plot(frames_all, norm, pen=None, symbol='o', symbolSize=4,
                symbolBrush=pg.mkBrush(150, 150, 255, 180), name='Data')

        # Fit curve (post-bleach only)
        fit_curve = fit_result['fit_curve']
        if fit_curve is not None and not np.all(np.isnan(fit_curve)):
            post_frames = np.arange(bleach_frame, n_frames)
            pw.plot(post_frames, fit_curve, pen=pg.mkPen('r', width=2), name='Fit')

        # Half-time annotation
        if not np.isnan(t_half):
            t_half_frame = bleach_frame + t_half
            vline = pg.InfiniteLine(
                pos=t_half_frame, angle=90,
                pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashLine),
                label='t_half={:.2f}'.format(t_half),
                labelOpts={'position': 0.85, 'color': 'g'},
            )
            pw.addItem(vline)

        # Bleach frame marker
        bleach_line = pg.InfiniteLine(
            pos=bleach_frame, angle=90,
            pen=pg.mkPen('y', width=1, style=QtCore.Qt.DotLine),
            label='bleach',
            labelOpts={'position': 0.95, 'color': 'y'},
        )
        pw.addItem(bleach_line)

        # Stats overlay
        if model == 'double_exponential':
            stats_text = (
                "tau1={:.2f}  tau2={:.2f}\n"
                "mobile={:.3f}\n"
                "t_half={:.2f}  R^2={:.3f}".format(
                    fit_result['tau1'], fit_result['tau2'],
                    fit_result['mobile_fraction'],
                    t_half, fit_result['r_squared']))
        else:
            stats_text = (
                "tau={:.2f}\n"
                "mobile={:.3f}\n"
                "t_half={:.2f}  R^2={:.3f}".format(
                    fit_result.get('tau', np.nan),
                    fit_result['mobile_fraction'],
                    t_half, fit_result['r_squared']))
        text_item = pg.TextItem(stats_text, anchor=(0, 0), color='w')
        text_item.setPos(0, np.max(norm))
        pw.addItem(text_item)

        self._plot_widget.show()

        return None

    def get_init_settings_dict(self):
        return {
            'pre_bleach_frames': 10,
            'bleach_frame': 10,
            'bleach_radius_um': 1.0,
            'model': 'single_exponential',
            'use_roi': False,
        }


frap_analysis = FRAPAnalysis()

logger.debug("Completed 'reading process/frap.py'")
