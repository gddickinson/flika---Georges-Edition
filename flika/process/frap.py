# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/frap.py'")
import numpy as np
from scipy.optimize import curve_fit
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox, SliderLabel, ComboBox

__all__ = ['frap_analysis']


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
        # Constant intensity -- return zeros to avoid division by zero
        return np.zeros_like(intensities)

    return (intensities - post_min) / denom


def _single_exp_model(t, A, tau, offset):
    """y = A * (1 - exp(-t / tau)) + offset"""
    return A * (1.0 - np.exp(-t / tau)) + offset


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
        Returns None values on fit failure.
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

    # Initial guesses: A ~ recovery range, tau ~ half the time span, offset ~ minimum
    A0 = float(np.max(intensities) - np.min(intensities))
    tau0 = float((time[-1] - time[0]) / 2.0)
    offset0 = float(np.min(intensities))

    if tau0 <= 0:
        tau0 = 1.0

    try:
        popt, _ = curve_fit(
            _single_exp_model, time, intensities,
            p0=[A0, tau0, offset0],
            bounds=([0, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        A, tau, offset = popt
    except (RuntimeError, ValueError):
        return result

    fit_curve = _single_exp_model(time, A, tau, offset)

    # R-squared
    ss_res = np.sum((intensities - fit_curve) ** 2)
    ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Mobile fraction = plateau value = A + offset
    mobile_fraction = float(A + offset)
    immobile_fraction = 1.0 - mobile_fraction

    result['tau'] = float(tau)
    result['mobile_fraction'] = float(mobile_fraction)
    result['immobile_fraction'] = float(immobile_fraction)
    result['r_squared'] = float(r_squared)
    result['fit_curve'] = fit_curve

    return result


def _double_exp_model(t, A1, tau1, A2, tau2, offset):
    """y = A1 * (1 - exp(-t / tau1)) + A2 * (1 - exp(-t / tau2)) + offset"""
    return A1 * (1.0 - np.exp(-t / tau1)) + A2 * (1.0 - np.exp(-t / tau2)) + offset


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

    A_total = float(np.max(intensities) - np.min(intensities))
    t_span = float(time[-1] - time[0])
    offset0 = float(np.min(intensities))

    if t_span <= 0:
        t_span = 1.0

    # Initial guesses: fast and slow components
    A1_0 = A_total * 0.6
    tau1_0 = t_span / 4.0
    A2_0 = A_total * 0.4
    tau2_0 = t_span
    p0 = [A1_0, tau1_0, A2_0, tau2_0, offset0]

    try:
        popt, _ = curve_fit(
            _double_exp_model, time, intensities,
            p0=p0,
            bounds=(
                [0, 1e-12, 0, 1e-12, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf],
            ),
            maxfev=20000,
        )
        A1, tau1, A2, tau2, offset = popt
    except (RuntimeError, ValueError):
        return result

    # Ensure tau1 <= tau2 (fast component first)
    if tau1 > tau2:
        A1, A2 = A2, A1
        tau1, tau2 = tau2, tau1

    fit_curve = _double_exp_model(time, A1, tau1, A2, tau2, offset)

    # R-squared
    ss_res = np.sum((intensities - fit_curve) ** 2)
    ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

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
    result['r_squared'] = float(r_squared)
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
            f"Soumpasis equation is only valid for 2-D; got dimensionality={dimensionality}"
        )
    if half_time <= 0:
        raise ValueError(f"half_time must be positive; got {half_time}")

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
            g.alert(f"Bleach frame ({bleach_frame}) exceeds stack length ({n_frames}).")
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
                    g.status_msg('ROI mask is empty; analysing full image.')
            else:
                g.status_msg('No ROI found; analysing full image.')

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
            # Use amplitude-weighted average tau for half-time
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
        print(f"FRAP Analysis: {name}")
        print("=" * 55)
        print(f"  Model              = {model}")
        print(f"  Pre-bleach frames  = {pre_bleach_frames}")
        print(f"  Bleach frame       = {bleach_frame}")
        print(f"  Bleach radius (um) = {bleach_radius_um:.2f}")

        if model == 'double_exponential':
            print(f"  tau1 (frames)      = {fit_result['tau1']:.4f}")
            print(f"  tau2 (frames)      = {fit_result['tau2']:.4f}")
            print(f"  Fraction 1         = {fit_result['fraction1']:.4f}")
            print(f"  Fraction 2         = {fit_result['fraction2']:.4f}")
            print(f"  tau_eff (frames)   = {tau_eff:.4f}")
        else:
            print(f"  tau (frames)       = {fit_result.get('tau', np.nan):.4f}")

        print(f"  Mobile fraction    = {fit_result['mobile_fraction']:.4f}")
        print(f"  Immobile fraction  = {fit_result['immobile_fraction']:.4f}")
        print(f"  R-squared          = {fit_result['r_squared']:.4f}")
        print(f"  t_half (frames)    = {t_half:.4f}")
        print(f"  D (um^2/frame)     = {D:.6f}")
        print("=" * 55)

        # --- Store in metadata --------------------------------------------
        window.metadata['frap'] = results

        # --- Status bar ----------------------------------------------------
        g.status_msg(
            f'FRAP: mobile={fit_result["mobile_fraction"]:.3f}, '
            f't_half={t_half:.2f}, R^2={fit_result["r_squared"]:.3f}'
        )

        # --- Plot ----------------------------------------------------------
        self._plot_widget = QtWidgets.QWidget()
        self._plot_widget.setWindowTitle(f'FRAP Recovery: {name}')
        self._plot_widget.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self._plot_widget)

        pw = pg.PlotWidget()
        pw.setLabel('bottom', 'Frame')
        pw.setLabel('left', 'Normalized Intensity')
        pw.setTitle(f'FRAP Recovery: {name}')
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
                label=f't_half={t_half:.2f}',
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
                f"tau1={fit_result['tau1']:.2f}  tau2={fit_result['tau2']:.2f}\n"
                f"mobile={fit_result['mobile_fraction']:.3f}\n"
                f"t_half={t_half:.2f}  R^2={fit_result['r_squared']:.3f}"
            )
        else:
            stats_text = (
                f"tau={fit_result.get('tau', np.nan):.2f}\n"
                f"mobile={fit_result['mobile_fraction']:.3f}\n"
                f"t_half={t_half:.2f}  R^2={fit_result['r_squared']:.3f}"
            )
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
