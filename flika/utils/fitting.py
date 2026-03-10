# -*- coding: utf-8 -*-
"""Shared curve-fitting utilities for exponential models.

Consolidates duplicated fitting code from:
- process/frap.py (single/double exponential recovery)
- process/filters.py (exponential decay for bleach correction)
- viewers/diffusion_plot.py (R-squared calculation)

All functions are pure NumPy/SciPy with no Qt dependencies.
"""
import numpy as np
from scipy.optimize import curve_fit


__all__ = [
    'r_squared',
    'exp_decay',
    'exp_recovery',
    'double_exp_recovery',
    'fit_exponential_decay',
    'fit_exponential_recovery',
    'fit_double_exponential_recovery',
]


# ---------------------------------------------------------------------------
# Goodness-of-fit
# ---------------------------------------------------------------------------

def r_squared(y_true, y_predicted):
    """Compute R-squared (coefficient of determination).

    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_predicted : array-like
        Predicted (fitted) values.

    Returns
    -------
    float
        R-squared value.  Returns ``numpy.nan`` if the total sum of squares
        is zero (constant data).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_predicted = np.asarray(y_predicted, dtype=np.float64)
    ss_res = np.sum((y_true - y_predicted) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def exp_decay(t, amplitude, rate, offset):
    """Exponential decay model.

    .. math:: y = amplitude \\cdot e^{-rate \\cdot t} + offset

    Parameters
    ----------
    t : array-like
        Independent variable (e.g. time).
    amplitude : float
        Decay amplitude.
    rate : float
        Decay rate constant (> 0).
    offset : float
        Asymptotic baseline.

    Returns
    -------
    numpy.ndarray
    """
    return amplitude * np.exp(-rate * np.asarray(t, dtype=np.float64)) + offset


def exp_recovery(t, amplitude, tau, offset):
    """Single-exponential recovery model.

    .. math:: y = amplitude \\cdot (1 - e^{-t/\\tau}) + offset

    Parameters
    ----------
    t : array-like
        Independent variable (e.g. time).
    amplitude : float
        Recovery amplitude.
    tau : float
        Recovery time constant (> 0).
    offset : float
        Baseline offset.

    Returns
    -------
    numpy.ndarray
    """
    return amplitude * (1.0 - np.exp(-np.asarray(t, dtype=np.float64) / tau)) + offset


def double_exp_recovery(t, A1, tau1, A2, tau2, offset):
    """Double-exponential recovery model.

    .. math:: y = A_1(1-e^{-t/\\tau_1}) + A_2(1-e^{-t/\\tau_2}) + offset

    Parameters
    ----------
    t : array-like
        Independent variable.
    A1, A2 : float
        Amplitudes of the fast and slow components.
    tau1, tau2 : float
        Time constants of the fast and slow components.
    offset : float
        Baseline offset.

    Returns
    -------
    numpy.ndarray
    """
    t = np.asarray(t, dtype=np.float64)
    return A1 * (1.0 - np.exp(-t / tau1)) + A2 * (1.0 - np.exp(-t / tau2)) + offset


# ---------------------------------------------------------------------------
# Fitting wrappers
# ---------------------------------------------------------------------------

def fit_exponential_decay(time, data, p0=None):
    """Fit an exponential decay to data.

    Model: ``amplitude * exp(-rate * t) + offset``

    Parameters
    ----------
    time : array-like
        Independent variable values.
    data : array-like
        Dependent variable values to fit.
    p0 : tuple of 3 floats, optional
        Initial guesses ``(amplitude, rate, offset)``.  If *None*, guesses
        are derived from the data.

    Returns
    -------
    params : tuple
        ``(amplitude, rate, offset)``
    r2 : float
        R-squared goodness of fit.

    Raises
    ------
    RuntimeError
        If the curve fit fails to converge.
    """
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    if p0 is None:
        amp0 = float(data[0] - data[-1]) if len(data) > 1 else 1.0
        rate0 = 0.01
        off0 = float(data[-1]) if len(data) > 0 else 0.0
        p0 = (amp0, rate0, off0)

    popt, _ = curve_fit(
        exp_decay, time, data,
        p0=p0,
        maxfev=10000,
    )
    fitted = exp_decay(time, *popt)
    r2 = r_squared(data, fitted)
    return tuple(float(v) for v in popt), r2


def fit_exponential_recovery(time, data, p0=None):
    """Fit a single-exponential recovery to data.

    Model: ``amplitude * (1 - exp(-t / tau)) + offset``

    Parameters
    ----------
    time : array-like
        Independent variable values.
    data : array-like
        Dependent variable values to fit.
    p0 : tuple of 3 floats, optional
        Initial guesses ``(amplitude, tau, offset)``.  If *None*, guesses
        are derived from the data.

    Returns
    -------
    params : tuple
        ``(amplitude, tau, offset)``
    r2 : float
        R-squared goodness of fit.

    Raises
    ------
    RuntimeError
        If the curve fit fails to converge.
    """
    time = np.asarray(time, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    if p0 is None:
        amp0 = float(np.max(data) - np.min(data))
        t_span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
        tau0 = max(t_span / 2.0, 1.0)
        off0 = float(np.min(data))
        p0 = (amp0, tau0, off0)

    popt, _ = curve_fit(
        exp_recovery, time, data,
        p0=p0,
        bounds=([0, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=10000,
    )
    fitted = exp_recovery(time, *popt)
    r2 = r_squared(data, fitted)
    return tuple(float(v) for v in popt), r2


def fit_double_exponential_recovery(time, data, p0=None):
    """Fit a double-exponential recovery to data.

    Model: ``A1*(1-exp(-t/tau1)) + A2*(1-exp(-t/tau2)) + offset``

    The returned parameters are sorted so that ``tau1 <= tau2`` (fast
    component first).

    Parameters
    ----------
    time : array-like
        Independent variable values.
    data : array-like
        Dependent variable values to fit.
    p0 : tuple of 5 floats, optional
        Initial guesses ``(A1, tau1, A2, tau2, offset)``.  If *None*,
        guesses are derived from the data.

    Returns
    -------
    params : tuple
        ``(A1, tau1, A2, tau2, offset)`` with ``tau1 <= tau2``.
    r2 : float
        R-squared goodness of fit.

    Raises
    ------
    RuntimeError
        If the curve fit fails to converge.
    """
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
        double_exp_recovery, time, data,
        p0=p0,
        bounds=(
            [0, 1e-12, 0, 1e-12, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
        maxfev=20000,
    )
    A1, tau1, A2, tau2, offset = popt

    # Ensure tau1 <= tau2 (fast component first)
    if tau1 > tau2:
        A1, A2 = A2, A1
        tau1, tau2 = tau2, tau1

    fitted = double_exp_recovery(time, A1, tau1, A2, tau2, offset)
    r2 = r_squared(data, fitted)
    return (float(A1), float(tau1), float(A2), float(tau2), float(offset)), r2
