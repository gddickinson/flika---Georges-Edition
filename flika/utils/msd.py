# -*- coding: utf-8 -*-
"""Shared Mean Squared Displacement (MSD) utilities.

Consolidates duplicated MSD computation from:
- viewers/diffusion_plot.py (_msd_single_track)
- spt/features/kinematic.py (msd_analysis)

All functions are pure NumPy with no Qt dependencies.
"""
import numpy as np


__all__ = [
    'msd_single_track',
    'msd_ensemble',
    'fit_msd_linear',
    'fit_msd_anomalous',
]


def msd_single_track(positions, max_lag=None):
    """Compute Mean Squared Displacement for a single track.

    Parameters
    ----------
    positions : ndarray, shape (N, 2) or (N, 3)
        XY or XYZ positions per frame (in physical units).
    max_lag : int, optional
        Maximum lag in frames.  Default: ``N // 4`` (at least 2).

    Returns
    -------
    lags : ndarray
        Lag values ``[1, 2, ..., actual_max]``.
    msd : ndarray
        Mean squared displacement for each lag.
    counts : ndarray
        Number of displacement pairs contributing to each lag.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n < 2:
        return np.array([]), np.array([]), np.array([], dtype=int)

    if max_lag is None:
        max_lag = max(n // 4, 2)
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


def msd_ensemble(tracks_dict, max_lag=None):
    """Compute ensemble-averaged MSD from multiple tracks.

    Parameters
    ----------
    tracks_dict : dict
        Mapping ``{track_id: ndarray}`` where each array has shape
        ``(N, 2)`` or ``(N, 3)`` with XY(Z) positions in physical units.
    max_lag : int, optional
        Maximum lag in frames.  Default: ``N // 4`` per track.

    Returns
    -------
    lags : ndarray
        Sorted unique lag values present across all tracks.
    msd_mean : ndarray
        Weighted mean MSD at each lag.
    msd_sem : ndarray
        Standard error of the mean at each lag.
    """
    lag_data = {}  # lag -> list of (msd_value, count)

    for tid, positions in tracks_dict.items():
        positions = np.asarray(positions, dtype=np.float64)
        if len(positions) < 2:
            continue
        lags_t, msd_t, counts_t = msd_single_track(positions, max_lag=max_lag)
        for i, lag in enumerate(lags_t):
            lag_data.setdefault(int(lag), []).append((msd_t[i], counts_t[i]))

    if not lag_data:
        return np.array([]), np.array([]), np.array([])

    sorted_lags = np.array(sorted(lag_data.keys()))
    msd_mean = np.empty(len(sorted_lags))
    msd_sem = np.empty(len(sorted_lags))

    for i, lag in enumerate(sorted_lags):
        entries = lag_data[lag]
        values = np.array([e[0] for e in entries])
        weights = np.array([e[1] for e in entries], dtype=np.float64)
        total_weight = np.sum(weights)
        if total_weight > 0:
            wm = np.sum(values * weights) / total_weight
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

    return sorted_lags, msd_mean, msd_sem


def fit_msd_linear(lags, msd, ndim=2):
    """Fit MSD with a linear diffusion model.

    Model: ``MSD = 2 * ndim * D * t``

    Uses the first 10 lag points (or fewer if not available) for fitting.

    Parameters
    ----------
    lags : array-like
        Lag values (in frames or time units).
    msd : array-like
        MSD values corresponding to each lag.
    ndim : int
        Spatial dimensionality (default 2).

    Returns
    -------
    float
        Diffusion coefficient *D*.  Clamped to >= 0.
    """
    lags = np.asarray(lags, dtype=np.float64)
    msd = np.asarray(msd, dtype=np.float64)

    if len(lags) < 2:
        return 0.0

    fit_n = min(len(lags), 10)
    coeffs = np.polyfit(lags[:fit_n], msd[:fit_n], 1)
    D = coeffs[0] / (2.0 * ndim)
    return max(float(D), 0.0)


def fit_msd_anomalous(lags, msd, ndim=2):
    """Fit MSD with an anomalous diffusion model.

    Model: ``MSD = 2 * ndim * D * t^alpha``

    Uses a log-log linear fit to extract *D* and *alpha*.

    Parameters
    ----------
    lags : array-like
        Lag values (in frames or time units).
    msd : array-like
        MSD values corresponding to each lag.
    ndim : int
        Spatial dimensionality (default 2).

    Returns
    -------
    D : float
        Generalised diffusion coefficient.  Clamped to >= 0.
    alpha : float
        Anomalous exponent (1.0 = normal diffusion).
    """
    lags = np.asarray(lags, dtype=np.float64)
    msd = np.asarray(msd, dtype=np.float64)

    if len(lags) < 3:
        return 0.0, 1.0

    valid = msd > 0
    if np.sum(valid) < 2:
        return 0.0, 1.0

    log_lags = np.log(lags[valid])
    log_msd = np.log(msd[valid])
    coeffs = np.polyfit(log_lags, log_msd, 1)
    alpha = float(coeffs[0])
    # intercept = log(2 * ndim * D)  =>  D = exp(intercept) / (2 * ndim)
    D = float(np.exp(coeffs[1]) / (2.0 * ndim))
    D = max(D, 0.0)

    return D, alpha
