"""2D Gaussian fitting utilities.

Provides symmetric and rotatable (elliptical) 2D Gaussian fitting
via least-squares optimisation.  These are useful for sub-pixel
localisation of spots, PSF characterisation, and similar tasks.

Example::

    from flika.utils.gaussian_fit import fit_gaussian_2d, fit_rotatable_gaussian_2d

    # Symmetric Gaussian: returns (params, fitted_image)
    #   params = [x0, y0, sigma, amplitude]
    params, fitted = fit_gaussian_2d(image_patch)

    # Elliptical / rotatable: returns (params, fitted_image)
    #   params = [x0, y0, sigma_x, sigma_y, angle_deg, amplitude]
    params, fitted = fit_rotatable_gaussian_2d(image_patch)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Symmetric 2D Gaussian
# ---------------------------------------------------------------------------

def gaussian_2d(x, y, x0, y0, sigma, amplitude):
    """Evaluate a symmetric 2D Gaussian on meshgrid arrays *x*, *y*."""
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def _sym_residuals(p, data, x, y):
    return (data - gaussian_2d(x, y, *p)).ravel()


def fit_gaussian_2d(image, p0=None, bounds=None):
    """Fit a symmetric 2D Gaussian to *image* (2D array).

    Parameters
    ----------
    image : ndarray (M, N)
    p0 : list, optional
        Initial guess ``[x0, y0, sigma, amplitude]``.
        Defaults to centre of image, sigma=2, amplitude=max(image).
    bounds : tuple of (lower, upper), optional
        Each is a list of length 4.

    Returns
    -------
    params : ndarray  — ``[x0, y0, sigma, amplitude]``
    fitted : ndarray (M, N) — the fitted Gaussian evaluated on the grid
    """
    ny, nx = image.shape  # rows=Y, cols=X but we keep (row, col) = (x, y) convention
    mx, my = image.shape
    xs = np.arange(mx)
    ys = np.arange(my)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    if p0 is None:
        p0 = [mx / 2.0, my / 2.0, 2.0, float(np.max(image))]

    if bounds is not None:
        lb, ub = bounds
        result = least_squares(_sym_residuals, p0, args=(image, X, Y),
                               bounds=(lb, ub), max_nfev=200)
    else:
        result = least_squares(_sym_residuals, p0, args=(image, X, Y), max_nfev=200)

    params = result.x
    fitted = gaussian_2d(X, Y, *params)
    return params, fitted


# ---------------------------------------------------------------------------
# Rotatable (elliptical) 2D Gaussian
# ---------------------------------------------------------------------------

def gaussian_2d_rotatable(x, y, x0, y0, sigma_x, sigma_y, angle_deg, amplitude):
    """Evaluate a rotatable elliptical 2D Gaussian."""
    cos_a = np.cos(np.radians(angle_deg))
    sin_a = np.sin(np.radians(angle_deg))
    dx = x - x0
    dy = y - y0
    xr = cos_a * dx - sin_a * dy
    yr = sin_a * dx + cos_a * dy
    return amplitude * np.exp(-(xr ** 2 / (2.0 * sigma_x ** 2) + yr ** 2 / (2.0 * sigma_y ** 2)))


def _rot_residuals(p, data, x, y):
    return (data - gaussian_2d_rotatable(x, y, *p)).ravel()


def fit_rotatable_gaussian_2d(image, p0=None, bounds=None):
    """Fit a rotatable elliptical 2D Gaussian to *image*.

    Parameters
    ----------
    image : ndarray (M, N)
    p0 : list, optional
        ``[x0, y0, sigma_x, sigma_y, angle_deg, amplitude]``
    bounds : tuple of (lower, upper), optional

    Returns
    -------
    params : ndarray — ``[x0, y0, sigma_x, sigma_y, angle_deg, amplitude]``
    fitted : ndarray (M, N)
    """
    mx, my = image.shape
    xs = np.arange(mx)
    ys = np.arange(my)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    if p0 is None:
        p0 = [mx / 2.0, my / 2.0, 2.0, 2.0, 0.0, float(np.max(image))]

    if bounds is not None:
        lb, ub = bounds
        result = least_squares(_rot_residuals, p0, args=(image, X, Y),
                               bounds=(lb, ub), max_nfev=200)
    else:
        result = least_squares(_rot_residuals, p0, args=(image, X, Y), max_nfev=200)

    params = result.x
    fitted = gaussian_2d_rotatable(X, Y, *params)
    return params, fitted


# ---------------------------------------------------------------------------
# Multi-Gaussian fitting
# ---------------------------------------------------------------------------

def fit_multi_gaussian_2d(image, n_gaussians, p0=None, bounds=None):
    """Fit *n_gaussians* symmetric 2D Gaussians simultaneously.

    Parameters
    ----------
    image : ndarray (M, N)
    n_gaussians : int
    p0 : list, optional
        Flattened ``[x0_1, y0_1, sigma_1, amp_1, x0_2, ...]``
    bounds : tuple of (lower, upper), optional

    Returns
    -------
    params : ndarray — shape (n_gaussians, 4)
    fitted : ndarray (M, N) — sum of all fitted Gaussians
    """
    mx, my = image.shape
    xs = np.arange(mx)
    ys = np.arange(my)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    if p0 is None:
        p0 = []
        for _ in range(n_gaussians):
            p0.extend([mx / 2.0, my / 2.0, 2.0, float(np.max(image)) / n_gaussians])

    def _multi_residuals(p, data, x, y):
        model = np.zeros_like(data, dtype=np.float64)
        for i in range(0, len(p), 4):
            model += gaussian_2d(x, y, *p[i:i + 4])
        return (data - model).ravel()

    if bounds is not None:
        lb, ub = bounds
        result = least_squares(_multi_residuals, p0, args=(image, X, Y),
                               bounds=(lb, ub), max_nfev=500)
    else:
        result = least_squares(_multi_residuals, p0, args=(image, X, Y), max_nfev=500)

    params = result.x.reshape(n_gaussians, 4)
    fitted = np.zeros_like(image, dtype=np.float64)
    for i in range(n_gaussians):
        fitted += gaussian_2d(X, Y, *params[i])
    return params, fitted


def generate_gaussian_kernel(size, sigma=1.15):
    """Generate a normalised 2D Gaussian kernel (zero-mean) of given *size*.

    Parameters
    ----------
    size : int
        Must be odd.
    sigma : float

    Returns
    -------
    kernel : ndarray (size, size)
    """
    assert size % 2 == 1, "size must be odd"
    xs = np.arange(size)
    ys = np.arange(size)
    centre = size // 2
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    kernel = np.exp(-((X - centre) ** 2 + (Y - centre) ** 2) / (2.0 * sigma ** 2))
    kernel -= kernel.mean()
    return kernel
