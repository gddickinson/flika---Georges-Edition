"""Super-resolution image rendering from localization data.

Provides four rendering modes matching the original ThunderSTORM ImageJ
plugin (Ovesny et al., Bioinformatics 2014):

1. **Gaussian rendering** — each localization rendered as 2D Gaussian
   with σ = localization precision.
2. **Average shifted histograms (ASH)** — sub-pixel histogram averaged
   over multiple shifted grids (Scott 1985).
3. **Scatter plot** — simple point plot at sub-pixel positions.
4. **Normalized Gaussian** — Gaussians normalized by peak (uniform
   brightness regardless of photon count).

All renderers accept localizations as (N, 3+) arrays with columns
``[x, y, intensity, ...]`` and optional per-localization uncertainty
(column index configurable).

Usage::

    from flika.spt.detection.rendering import render_localizations
    sr_image = render_localizations(locs, method='gaussian',
                                     pixel_size=10.0, sigma_col=5)
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def render_localizations(localizations, method='gaussian',
                         pixel_size=10.0, image_size=None,
                         sigma_col=None, fixed_sigma=None,
                         magnification=10, n_shifts=5):
    """Render a super-resolution image from localization data.

    Parameters
    ----------
    localizations : ndarray
        (N, 3+) array. Columns 0,1 = x,y (in original pixel units).
        Column 2 = intensity.  Additional columns may include sigma
        and uncertainty.
    method : str
        Rendering method: 'gaussian', 'histogram', 'ash', 'scatter',
        'normalized_gaussian'.  Default 'gaussian'.
    pixel_size : float
        Output pixel size in original-pixel units.  E.g. ``pixel_size=0.1``
        gives 10x super-resolution.  Default 10.0 (nm-scale if input is nm).
    image_size : tuple of int, optional
        (height, width) of the output image in output pixels.  If None,
        auto-computed from localization extent.
    sigma_col : int, optional
        Column index for per-localization sigma/uncertainty.  If provided,
        each localization is rendered with its own σ.
    fixed_sigma : float, optional
        Fixed rendering sigma in original-pixel units.  Used when
        ``sigma_col`` is None.  Default: ``pixel_size``.
    magnification : int
        For 'ash' method: number of sub-pixel shifts per axis.  Default 10.
    n_shifts : int
        Alias for magnification (backward compat).  If magnification is
        not explicitly set and n_shifts differs, n_shifts is used.

    Returns
    -------
    ndarray
        2D rendered image (float64).
    """
    if len(localizations) == 0:
        if image_size is not None:
            return np.zeros(image_size, dtype=np.float64)
        return np.zeros((1, 1), dtype=np.float64)

    x = localizations[:, 0]
    y = localizations[:, 1]
    intensities = localizations[:, 2] if localizations.shape[1] > 2 else None

    # Determine output image size
    if image_size is None:
        x_max = np.max(x)
        y_max = np.max(y)
        width = int(np.ceil(x_max / pixel_size)) + 2
        height = int(np.ceil(y_max / pixel_size)) + 2
        image_size = (height, width)

    method = method.lower()
    if method == 'gaussian':
        return _render_gaussian(x, y, intensities, pixel_size, image_size,
                                sigma_col, localizations, fixed_sigma,
                                normalize_peak=False)
    elif method == 'normalized_gaussian':
        return _render_gaussian(x, y, intensities, pixel_size, image_size,
                                sigma_col, localizations, fixed_sigma,
                                normalize_peak=True)
    elif method == 'histogram':
        return _render_histogram(x, y, intensities, pixel_size, image_size)
    elif method in ('ash', 'average_shifted_histogram'):
        return _render_ash(x, y, intensities, pixel_size, image_size,
                           magnification)
    elif method == 'scatter':
        return _render_scatter(x, y, pixel_size, image_size)
    else:
        raise ValueError(f"Unknown rendering method: {method!r}. "
                         f"Available: gaussian, normalized_gaussian, "
                         f"histogram, ash, scatter.")


def _render_gaussian(x, y, intensities, pixel_size, image_size,
                     sigma_col, localizations, fixed_sigma,
                     normalize_peak=False):
    """Render each localization as a 2D Gaussian."""
    height, width = image_size
    image = np.zeros((height, width), dtype=np.float64)

    if fixed_sigma is None:
        fixed_sigma = pixel_size

    # Rendering radius in output pixels (3σ cutoff)
    max_sigma_px = fixed_sigma / pixel_size
    if sigma_col is not None and localizations.shape[1] > sigma_col:
        sigmas = localizations[:, sigma_col]
        max_sigma_px = max(max_sigma_px, np.max(sigmas) / pixel_size)
    render_radius = int(np.ceil(3.0 * max_sigma_px)) + 1

    for i in range(len(x)):
        # Output pixel coordinates
        cx = x[i] / pixel_size
        cy = y[i] / pixel_size
        intensity = intensities[i] if intensities is not None else 1.0

        # Per-localization sigma
        if sigma_col is not None and localizations.shape[1] > sigma_col:
            sigma_px = localizations[i, sigma_col] / pixel_size
        else:
            sigma_px = fixed_sigma / pixel_size

        sigma_px = max(sigma_px, 0.1)

        # Pixel range
        ix0 = max(0, int(cx) - render_radius)
        ix1 = min(width, int(cx) + render_radius + 1)
        iy0 = max(0, int(cy) - render_radius)
        iy1 = min(height, int(cy) + render_radius + 1)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        # Build Gaussian patch
        yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        gauss = np.exp(-r2 / (2.0 * sigma_px ** 2))

        if normalize_peak:
            # Normalized: peak = 1, uniform brightness
            image[iy0:iy1, ix0:ix1] += gauss
        else:
            # Weighted by intensity, normalized so integral = intensity
            norm = 2.0 * np.pi * sigma_px ** 2
            image[iy0:iy1, ix0:ix1] += intensity * gauss / norm

    return image


def _render_histogram(x, y, intensities, pixel_size, image_size):
    """Simple 2D histogram rendering."""
    height, width = image_size
    image = np.zeros((height, width), dtype=np.float64)

    xi = (x / pixel_size).astype(int)
    yi = (y / pixel_size).astype(int)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    if intensities is not None:
        for xv, yv, iv in zip(xi[valid], yi[valid], intensities[valid]):
            image[yv, xv] += iv
    else:
        for xv, yv in zip(xi[valid], yi[valid]):
            image[yv, xv] += 1.0

    return image


def _render_ash(x, y, intensities, pixel_size, image_size, magnification):
    """Average Shifted Histogram rendering (Scott 1985).

    Computes histograms at ``magnification`` shifted grid positions
    per axis and averages them.  This produces smoother results than
    a single histogram without the computational cost of Gaussian
    rendering.
    """
    height, width = image_size
    accumulator = np.zeros((height, width), dtype=np.float64)

    m = max(int(magnification), 1)
    delta = pixel_size / m

    for sy in range(m):
        for sx in range(m):
            shift_x = sx * delta
            shift_y = sy * delta
            xi = ((x + shift_x) / pixel_size).astype(int)
            yi = ((y + shift_y) / pixel_size).astype(int)
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

            if intensities is not None:
                for xv, yv, iv in zip(xi[valid], yi[valid],
                                      intensities[valid]):
                    accumulator[yv, xv] += iv
            else:
                for xv, yv in zip(xi[valid], yi[valid]):
                    accumulator[yv, xv] += 1.0

    accumulator /= (m * m)
    return accumulator


def _render_scatter(x, y, pixel_size, image_size):
    """Scatter plot rendering — one count per localization."""
    height, width = image_size
    image = np.zeros((height, width), dtype=np.float64)

    xi = (x / pixel_size).astype(int)
    yi = (y / pixel_size).astype(int)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    for xv, yv in zip(xi[valid], yi[valid]):
        image[yv, xv] = 1.0

    return image
