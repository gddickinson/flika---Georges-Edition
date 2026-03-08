# -*- coding: utf-8 -*-
"""Point Spread Function models for microscopy simulation."""
import numpy as np
from scipy.special import j1


def gaussian_psf_2d(shape, sigma_xy, center=None):
    """Generate a 2D Gaussian PSF.

    Parameters
    ----------
    shape : tuple
        (H, W) output shape.
    sigma_xy : float
        Standard deviation in pixels.
    center : tuple or None
        (y, x) center; defaults to image center.

    Returns
    -------
    ndarray
        Normalized 2D PSF.
    """
    h, w = shape
    if center is None:
        center = ((h - 1) / 2.0, (w - 1) / 2.0)
    y, x = np.ogrid[:h, :w]
    r2 = (y - center[0])**2 + (x - center[1])**2
    psf = np.exp(-r2 / (2 * sigma_xy**2))
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def gaussian_psf_3d(shape, sigma_xy, sigma_z, center=None):
    """Generate a 3D Gaussian PSF.

    Parameters
    ----------
    shape : tuple
        (D, H, W) output shape.
    sigma_xy : float
        Lateral standard deviation in pixels.
    sigma_z : float
        Axial standard deviation in pixels.
    center : tuple or None
        (z, y, x) center; defaults to volume center.

    Returns
    -------
    ndarray
        Normalized 3D PSF.
    """
    d, h, w = shape
    if center is None:
        center = ((d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0)
    z, y, x = np.ogrid[:d, :h, :w]
    r2 = ((z - center[0]) / sigma_z)**2 + \
         ((y - center[1]) / sigma_xy)**2 + \
         ((x - center[2]) / sigma_xy)**2
    psf = np.exp(-r2 / 2.0)
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def airy_psf_2d(shape, wavelength, NA, pixel_size):
    """Generate a 2D Airy disk PSF.

    Parameters
    ----------
    shape : tuple
        (H, W) output shape.
    wavelength : float
        Emission wavelength in nm.
    NA : float
        Numerical aperture.
    pixel_size : float
        Pixel size in microns.

    Returns
    -------
    ndarray
        Normalized 2D Airy PSF.
    """
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.ogrid[:h, :w]
    # Convert to physical coordinates in microns
    ry = (y - cy) * pixel_size
    rx = (x - cx) * pixel_size
    r = np.sqrt(ry**2 + rx**2)
    # Airy pattern argument
    wl_um = wavelength / 1000.0  # nm to µm
    k = 2 * np.pi * NA / wl_um
    v = k * r
    # Handle center singularity
    psf = np.ones_like(v, dtype=float)
    mask = v > 0
    psf[mask] = (2 * j1(v[mask]) / v[mask])**2
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


def born_wolf_psf_3d(shape, wavelength, NA, n, pixel_size, z_step):
    """Generate a 3D PSF using the scalar Born-Wolf model.

    Uses numerical integration of the scalar diffraction integral
    with Bessel J0.

    Parameters
    ----------
    shape : tuple
        (D, H, W) output shape.
    wavelength : float
        Emission wavelength in nm.
    NA : float
        Numerical aperture.
    n : float
        Refractive index of immersion medium.
    pixel_size : float
        Lateral pixel size in microns.
    z_step : float
        Axial step size in microns.

    Returns
    -------
    ndarray
        Normalized 3D PSF.
    """
    from scipy.special import j0
    d, h, w = shape
    wl_um = wavelength / 1000.0
    k = 2 * np.pi * n / wl_um
    alpha = np.arcsin(NA / n)

    # Radial coordinates
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y_coords = (np.arange(h) - cy) * pixel_size
    x_coords = (np.arange(w) - cx) * pixel_size
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    r_lateral = np.sqrt(yy**2 + xx**2)

    # Axial coordinates
    cz = (d - 1) / 2.0
    z_coords = (np.arange(d) - cz) * z_step

    # Integration over pupil (Simpson's rule)
    n_pts = 50
    theta = np.linspace(0, alpha, n_pts)
    dtheta = theta[1] - theta[0] if n_pts > 1 else 1.0

    psf = np.zeros((d, h, w))
    for zi, zv in enumerate(z_coords):
        integral = np.zeros((h, w), dtype=complex)
        for ti, th in enumerate(theta):
            # Simpson weights
            if ti == 0 or ti == n_pts - 1:
                sw = 1.0
            elif ti % 2 == 1:
                sw = 4.0
            else:
                sw = 2.0
            rho = k * np.sin(th) * r_lateral
            phase = k * zv * np.cos(th)
            integrand = np.sqrt(np.cos(th)) * np.sin(th) * \
                j0(rho) * np.exp(1j * phase)
            integral += sw * integrand
        integral *= dtheta / 3.0
        psf[zi] = np.abs(integral)**2

    # Normalize per z-plane
    for zi in range(d):
        total = psf[zi].sum()
        if total > 0:
            psf[zi] /= total
    return psf


def vectorial_psf_3d(shape, wavelength, NA, n, pixel_size, z_step,
                     polarization='circular'):
    """Generate a 3D PSF using the vectorial (Richards-Wolf) model.

    Parameters
    ----------
    shape : tuple
        (D, H, W) output shape.
    wavelength : float
        Emission wavelength in nm.
    NA : float
        Numerical aperture.
    n : float
        Refractive index.
    pixel_size : float
        Lateral pixel size in microns.
    z_step : float
        Axial step size in microns.
    polarization : str
        'circular', 'x', or 'y'.

    Returns
    -------
    ndarray
        Normalized 3D PSF.
    """
    from scipy.special import j0, jv
    d, h, w = shape
    wl_um = wavelength / 1000.0
    k = 2 * np.pi * n / wl_um
    alpha = np.arcsin(NA / n)

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y_coords = (np.arange(h) - cy) * pixel_size
    x_coords = (np.arange(w) - cx) * pixel_size
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    r_lat = np.sqrt(yy**2 + xx**2)
    phi = np.arctan2(yy, xx)

    cz = (d - 1) / 2.0
    z_coords = (np.arange(d) - cz) * z_step

    n_pts = 50
    theta = np.linspace(0, alpha, n_pts)
    dtheta = theta[1] - theta[0] if n_pts > 1 else 1.0

    psf = np.zeros((d, h, w))
    for zi, zv in enumerate(z_coords):
        I0 = np.zeros((h, w), dtype=complex)
        I1 = np.zeros((h, w), dtype=complex)
        I2 = np.zeros((h, w), dtype=complex)
        for ti, th in enumerate(theta):
            if ti == 0 or ti == n_pts - 1:
                sw = 1.0
            elif ti % 2 == 1:
                sw = 4.0
            else:
                sw = 2.0
            rho = k * np.sin(th) * r_lat
            phase = np.exp(1j * k * zv * np.cos(th))
            apod = np.sqrt(np.cos(th))
            sin_t = np.sin(th)
            cos_t = np.cos(th)
            I0 += sw * apod * sin_t * (1 + cos_t) * j0(rho) * phase
            I1 += sw * apod * sin_t**2 * jv(1, rho) * phase
            I2 += sw * apod * sin_t * (1 - cos_t) * jv(2, rho) * phase
        I0 *= dtheta / 3.0
        I1 *= dtheta / 3.0
        I2 *= dtheta / 3.0

        if polarization == 'circular':
            psf[zi] = (np.abs(I0)**2 + 2 * np.abs(I1)**2 +
                       np.abs(I2)**2) / 2.0
        elif polarization == 'x':
            Ex = I0 + I2 * np.cos(2 * phi)
            Ey = I2 * np.sin(2 * phi)
            Ez = -2j * I1 * np.cos(phi)
            psf[zi] = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
        else:  # 'y'
            Ex = I2 * np.sin(2 * phi)
            Ey = I0 - I2 * np.cos(2 * phi)
            Ez = -2j * I1 * np.sin(phi)
            psf[zi] = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

    # Normalize per z-plane
    for zi in range(d):
        total = psf[zi].sum()
        if total > 0:
            psf[zi] /= total
    return psf


def astigmatic_psf_3d(shape, sigma_x_func, sigma_y_func, z_positions):
    """Generate an astigmatic 3D PSF for 3D SMLM.

    Parameters
    ----------
    shape : tuple
        (D, H, W) output shape.
    sigma_x_func : callable
        Function mapping z (µm) → sigma_x (pixels).
    sigma_y_func : callable
        Function mapping z (µm) → sigma_y (pixels).
    z_positions : array-like
        Z positions in microns for each slice.

    Returns
    -------
    ndarray
        Normalized 3D PSF.
    """
    d, h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.ogrid[:h, :w]
    psf = np.zeros((d, h, w))
    for zi, zv in enumerate(z_positions[:d]):
        sx = sigma_x_func(zv)
        sy = sigma_y_func(zv)
        g = np.exp(-((y - cy)**2 / (2 * sy**2) +
                     (x - cx)**2 / (2 * sx**2)))
        total = g.sum()
        if total > 0:
            g /= total
        psf[zi] = g
    return psf


def generate_psf(model='gaussian', **kwargs):
    """Factory function for PSF generation.

    Parameters
    ----------
    model : str
        One of 'gaussian', 'gaussian_3d', 'airy', 'born_wolf',
        'vectorial', 'astigmatic'.
    **kwargs
        Parameters passed to the selected PSF function.

    Returns
    -------
    ndarray
        Generated PSF.
    """
    models = {
        'gaussian': gaussian_psf_2d,
        'gaussian_3d': gaussian_psf_3d,
        'airy': airy_psf_2d,
        'born_wolf': born_wolf_psf_3d,
        'vectorial': vectorial_psf_3d,
        'astigmatic': astigmatic_psf_3d,
    }
    if model not in models:
        raise ValueError(f"Unknown PSF model '{model}'. "
                         f"Available: {list(models.keys())}")
    return models[model](**kwargs)
