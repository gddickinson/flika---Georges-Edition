# -*- coding: utf-8 -*-
"""Modality-specific optical models for microscopy simulation."""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# TIRF
# ---------------------------------------------------------------------------

def tirf_excitation_profile(z, penetration_depth=100.0):
    """TIRF evanescent wave excitation profile.

    Parameters
    ----------
    z : ndarray or float
        Axial position(s) in nm.
    penetration_depth : float
        Evanescent field penetration depth in nm.

    Returns
    -------
    ndarray
        Excitation intensity I(z) = exp(-z/d).
    """
    return np.exp(-np.asarray(z, dtype=float) / penetration_depth)


def tirf_penetration_depth(wavelength, NA, n_glass=1.52, n_sample=1.33,
                           angle=None):
    """Calculate TIRF evanescent field penetration depth.

    Parameters
    ----------
    wavelength : float
        Excitation wavelength in nm.
    NA : float
        Numerical aperture.
    n_glass : float
        Refractive index of glass.
    n_sample : float
        Refractive index of sample.
    angle : float or None
        Incidence angle in radians. If None, uses critical angle + 2 deg.

    Returns
    -------
    float
        Penetration depth in nm.
    """
    theta_c = np.arcsin(n_sample / n_glass)
    if angle is None:
        angle = theta_c + np.radians(2)
    d = wavelength / (4 * np.pi * np.sqrt(
        n_glass**2 * np.sin(angle)**2 - n_sample**2))
    return d


# ---------------------------------------------------------------------------
# Confocal
# ---------------------------------------------------------------------------

def confocal_detection_psf(psf_exc, pinhole_au=1.0, wavelength=520.0,
                           NA=1.4):
    """Apply confocal pinhole to detection PSF.

    Parameters
    ----------
    psf_exc : ndarray
        3D excitation PSF.
    pinhole_au : float
        Pinhole diameter in Airy units.
    wavelength : float
        Emission wavelength in nm.
    NA : float
        Numerical aperture.

    Returns
    -------
    ndarray
        Effective confocal PSF (PSF_exc * pinhole_mask).
    """
    d, h, w = psf_exc.shape
    # Airy unit = 0.61 * lambda / NA (in lateral pixels)
    # We approximate by applying a Gaussian detection PSF
    # with width determined by pinhole
    airy_radius = 0.61 * wavelength / NA / 1000.0  # in µm (approx)

    # Detection PSF is broadened by pinhole convolution
    # For pinhole = 1 AU, detection PSF ≈ excitation PSF
    # Effective PSF = PSF_exc * PSF_det
    # Approximate PSF_det as scaled PSF_exc
    scale = max(0.5, pinhole_au)
    psf_det = ndimage.gaussian_filter(psf_exc, sigma=(0.5 * scale,
                                                       scale, scale))
    effective = psf_exc * psf_det
    # Normalize per z-plane
    for zi in range(d):
        total = effective[zi].sum()
        if total > 0:
            effective[zi] /= total
    return effective


def confocal_scan(sample_3d, psf, pinhole_au=1.0, z_sections=None):
    """Simulate confocal scanning of a 3D sample.

    Parameters
    ----------
    sample_3d : ndarray
        3D sample volume (D, H, W).
    psf : ndarray
        3D PSF.
    pinhole_au : float
        Pinhole size in Airy units.
    z_sections : list or None
        Which z-sections to image. None = all.

    Returns
    -------
    ndarray
        Optically sectioned stack.
    """
    d, h, w = sample_3d.shape
    if z_sections is None:
        z_sections = list(range(d))

    # Convolve sample with PSF
    convolved = fftconvolve(sample_3d, psf, mode='same')

    # Apply pinhole effect (suppress out-of-focus light)
    # Approximate: scale each z-plane by detection efficiency
    psf_d = psf.shape[0]
    center_z = psf_d // 2
    # Detection efficiency drops with defocus
    z_profile = psf[:, psf.shape[1] // 2, psf.shape[2] // 2]
    z_profile = z_profile / (z_profile.max() + 1e-10)

    result = np.zeros((len(z_sections), h, w))
    for i, zi in enumerate(z_sections):
        if 0 <= zi < d:
            result[i] = convolved[zi]
    return result


# ---------------------------------------------------------------------------
# Light-sheet / SPIM
# ---------------------------------------------------------------------------

def lightsheet_excitation(shape, sheet_thickness=2.0, direction='x'):
    """Generate light-sheet excitation profile.

    Parameters
    ----------
    shape : tuple
        (D, H, W) volume shape.
    sheet_thickness : float
        Sheet thickness (sigma) in pixels.
    direction : str
        'x' or 'y' - sheet propagation direction.

    Returns
    -------
    ndarray
        3D excitation profile.
    """
    d, h, w = shape
    profile = np.zeros(shape)
    z_center = d / 2.0
    for zi in range(d):
        intensity = np.exp(-0.5 * ((zi - z_center) / sheet_thickness)**2)
        profile[zi] = intensity
    return profile


def lightsheet_scan(sample_3d, excitation_sheet, detection_psf,
                    n_planes=None):
    """Simulate light-sheet microscopy.

    Parameters
    ----------
    sample_3d : ndarray
        3D sample volume.
    excitation_sheet : ndarray
        3D excitation profile.
    detection_psf : ndarray
        3D detection PSF.
    n_planes : int or None
        Number of planes; None = all z.

    Returns
    -------
    ndarray
        Light-sheet image stack.
    """
    d, h, w = sample_3d.shape
    if n_planes is None:
        n_planes = d

    result = np.zeros((n_planes, h, w))
    for zi in range(n_planes):
        if zi >= d:
            break
        # Excite: sample * sheet at this z
        excited = sample_3d.copy()
        # Shift sheet to focus at zi
        sheet = np.zeros_like(sample_3d)
        for zj in range(d):
            dz = zj - zi
            sheet[zj] = np.exp(-0.5 * (dz / 2.0)**2)
        excited *= sheet
        # Detect: convolve with detection PSF and take focal plane
        detected = fftconvolve(excited, detection_psf, mode='same')
        result[zi] = detected[zi]
    return result


# ---------------------------------------------------------------------------
# Widefield
# ---------------------------------------------------------------------------

def widefield_image(sample_3d, psf_3d):
    """Simulate widefield microscopy via 3D convolution.

    Parameters
    ----------
    sample_3d : ndarray
        3D sample volume (D, H, W).
    psf_3d : ndarray
        3D PSF.

    Returns
    -------
    ndarray
        Widefield image stack (D, H, W).
    """
    return fftconvolve(sample_3d, psf_3d, mode='same')


# ---------------------------------------------------------------------------
# SIM (Structured Illumination Microscopy)
# ---------------------------------------------------------------------------

def sim_patterns(shape, n_orientations=3, n_phases=3, period=5.0):
    """Generate structured illumination patterns.

    Parameters
    ----------
    shape : tuple
        (H, W) image shape.
    n_orientations : int
        Number of pattern orientations.
    n_phases : int
        Number of phase shifts per orientation.
    period : float
        Pattern period in pixels.

    Returns
    -------
    ndarray
        Shape (n_orientations * n_phases, H, W) illumination patterns.
    """
    h, w = shape
    y, x = np.mgrid[:h, :w]
    patterns = []
    for oi in range(n_orientations):
        angle = np.pi * oi / n_orientations
        for pi in range(n_phases):
            phase = 2 * np.pi * pi / n_phases
            freq = 2 * np.pi / period
            pattern = 0.5 * (1 + np.cos(freq * (x * np.cos(angle) +
                                                  y * np.sin(angle)) +
                                         phase))
            patterns.append(pattern)
    return np.array(patterns)


def sim_reconstruct(raw_images, patterns):
    """Simple SIM reconstruction via Wiener filtering.

    Parameters
    ----------
    raw_images : ndarray
        (N, H, W) raw SIM images.
    patterns : ndarray
        (N, H, W) illumination patterns.

    Returns
    -------
    ndarray
        Reconstructed super-resolved image (H, W).
    """
    # Simple approach: separate frequency components
    n, h, w = raw_images.shape
    result_ft = np.zeros((h, w), dtype=complex)
    weight = np.zeros((h, w))

    for i in range(n):
        ft_raw = np.fft.fft2(raw_images[i])
        ft_pat = np.fft.fft2(patterns[i])
        # Wiener deconvolution of illumination pattern
        wiener_eps = 0.01
        ft_demod = ft_raw * np.conj(ft_pat) / (np.abs(ft_pat)**2 +
                                                 wiener_eps)
        result_ft += ft_demod
        weight += np.abs(ft_pat)**2 / (np.abs(ft_pat)**2 + wiener_eps)

    weight[weight == 0] = 1
    result_ft /= weight
    result = np.real(np.fft.ifft2(result_ft))
    return np.clip(result, 0, None)


# ---------------------------------------------------------------------------
# STED
# ---------------------------------------------------------------------------

def sted_psf(excitation_psf, depletion_power=1.0, saturation=1.0):
    """Generate effective STED PSF.

    Parameters
    ----------
    excitation_psf : ndarray
        Excitation PSF (2D or 3D).
    depletion_power : float
        Depletion laser power (relative to saturation).
    saturation : float
        Saturation intensity.

    Returns
    -------
    ndarray
        Effective STED PSF with sub-diffraction resolution.
    """
    shape = excitation_psf.shape
    ndim = len(shape)

    # Generate donut (vortex) beam profile
    if ndim == 2:
        h, w = shape
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        # Laguerre-Gaussian donut: r^2 * exp(-r^2/w0^2)
        w0 = max(h, w) / 6.0
        donut = (r / w0)**2 * np.exp(-(r / w0)**2)
        donut /= donut.max() + 1e-10
    else:
        d, h, w = shape
        cz, cy, cx = d / 2.0, h / 2.0, w / 2.0
        z, y, x = np.ogrid[:d, :h, :w]
        r_lat = np.sqrt((y - cy)**2 + (x - cx)**2)
        w0 = max(h, w) / 6.0
        donut = (r_lat / w0)**2 * np.exp(-(r_lat / w0)**2)
        donut /= donut.max() + 1e-10

    # Effective PSF = exc * exp(-depletion * donut / saturation)
    effective = excitation_psf * np.exp(
        -depletion_power * donut / saturation)

    # Normalize
    total = effective.sum()
    if total > 0:
        effective /= total
    return effective
