# -*- coding: utf-8 -*-
"""Camera and noise models for microscopy simulation."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Camera sensor configuration.

    Parameters
    ----------
    type : str
        Sensor type: 'CCD', 'EMCCD', or 'sCMOS'.
    pixel_size : float
        Physical pixel size in microns.
    quantum_efficiency : float
        QE (0-1).
    read_noise : float
        RMS read noise in electrons.
    dark_current : float
        Dark current in electrons/pixel/second.
    gain : float
        System gain in ADU/electron.
    em_gain : float
        EM gain (EMCCD only).
    baseline : int
        ADU offset (camera bias).
    bit_depth : int
        Sensor bit depth.
    exposure_time : float
        Exposure time in seconds.
    """
    type: str = 'sCMOS'
    pixel_size: float = 6.5
    quantum_efficiency: float = 0.82
    read_noise: float = 1.5
    dark_current: float = 0.06
    gain: float = 1.0
    em_gain: float = 300
    baseline: int = 100
    bit_depth: int = 16
    exposure_time: float = 0.03


def apply_shot_noise(photons):
    """Apply Poisson shot noise to a photon count image.

    Parameters
    ----------
    photons : ndarray
        Expected photon counts (non-negative).

    Returns
    -------
    ndarray
        Noisy photon counts.
    """
    photons = np.clip(photons, 0, None)
    # For large values, use normal approximation to avoid overflow
    large = photons > 1e7
    result = np.empty_like(photons, dtype=float)
    if np.any(~large):
        result[~large] = np.random.poisson(photons[~large].astype(np.float64))
    if np.any(large):
        result[large] = np.random.normal(photons[large],
                                         np.sqrt(photons[large]))
    return np.clip(result, 0, None)


def apply_camera(photons, config=None):
    """Apply full camera model to convert photon counts to ADU.

    Pipeline: QE -> Poisson -> [EM gain + excess noise] -> read noise ->
              dark current -> ADU conversion -> clip to bit depth.

    Parameters
    ----------
    photons : ndarray
        Incident photon counts.
    config : CameraConfig or None
        Camera configuration; defaults to sCMOS.

    Returns
    -------
    ndarray
        Image in ADU (uint16 or appropriate type).
    """
    if config is None:
        config = CameraConfig()

    # Quantum efficiency: photons -> photoelectrons
    electrons = photons * config.quantum_efficiency

    # Shot noise (Poisson on photoelectrons)
    electrons = apply_shot_noise(electrons)

    # EM gain with excess noise factor (EMCCD)
    if config.type == 'EMCCD' and config.em_gain > 1:
        # Gamma distribution models stochastic EM multiplication
        # Mean = em_gain * electrons, excess noise factor F^2 = 2
        mask = electrons > 0
        amplified = np.zeros_like(electrons)
        if np.any(mask):
            amplified[mask] = np.random.gamma(
                electrons[mask], config.em_gain)
        electrons = amplified

    # Dark current
    dark = config.dark_current * config.exposure_time
    if dark > 0:
        electrons += np.random.poisson(dark, size=electrons.shape)

    # Read noise (Gaussian)
    if config.read_noise > 0:
        electrons += np.random.normal(0, config.read_noise,
                                      size=electrons.shape)

    # Convert to ADU
    adu = electrons * config.gain + config.baseline

    # Clip to bit depth
    max_val = 2**config.bit_depth - 1
    adu = np.clip(adu, 0, max_val)

    return adu.astype(np.uint16 if config.bit_depth <= 16 else np.uint32)


def apply_background(image, level, mode='uniform'):
    """Add background fluorescence to an image.

    Parameters
    ----------
    image : ndarray
        Input photon image.
    level : float
        Mean background photon count.
    mode : str
        'uniform': constant background.
        'gradient': linear gradient along y-axis.
        'autofluorescence': spatially varying (Perlin-like).

    Returns
    -------
    ndarray
        Image with added background.
    """
    result = image.astype(float)
    shape = image.shape[-2:]  # last two dimensions (H, W)

    if mode == 'uniform':
        result += level
    elif mode == 'gradient':
        h = shape[0]
        grad = np.linspace(0.5 * level, 1.5 * level, h)
        # Broadcast to image shape
        if image.ndim == 2:
            result += grad[:, np.newaxis]
        elif image.ndim == 3:
            result += grad[np.newaxis, :, np.newaxis]
        else:
            result += level
    elif mode == 'autofluorescence':
        # Smooth spatially varying background
        h, w = shape
        # Low-frequency noise
        small_h, small_w = max(4, h // 16), max(4, w // 16)
        noise = np.random.rand(small_h, small_w)
        from scipy.ndimage import zoom
        scale_y, scale_x = h / small_h, w / small_w
        bg = zoom(noise, (scale_y, scale_x), order=3)
        bg = bg[:h, :w]  # trim to exact size
        bg = bg / bg.mean() * level
        if image.ndim == 2:
            result += bg
        elif image.ndim == 3:
            result += bg[np.newaxis, :, :]
        else:
            result += level
    else:
        result += level

    return result
