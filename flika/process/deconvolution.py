# -*- coding: utf-8 -*-
"""Image deconvolution processes.

Provides Richardson-Lucy and Wiener deconvolution with PSF generation.
"""
from ..logger import logger
logger.debug("Started 'reading process/deconvolution.py'")

import numpy as np
from qtpy import QtWidgets
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess, SliderLabel, SliderLabelOdd, CheckBox, ComboBox
from ..utils.ndim import per_plane

__all__ = ['richardson_lucy', 'wiener_deconvolution', 'generate_psf']

from ..optics.psf import gaussian_psf as _gaussian_psf, airy_psf as _airy_psf


@per_plane(expects_2d=True)
def _richardson_lucy_impl(image, psf, iterations):
    """Richardson-Lucy deconvolution on a single 2D plane."""
    from scipy.signal import fftconvolve
    psf_mirror = psf[::-1, ::-1]
    estimate = np.full_like(image, image.mean(), dtype=np.float64)
    for _ in range(iterations):
        convolved = fftconvolve(estimate, psf, mode='same')
        convolved = np.maximum(convolved, 1e-12)
        ratio = image / convolved
        correction = fftconvolve(ratio, psf_mirror, mode='same')
        estimate *= correction
    return estimate


@per_plane(expects_2d=True)
def _wiener_impl(image, psf, noise_var):
    """Wiener deconvolution on a single 2D plane."""
    img_fft = np.fft.fft2(image)
    psf_padded = np.zeros_like(image)
    py, px = psf.shape
    iy, ix = image.shape
    sy = iy // 2 - py // 2
    sx = ix // 2 - px // 2
    psf_padded[sy:sy+py, sx:sx+px] = psf
    psf_fft = np.fft.fft2(psf_padded)
    psf_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft)**2
    if noise_var <= 0:
        noise_var = 1e-6
    result_fft = (psf_conj / (psf_power + noise_var)) * img_fft
    return np.real(np.fft.ifft2(result_fft))


class Richardson_Lucy(BaseProcess):
    """richardson_lucy(psf_sigma, psf_size, iterations, keepSourceWindow=False)

    Performs Richardson-Lucy iterative deconvolution.

    Parameters:
        psf_sigma (float): Standard deviation of the Gaussian PSF.
        psf_size (int): Size of the PSF kernel (must be odd).
        iterations (int): Number of RL iterations (more = sharper but noisier).
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        psf_sigma = SliderLabel(2)
        psf_sigma.setRange(0.5, 20)
        psf_sigma.setValue(2.0)
        psf_size = SliderLabelOdd()
        psf_size.setRange(3, 51)
        psf_size.setValue(11)
        iterations = SliderLabel(0)
        iterations.setRange(1, 200)
        iterations.setValue(20)
        self.items.append({'name': 'psf_sigma', 'string': 'PSF Sigma', 'object': psf_sigma})
        self.items.append({'name': 'psf_size', 'string': 'PSF Size', 'object': psf_size})
        self.items.append({'name': 'iterations', 'string': 'Iterations', 'object': iterations})
        super().gui()

    def __call__(self, psf_sigma, psf_size=11, iterations=20, keepSourceWindow=False):
        self.start(keepSourceWindow)
        psf = _gaussian_psf(int(psf_size), psf_sigma)
        self.newtif = _richardson_lucy_impl(self.tif.astype(np.float64), psf, int(iterations))
        self.newname = self.oldname + ' - Richardson-Lucy'
        return self.end()

richardson_lucy = Richardson_Lucy()


class Wiener_Deconvolution(BaseProcess):
    """wiener_deconvolution(psf_sigma, psf_size, noise_variance, keepSourceWindow=False)

    Performs Wiener deconvolution (frequency-domain).

    Parameters:
        psf_sigma (float): Standard deviation of the Gaussian PSF.
        psf_size (int): Size of the PSF kernel (must be odd).
        noise_variance (float): Estimated noise variance (regularization).
    Returns:
        newWindow
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        psf_sigma = SliderLabel(2)
        psf_sigma.setRange(0.5, 20)
        psf_sigma.setValue(2.0)
        psf_size = SliderLabelOdd()
        psf_size.setRange(3, 51)
        psf_size.setValue(11)
        noise_var = SliderLabel(4)
        noise_var.setRange(0.0001, 1.0)
        noise_var.setValue(0.01)
        self.items.append({'name': 'psf_sigma', 'string': 'PSF Sigma', 'object': psf_sigma})
        self.items.append({'name': 'psf_size', 'string': 'PSF Size', 'object': psf_size})
        self.items.append({'name': 'noise_variance', 'string': 'Noise Variance', 'object': noise_var})
        super().gui()

    def __call__(self, psf_sigma, psf_size=11, noise_variance=0.01, keepSourceWindow=False):
        self.start(keepSourceWindow)
        psf = _gaussian_psf(int(psf_size), psf_sigma)
        self.newtif = _wiener_impl(self.tif.astype(np.float64), psf, noise_variance)
        self.newname = self.oldname + ' - Wiener'
        return self.end()

wiener_deconvolution = Wiener_Deconvolution()


class Generate_PSF(BaseProcess):
    """generate_psf(psf_type, size, sigma_or_radius, keepSourceWindow=False)

    Generates a Point Spread Function image.

    Parameters:
        psf_type (str): 'Gaussian' or 'Airy Disk'.
        size (int): Size of the PSF (must be odd).
        sigma_or_radius (float): Sigma for Gaussian, radius for Airy disk.
    Returns:
        newWindow containing the PSF
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        psf_type = ComboBox()
        psf_type.addItems(['Gaussian', 'Airy Disk'])
        size = SliderLabelOdd()
        size.setRange(3, 101)
        size.setValue(21)
        param = SliderLabel(2)
        param.setRange(0.5, 20)
        param.setValue(3.0)
        self.items.append({'name': 'psf_type', 'string': 'PSF Type', 'object': psf_type})
        self.items.append({'name': 'size', 'string': 'Size', 'object': size})
        self.items.append({'name': 'sigma_or_radius', 'string': 'Sigma / Radius', 'object': param})
        super().gui()

    def __call__(self, psf_type='Gaussian', size=21, sigma_or_radius=3.0, keepSourceWindow=False):
        if psf_type == 'Gaussian':
            psf = _gaussian_psf(int(size), sigma_or_radius)
        else:
            psf = _airy_psf(int(size), sigma_or_radius)
        return Window(psf, 'PSF ({})'.format(psf_type))

generate_psf = Generate_PSF()


logger.debug("Completed 'reading process/deconvolution.py'")
