"""Per-pixel feature extraction for pixel classification.

Computes a bank of ~37 per-pixel features (intensity, Gaussian blurs,
edges, LBP, Hessian, Gabor, entropy, structure tensor) for use with
random-forest or CNN pixel classifiers.

All scipy/skimage imports are deferred so this module can be imported
without those packages being available at import time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FeatureConfig:
    """Configuration for which features to compute and their parameters."""
    gaussian_sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0)
    gabor_frequencies: Tuple[float, ...] = (0.1, 0.25, 0.4)
    gabor_orientations: int = 4
    include_intensity: bool = True       # 1 feature
    include_gaussian: bool = True        # 9 features (3 sigmas x {filtered, grad_mag, laplacian})
    include_edges: bool = True           # 4 features (sobel, scharr, roberts, prewitt)
    include_lbp: bool = True             # 1 feature
    include_hessian: bool = True         # 6 features (3 sigmas x {det, trace})
    include_gabor: bool = True           # 12 features (3 freq x 4 orient)
    include_extras: bool = True          # 4 features (entropy, struct orient/coherency/eigenval)


class FeatureExtractor:
    """Extracts per-pixel feature maps from a 2D image.

    Parameters
    ----------
    config : FeatureConfig, optional
        Feature configuration. Uses defaults if not provided.

    Example
    -------
    >>> fe = FeatureExtractor()
    >>> features = fe.extract(image)  # (H, W) -> (H, W, N)
    >>> print(fe.feature_names())
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._names: Optional[List[str]] = None

    def feature_names(self) -> List[str]:
        """Return ordered list of feature names for the current config."""
        if self._names is not None:
            return self._names
        names = []
        cfg = self.config
        if cfg.include_intensity:
            names.append('intensity')
        if cfg.include_gaussian:
            for s in cfg.gaussian_sigmas:
                names.append(f'gaussian_s{s}')
                names.append(f'grad_mag_s{s}')
                names.append(f'laplacian_s{s}')
        if cfg.include_edges:
            names.extend(['sobel', 'scharr', 'roberts', 'prewitt'])
        if cfg.include_lbp:
            names.append('lbp')
        if cfg.include_hessian:
            for s in cfg.gaussian_sigmas:
                names.append(f'hessian_det_s{s}')
                names.append(f'hessian_trace_s{s}')
        if cfg.include_gabor:
            for f in cfg.gabor_frequencies:
                for oi in range(cfg.gabor_orientations):
                    theta = oi * np.pi / cfg.gabor_orientations
                    names.append(f'gabor_f{f}_t{theta:.2f}')
        if cfg.include_extras:
            names.extend(['entropy', 'struct_orient', 'struct_coherency', 'struct_eigenval'])
        self._names = names
        return names

    def n_features(self) -> int:
        """Return the number of features for the current config."""
        return len(self.feature_names())

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract per-pixel features from a 2D image.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            Input image (any dtype, will be normalized to [0, 1] float32).

        Returns
        -------
        ndarray, shape (H, W, N)
            Feature stack with N channels, float32.
        """
        from scipy.ndimage import gaussian_filter, gaussian_laplace, uniform_filter
        from scipy.ndimage import sobel as scipy_sobel

        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got {image.ndim}D")

        img = image.astype(np.float32)
        # Normalize to [0, 1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

        cfg = self.config
        features = []

        if cfg.include_intensity:
            features.append(img)

        if cfg.include_gaussian:
            features.extend(self._gaussian_features(img, cfg.gaussian_sigmas))

        if cfg.include_edges:
            features.extend(self._edge_features(img))

        if cfg.include_lbp:
            features.append(self._lbp_feature(img))

        if cfg.include_hessian:
            features.extend(self._hessian_features(img, cfg.gaussian_sigmas))

        if cfg.include_gabor:
            features.extend(self._gabor_features(img, cfg.gabor_frequencies,
                                                  cfg.gabor_orientations))

        if cfg.include_extras:
            features.extend(self._extra_features(img))

        result = np.stack(features, axis=-1).astype(np.float32)
        # Replace any NaN/inf with 0
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    @staticmethod
    def _gaussian_features(img: np.ndarray, sigmas) -> List[np.ndarray]:
        """Gaussian blur, gradient magnitude, and Laplacian at each sigma."""
        from scipy.ndimage import gaussian_filter, gaussian_laplace

        features = []
        for s in sigmas:
            blurred = gaussian_filter(img, sigma=s)
            features.append(blurred)

            # Gradient magnitude
            gy = gaussian_filter(img, sigma=s, order=[1, 0])
            gx = gaussian_filter(img, sigma=s, order=[0, 1])
            grad_mag = np.sqrt(gx**2 + gy**2)
            features.append(grad_mag)

            # Laplacian of Gaussian
            lap = gaussian_laplace(img, sigma=s)
            features.append(lap)

        return features

    @staticmethod
    def _edge_features(img: np.ndarray) -> List[np.ndarray]:
        """Edge detectors: Sobel, Scharr, Roberts, Prewitt."""
        from scipy.ndimage import sobel as scipy_sobel, prewitt as scipy_prewitt

        # Sobel magnitude
        sx = scipy_sobel(img, axis=0)
        sy = scipy_sobel(img, axis=1)
        sobel_mag = np.sqrt(sx**2 + sy**2)

        # Scharr (3x3 optimized Sobel)
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
        scharr_y = scharr_x.T
        from scipy.ndimage import convolve
        scx = convolve(img, scharr_x)
        scy = convolve(img, scharr_y)
        scharr_mag = np.sqrt(scx**2 + scy**2)

        # Roberts cross
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        from scipy.signal import correlate2d
        rx = correlate2d(img, roberts_x, mode='same', boundary='symm')
        ry = correlate2d(img, roberts_y, mode='same', boundary='symm')
        roberts_mag = np.sqrt(rx**2 + ry**2)

        # Prewitt magnitude
        px = scipy_prewitt(img, axis=0)
        py = scipy_prewitt(img, axis=1)
        prewitt_mag = np.sqrt(px**2 + py**2)

        return [sobel_mag, scharr_mag, roberts_mag, prewitt_mag]

    @staticmethod
    def _lbp_feature(img: np.ndarray) -> np.ndarray:
        """Local Binary Pattern (simplified uniform LBP)."""
        # Simple 8-neighbor LBP without skimage dependency
        h, w = img.shape
        lbp = np.zeros_like(img, dtype=np.float32)
        padded = np.pad(img, 1, mode='reflect')
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
            lbp += (neighbor >= padded[1:1+h, 1:1+w]).astype(np.float32) * (2**bit)
        # Normalize to [0, 1]
        lbp = lbp / 255.0
        return lbp

    @staticmethod
    def _hessian_features(img: np.ndarray, sigmas) -> List[np.ndarray]:
        """Hessian determinant and trace at each sigma."""
        from scipy.ndimage import gaussian_filter

        features = []
        for s in sigmas:
            hxx = gaussian_filter(img, sigma=s, order=[2, 0])
            hyy = gaussian_filter(img, sigma=s, order=[0, 2])
            hxy = gaussian_filter(img, sigma=s, order=[1, 1])

            det = hxx * hyy - hxy**2
            trace = hxx + hyy
            features.append(det)
            features.append(trace)
        return features

    @staticmethod
    def _gabor_features(img: np.ndarray, frequencies, n_orientations) -> List[np.ndarray]:
        """Gabor filter responses at multiple frequencies and orientations."""
        from scipy.ndimage import convolve

        features = []
        for freq in frequencies:
            for oi in range(n_orientations):
                theta = oi * np.pi / n_orientations
                # Build Gabor kernel
                kernel = _make_gabor_kernel(freq, theta)
                response = convolve(img, kernel)
                features.append(np.abs(response))
        return features

    @staticmethod
    def _extra_features(img: np.ndarray) -> List[np.ndarray]:
        """Entropy and structure tensor features."""
        from scipy.ndimage import uniform_filter, gaussian_filter

        # Local entropy (approximated via local variance of log)
        eps = 1e-10
        log_img = np.log(img + eps)
        local_mean = uniform_filter(log_img, size=9)
        local_sq_mean = uniform_filter(log_img**2, size=9)
        entropy = local_sq_mean - local_mean**2
        entropy = np.maximum(entropy, 0)

        # Structure tensor
        sigma = 1.0
        window = 3.0
        gy = gaussian_filter(img, sigma=sigma, order=[1, 0])
        gx = gaussian_filter(img, sigma=sigma, order=[0, 1])

        Jxx = gaussian_filter(gx * gx, sigma=window)
        Jyy = gaussian_filter(gy * gy, sigma=window)
        Jxy = gaussian_filter(gx * gy, sigma=window)

        # Orientation
        orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)

        # Coherency
        trace = Jxx + Jyy
        diff = Jxx - Jyy
        disc = np.sqrt(diff**2 + 4 * Jxy**2)
        coherency = np.zeros_like(trace)
        mask = trace > 1e-10
        coherency[mask] = disc[mask] / trace[mask]

        # Largest eigenvalue
        eigenval = 0.5 * (trace + disc)

        return [entropy, orientation, coherency, eigenval]


def _make_gabor_kernel(frequency: float, theta: float, sigma: float = None,
                       n_stds: float = 3.0) -> np.ndarray:
    """Create a real-valued Gabor filter kernel.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the sinusoidal carrier.
    theta : float
        Orientation in radians.
    sigma : float, optional
        Standard deviation of the Gaussian envelope. Defaults to 1/frequency.
    n_stds : float
        Number of standard deviations for the kernel extent.
    """
    if sigma is None:
        sigma = 1.0 / frequency

    x0 = np.ceil(max(np.abs(n_stds * sigma * np.cos(theta)),
                      np.abs(n_stds * sigma * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma * np.sin(theta)),
                      np.abs(n_stds * sigma * np.cos(theta)), 1))
    y, x = np.mgrid[-y0:y0+1, -x0:x0+1]

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    kernel = np.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2) * \
             np.cos(2 * np.pi * frequency * x_theta)
    return kernel.astype(np.float32)
