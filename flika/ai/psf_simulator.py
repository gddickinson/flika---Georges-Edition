"""PSF simulation for generating training data for particle localization.

Generates synthetic microscopy frames with known particle positions and
corresponding Gaussian density maps. Used to train DeepSTORM-style
localization networks without requiring manually annotated data.

Dependencies: numpy, scipy (both already core deps).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from flika.simulation.noise import CameraConfig, apply_camera


@dataclass
class PSFConfig:
    """Configuration for PSF simulation."""
    image_size: int = 128
    n_particles: int = 50
    psf_sigma: float = 1.5
    intensity_range: Tuple[float, float] = (500.0, 2000.0)
    background_mean: float = 100.0
    background_std: float = 10.0
    noise_type: str = 'poisson'     # 'poisson', 'gaussian', 'mixed'
    read_noise_std: float = 5.0
    n_frames: int = 1


class PSFSimulator:
    """Generates synthetic microscopy images with known particle positions.

    Parameters
    ----------
    config : PSFConfig, optional
        Simulation parameters. Uses defaults if not provided.

    Example
    -------
    >>> sim = PSFSimulator(PSFConfig(n_particles=20, psf_sigma=1.5))
    >>> image, positions = sim.generate_frame()
    >>> density = sim.positions_to_density_map(positions, image.shape)
    """

    def __init__(self, config: Optional[PSFConfig] = None):
        self.config = config or PSFConfig()

    def generate_frame(self, rng: Optional[np.random.Generator] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single simulated microscopy frame.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        image : ndarray, shape (H, W), float32
            Noisy simulated microscopy image.
        positions : ndarray, shape (N, 2), float64
            Ground-truth particle positions as (y, x) coordinates.
        """
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.config
        h = w = cfg.image_size

        # Random particle positions (sub-pixel)
        positions = np.column_stack([
            rng.uniform(0, h, cfg.n_particles),
            rng.uniform(0, w, cfg.n_particles),
        ])

        # Random intensities
        intensities = rng.uniform(cfg.intensity_range[0], cfg.intensity_range[1],
                                   cfg.n_particles)

        # Render clean image
        clean = np.full((h, w), cfg.background_mean, dtype=np.float64)
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        for (py, px), intensity in zip(positions, intensities):
            gauss = np.exp(-((y_grid - py)**2 + (x_grid - px)**2) /
                           (2 * cfg.psf_sigma**2))
            clean += intensity * gauss

        # Add noise
        image = self._add_noise(clean, rng)
        return image.astype(np.float32), positions

    def generate_stack(self, rng: Optional[np.random.Generator] = None
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a stack of simulated frames.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        stack : ndarray, shape (T, H, W), float32
            Stack of noisy frames.
        all_positions : list of ndarray
            Per-frame particle positions, each shape (N, 2).
        """
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.config
        frames = []
        all_positions = []

        for _ in range(cfg.n_frames):
            image, positions = self.generate_frame(rng)
            frames.append(image)
            all_positions.append(positions)

        return np.stack(frames, axis=0), all_positions

    def positions_to_density_map(self, positions: np.ndarray,
                                  shape: Tuple[int, int],
                                  sigma: Optional[float] = None) -> np.ndarray:
        """Render particle positions as a Gaussian density map.

        Parameters
        ----------
        positions : ndarray, shape (N, 2)
            Particle positions as (y, x).
        shape : tuple of int
            (H, W) of the output map.
        sigma : float, optional
            Gaussian sigma for rendering. Defaults to config psf_sigma.

        Returns
        -------
        density : ndarray, shape (H, W), float32
            Density map with values in [0, 1].
        """
        if sigma is None:
            sigma = self.config.psf_sigma

        h, w = shape
        density = np.zeros((h, w), dtype=np.float64)
        y_grid, x_grid = np.mgrid[0:h, 0:w]

        for py, px in positions:
            gauss = np.exp(-((y_grid - py)**2 + (x_grid - px)**2) /
                           (2 * sigma**2))
            density += gauss

        # Normalize to [0, 1]
        if density.max() > 0:
            density /= density.max()

        return density.astype(np.float32)

    @staticmethod
    def extract_coordinates(density_map: np.ndarray, threshold: float = 0.2,
                            min_distance: int = 3) -> np.ndarray:
        """Extract sub-pixel particle coordinates from a density map.

        Uses peak detection followed by centroid refinement.

        Parameters
        ----------
        density_map : ndarray, shape (H, W)
            Predicted density map.
        threshold : float
            Minimum peak value to consider.
        min_distance : int
            Minimum distance between detected peaks.

        Returns
        -------
        coords : ndarray, shape (N, 2)
            Detected positions as (y, x), sub-pixel precision.
        """
        from scipy.ndimage import maximum_filter, label, center_of_mass

        if density_map.max() < threshold:
            return np.empty((0, 2), dtype=np.float64)

        # Find local maxima
        max_filtered = maximum_filter(density_map, size=2 * min_distance + 1)
        peaks = (density_map == max_filtered) & (density_map >= threshold)

        if not peaks.any():
            return np.empty((0, 2), dtype=np.float64)

        # Label connected components and find centroids for sub-pixel accuracy
        labeled, n_labels = label(peaks)
        if n_labels == 0:
            return np.empty((0, 2), dtype=np.float64)

        # Use intensity-weighted centroid for sub-pixel refinement
        coords = center_of_mass(density_map, labeled, range(1, n_labels + 1))
        return np.array(coords, dtype=np.float64)

    def _add_noise(self, clean: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        """Add realistic microscopy noise using the unified camera model.

        Delegates to ``flika.simulation.noise.apply_camera`` with a
        ``CameraConfig`` derived from this instance's ``PSFConfig``.  The
        camera model is configured so that its output closely matches the
        original simple noise model:

        * QE = 1 (clean image already represents detected photons)
        * gain = 1 (keep ADU == electrons, no rescaling)
        * baseline = 0 (caller adds background before this step)
        * bit_depth = 32 (avoid clipping float-range data)
        * read_noise mapped from ``PSFConfig.read_noise_std``

        For the ``'gaussian'`` noise type (no Poisson), shot noise is
        suppressed by passing photon values through ``apply_camera`` with
        a near-zero signal and adding Gaussian noise via the camera's
        read-noise path.
        """
        cfg = self.config

        if cfg.noise_type in ('poisson', 'mixed'):
            # Build a CameraConfig that reproduces the original pipeline:
            #   Poisson(clean) + Gaussian(0, read_noise_std)
            cam = CameraConfig(
                type='sCMOS',
                quantum_efficiency=1.0,   # clean is already in photons
                read_noise=cfg.read_noise_std if cfg.noise_type == 'mixed' else 0.0,
                dark_current=0.0,
                gain=1.0,
                baseline=0,
                bit_depth=32,             # avoid uint16 clipping
                exposure_time=0.0,
            )
            photons = np.maximum(clean, 0)
            noisy = apply_camera(photons, cam).astype(np.float64)
        else:
            # Pure Gaussian: no shot noise, just additive noise
            noisy = clean.copy()
            noisy += rng.normal(0, cfg.read_noise_std, clean.shape)
            noisy += rng.normal(0, cfg.background_std, clean.shape)

        return noisy
