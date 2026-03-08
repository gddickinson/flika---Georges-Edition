# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/spectral.py'")
import numpy as np
from scipy.optimize import nnls
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox, SliderLabel

__all__ = ['spectral_unmixing']


# ---------------------------------------------------------------------------
# Pure analysis functions (no Qt dependencies)
# ---------------------------------------------------------------------------

def linear_unmix(data, spectra, method='nnls'):
    """Linear spectral unmixing.

    Solves  data = spectra @ abundances  for each pixel.

    Parameters
    ----------
    data : ndarray, shape (..., n_channels)
        Multi-channel image data.  Last dimension is spectral channels.
    spectra : ndarray, shape (n_components, n_channels)
        Reference spectra (endmembers), one row per component.
    method : str
        'nnls' (non-negative least squares) or 'lstsq' (unconstrained).

    Returns
    -------
    ndarray, shape (..., n_components)
        Abundance maps.  Same spatial dimensions as input.
    """
    data = np.asarray(data, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)

    n_components, n_channels = spectra.shape
    orig_shape = data.shape
    spatial_shape = orig_shape[:-1]

    # Flatten spatial dimensions
    pixels = data.reshape(-1, n_channels)
    n_pixels = pixels.shape[0]
    abundances = np.zeros((n_pixels, n_components), dtype=np.float64)

    if method == 'nnls':
        A = spectra.T  # (n_channels, n_components)
        for i in range(n_pixels):
            abundances[i], _ = nnls(A, pixels[i])
    else:  # lstsq
        A = spectra.T
        result = np.linalg.lstsq(A, pixels.T, rcond=None)
        abundances = result[0].T

    return abundances.reshape(spatial_shape + (n_components,))


def compute_residual(data, spectra, abundances):
    """Compute unmixing residual image.

    Parameters
    ----------
    data : ndarray, shape (..., n_channels)
    spectra : ndarray, shape (n_components, n_channels)
    abundances : ndarray, shape (..., n_components)

    Returns
    -------
    ndarray, shape (..., n_channels)
        Residual = data - reconstructed.
    float
        Root mean square error.
    """
    data = np.asarray(data, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)
    abundances = np.asarray(abundances, dtype=np.float64)

    # Reconstruct: for each pixel, sum(abundance_i * spectrum_i)
    reconstructed = abundances @ spectra
    residual = data - reconstructed
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return residual, rmse


def estimate_endmembers_pca(data, n_components=3):
    """Estimate spectral endmembers using PCA + vertex extraction.

    Uses PCA to reduce dimensionality, then finds the most extreme
    pixels (vertices of the convex hull in PCA space) as endmember
    candidates.

    Parameters
    ----------
    data : ndarray, shape (..., n_channels)
        Multi-channel image data.
    n_components : int
        Number of endmembers to extract.

    Returns
    -------
    ndarray, shape (n_components, n_channels)
        Estimated endmember spectra.
    """
    data = np.asarray(data, dtype=np.float64)
    n_channels = data.shape[-1]
    spatial_shape = data.shape[:-1]
    pixels = data.reshape(-1, n_channels)

    # Remove zero pixels
    nonzero = np.sum(pixels, axis=1) > 0
    active = pixels[nonzero]

    if len(active) < n_components:
        logger.warning("Not enough non-zero pixels for %d endmembers",
                        n_components)
        return pixels[:n_components] if len(pixels) >= n_components else None

    # PCA
    mean = np.mean(active, axis=0)
    centered = active - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:n_components].T

    # Find vertices: pixels furthest from centroid in PCA space
    centroid = np.mean(projected, axis=0)
    distances = np.sqrt(np.sum((projected - centroid) ** 2, axis=1))

    endmember_indices = []
    remaining = np.arange(len(active))

    for _ in range(n_components):
        if len(remaining) == 0:
            break
        # Find pixel furthest from already selected endmembers
        if not endmember_indices:
            idx = np.argmax(distances[remaining])
        else:
            # Find pixel maximally distant from all selected
            selected_pts = projected[endmember_indices]
            min_dists = np.full(len(remaining), np.inf)
            for s in selected_pts:
                d = np.sqrt(np.sum((projected[remaining] - s) ** 2, axis=1))
                min_dists = np.minimum(min_dists, d)
            idx = np.argmax(min_dists)

        endmember_indices.append(remaining[idx])
        remaining = np.delete(remaining, idx)

    endmembers = active[endmember_indices]
    return endmembers


def normalize_spectra(spectra):
    """Normalize each spectrum to unit sum.

    Parameters
    ----------
    spectra : ndarray, shape (n_components, n_channels)

    Returns
    -------
    ndarray, same shape
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    sums = np.sum(spectra, axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return spectra / sums


# ---------------------------------------------------------------------------
# BaseProcess subclass
# ---------------------------------------------------------------------------

class SpectralUnmixing(BaseProcess):
    """spectral_unmixing(window, n_components=3, method='nnls', auto_endmembers=True, keepSourceWindow=True)

    Linear spectral unmixing of multi-channel images.

    Decomposes a multi-channel image into component abundance maps using
    either automatic endmember estimation (PCA + vertex extraction) or
    user-specified reference spectra.

    Parameters:
        window (Window): Multi-channel image (last axis = channels).
        n_components (int): Number of spectral components to extract.
        method (str): 'nnls' (non-negative) or 'lstsq' (unconstrained).
        auto_endmembers (bool): Automatically estimate endmembers via PCA.
    Returns:
        Window -- Abundance map stack (one frame per component).
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window = WindowSelector()
        n_components = SliderLabel(decimals=0)
        n_components.setRange(2, 20)
        n_components.setValue(3)
        method = self._method_combo = __import__('qtpy.QtWidgets', fromlist=['QComboBox']).QComboBox()
        method.addItem('nnls')
        method.addItem('lstsq')
        auto_endmembers = CheckBox()
        auto_endmembers.setChecked(True)
        self.items.append({'name': 'window', 'string': 'Window', 'object': window})
        self.items.append({'name': 'n_components', 'string': 'Components', 'object': n_components})
        self.items.append({'name': 'method', 'string': 'Method', 'object': method})
        self.items.append({'name': 'auto_endmembers', 'string': 'Auto endmembers', 'object': auto_endmembers})
        super().gui()

    def __call__(self, window, n_components=3, method='nnls',
                 auto_endmembers=True, keepSourceWindow=True):
        if window is None:
            raise MissingWindowError("A window must be selected.")

        img = window.image
        if img.ndim < 3:
            g.alert("Spectral unmixing requires a multi-channel image "
                    "(3-D with last axis = channels, or 4-D T,Z/C,Y,X).")
            return

        # Interpret the stack: if 3-D, treat as (Y, X, C) for multi-channel
        # or (T/C, Y, X) for time-series.  Use the smallest dimension as channels.
        if img.ndim == 3:
            # Convention: if first dim is small (<= 20), treat as (C, Y, X)
            if img.shape[0] <= 20:
                # (C, Y, X) -> (Y, X, C)
                data = np.moveaxis(img, 0, -1).astype(np.float64)
            else:
                g.alert("Cannot determine channel axis. For spectral unmixing, "
                        "provide a multi-channel image with <= 20 channels.")
                return
        elif img.ndim == 4:
            # (T, C, Y, X) -> use first time point
            data = np.moveaxis(img[0], 0, -1).astype(np.float64)
        else:
            g.alert(f"Unsupported image dimensions: {img.ndim}")
            return

        n_channels = data.shape[-1]
        n_comp = min(int(n_components), n_channels)

        # Estimate or use provided endmembers
        if auto_endmembers:
            spectra = estimate_endmembers_pca(data, n_comp)
            if spectra is None:
                g.alert("Could not estimate endmembers from the image.")
                return
            spectra = normalize_spectra(spectra)
        else:
            g.alert("Manual endmember specification not yet implemented. "
                    "Use auto endmembers.")
            return

        # Unmix
        abundances = linear_unmix(data, spectra, method=method)
        residual, rmse = compute_residual(data, spectra, abundances)

        # Build results
        name = window.name
        results = {
            'n_components': n_comp,
            'n_channels': n_channels,
            'method': method,
            'endmembers': spectra,
            'rmse': rmse,
        }

        print("=" * 55)
        print(f"Spectral Unmixing: {name}")
        print("=" * 55)
        print(f"  Channels           = {n_channels}")
        print(f"  Components         = {n_comp}")
        print(f"  Method             = {method}")
        print(f"  RMSE               = {rmse:.6f}")
        for i in range(n_comp):
            ab = abundances[..., i]
            print(f"  Component {i}: mean={np.mean(ab):.4f}, "
                  f"max={np.max(ab):.4f}")
        print("=" * 55)

        window.metadata['spectral_unmixing'] = results

        g.status_msg(f"Spectral unmixing: {n_comp} components, RMSE={rmse:.4f}")

        # Create abundance map as a stack (one frame per component)
        abundance_stack = np.moveaxis(abundances, -1, 0).astype(np.float32)
        from ..window import Window
        ab_window = Window(abundance_stack,
                           name=f'Unmixed: {name} ({n_comp} components)')

        # Plot endmember spectra
        self._spectra_widget = QtWidgets.QWidget()
        self._spectra_widget.setWindowTitle(f'Endmember Spectra: {name}')
        self._spectra_widget.resize(500, 350)
        layout = QtWidgets.QVBoxLayout(self._spectra_widget)
        pw = pg.PlotWidget()
        pw.setLabel('bottom', 'Channel')
        pw.setLabel('left', 'Intensity')
        pw.setTitle('Estimated Endmember Spectra')
        layout.addWidget(pw)

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        channels = np.arange(n_channels)
        for i in range(n_comp):
            color = colors[i % len(colors)]
            pw.plot(channels, spectra[i], pen=pg.mkPen(color, width=2),
                    name=f'Component {i}')
        pw.addLegend()
        self._spectra_widget.show()

        return ab_window

    def get_init_settings_dict(self):
        return {
            'n_components': 3,
            'method': 'nnls',
            'auto_endmembers': True,
        }


spectral_unmixing = SpectralUnmixing()

logger.debug("Completed 'reading process/spectral.py'")
