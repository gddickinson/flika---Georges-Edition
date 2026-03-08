# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/morphometry.py'")
import numpy as np
from scipy import ndimage
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox, SliderLabel

__all__ = ['morphometry_analysis']


# ---------------------------------------------------------------------------
# Pure analysis functions (no Qt dependencies)
# ---------------------------------------------------------------------------

def compute_region_properties(label_image, intensity_image=None):
    """Compute morphometric properties for each labelled region.

    Parameters
    ----------
    label_image : 2-D ndarray of int
        Labelled image where each region has a unique integer > 0.
    intensity_image : 2-D ndarray, optional
        Intensity image for intensity-based measurements.

    Returns
    -------
    list of dict
        One dict per region with keys:
        label, area, perimeter, centroid_y, centroid_x, bbox,
        major_axis, minor_axis, eccentricity, solidity, circularity,
        equivalent_diameter, extent, aspect_ratio,
        mean_intensity, max_intensity, min_intensity, std_intensity
        (intensity keys only if intensity_image is provided).
    """
    label_image = np.asarray(label_image, dtype=int)
    labels = np.unique(label_image)
    labels = labels[labels > 0]

    if intensity_image is not None:
        intensity_image = np.asarray(intensity_image, dtype=np.float64)

    results = []
    for lab in labels:
        mask = label_image == lab
        props = _compute_single_region(mask, lab, intensity_image)
        results.append(props)

    return results


def _compute_single_region(mask, label, intensity_image=None):
    """Compute properties for a single binary region."""
    # Area
    area = int(np.sum(mask))
    if area == 0:
        return {'label': int(label), 'area': 0}

    # Centroid
    coords = np.argwhere(mask)
    centroid_y = float(np.mean(coords[:, 0]))
    centroid_x = float(np.mean(coords[:, 1]))

    # Bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox = (int(y_min), int(x_min), int(y_max), int(x_max))
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)

    # Perimeter (count boundary pixels)
    eroded = ndimage.binary_erosion(mask)
    perimeter = float(np.sum(mask & ~eroded))
    if perimeter == 0:
        perimeter = float(area)  # single-pixel region

    # Equivalent diameter
    equivalent_diameter = float(np.sqrt(4 * area / np.pi))

    # Circularity = 4*pi*area / perimeter^2
    circularity = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Extent = area / bbox_area
    extent = float(area / bbox_area) if bbox_area > 0 else 0.0

    # Second-order moments for ellipse fitting
    centered = coords.astype(np.float64) - np.array([centroid_y, centroid_x])
    if len(centered) >= 2:
        cov = np.cov(centered.T)
        if cov.shape == (2, 2):
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # descending
            major_axis = float(4 * np.sqrt(max(eigenvalues[0], 0)))
            minor_axis = float(4 * np.sqrt(max(eigenvalues[1], 0)))
        else:
            major_axis = equivalent_diameter
            minor_axis = equivalent_diameter
    else:
        major_axis = 1.0
        minor_axis = 1.0

    # Eccentricity
    if major_axis > 0:
        ecc_ratio = minor_axis / major_axis
        eccentricity = float(np.sqrt(1 - ecc_ratio ** 2)) if ecc_ratio <= 1.0 else 0.0
    else:
        eccentricity = 0.0

    # Aspect ratio
    aspect_ratio = float(major_axis / minor_axis) if minor_axis > 0 else 1.0

    # Solidity = area / convex_hull_area
    # Approximate with bounding box for speed
    solidity = float(area / bbox_area) if bbox_area > 0 else 1.0

    props = {
        'label': int(label),
        'area': area,
        'perimeter': perimeter,
        'centroid_y': centroid_y,
        'centroid_x': centroid_x,
        'bbox': bbox,
        'major_axis': major_axis,
        'minor_axis': minor_axis,
        'eccentricity': eccentricity,
        'solidity': solidity,
        'circularity': circularity,
        'equivalent_diameter': equivalent_diameter,
        'extent': extent,
        'aspect_ratio': aspect_ratio,
    }

    # Intensity measurements
    if intensity_image is not None:
        vals = intensity_image[mask]
        props['mean_intensity'] = float(np.mean(vals))
        props['max_intensity'] = float(np.max(vals))
        props['min_intensity'] = float(np.min(vals))
        props['std_intensity'] = float(np.std(vals))
        props['total_intensity'] = float(np.sum(vals))

    return props


def compute_texture_features(image, mask=None, n_bins=32):
    """Compute texture features from a grey-level co-occurrence matrix (GLCM).

    Simplified Haralick-style texture features computed from the image
    histogram and spatial statistics.

    Parameters
    ----------
    image : 2-D ndarray
        Intensity image.
    mask : 2-D bool array, optional
        Region of interest.
    n_bins : int
        Number of grey levels for quantization.

    Returns
    -------
    dict
        Keys: contrast, dissimilarity, homogeneity, energy, correlation,
        entropy.
    """
    image = np.asarray(image, dtype=np.float64)
    if mask is not None:
        pixels = image[mask]
    else:
        pixels = image.ravel()

    if pixels.size < 4:
        return {
            'contrast': np.nan, 'dissimilarity': np.nan,
            'homogeneity': np.nan, 'energy': np.nan,
            'correlation': np.nan, 'entropy': np.nan,
        }

    # Quantize to n_bins levels
    pmin, pmax = np.min(pixels), np.max(pixels)
    if pmax - pmin < 1e-12:
        return {
            'contrast': 0.0, 'dissimilarity': 0.0,
            'homogeneity': 1.0, 'energy': 1.0,
            'correlation': np.nan, 'entropy': 0.0,
        }

    quantized = ((pixels - pmin) / (pmax - pmin) * (n_bins - 1)).astype(int)
    quantized = np.clip(quantized, 0, n_bins - 1)

    # Build simplified GLCM (horizontal adjacency)
    if mask is not None:
        # Use full image for spatial relationships
        q_img = ((image - pmin) / (pmax - pmin) * (n_bins - 1)).astype(int)
        q_img = np.clip(q_img, 0, n_bins - 1)
        glcm = np.zeros((n_bins, n_bins), dtype=np.float64)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            if c + 1 < image.shape[1] and mask[r, c + 1]:
                glcm[q_img[r, c], q_img[r, c + 1]] += 1
    else:
        q_img = ((image - pmin) / (pmax - pmin) * (n_bins - 1)).astype(int)
        q_img = np.clip(q_img, 0, n_bins - 1)
        glcm = np.zeros((n_bins, n_bins), dtype=np.float64)
        for r in range(image.shape[0]):
            for c in range(image.shape[1] - 1):
                glcm[q_img[r, c], q_img[r, c + 1]] += 1

    # Normalize
    total = glcm.sum()
    if total > 0:
        glcm /= total

    # Compute Haralick features
    i_idx, j_idx = np.meshgrid(np.arange(n_bins), np.arange(n_bins), indexing='ij')
    diff = np.abs(i_idx - j_idx).astype(np.float64)

    contrast = float(np.sum(diff ** 2 * glcm))
    dissimilarity = float(np.sum(diff * glcm))
    homogeneity = float(np.sum(glcm / (1 + diff ** 2)))

    energy = float(np.sum(glcm ** 2))

    # Entropy
    nonzero = glcm[glcm > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # Correlation
    mu_i = float(np.sum(i_idx * glcm))
    mu_j = float(np.sum(j_idx * glcm))
    std_i = float(np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm)))
    std_j = float(np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm)))
    if std_i > 0 and std_j > 0:
        correlation = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (std_i * std_j))
    else:
        correlation = np.nan

    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'entropy': entropy,
    }


def compute_shape_descriptors(mask):
    """Compute Hu moment invariants for a binary shape.

    Parameters
    ----------
    mask : 2-D bool array

    Returns
    -------
    dict
        Keys: hu_1 through hu_7 (rotation/scale/translation invariant).
    """
    mask = np.asarray(mask, dtype=np.float64)
    if mask.sum() == 0:
        return {f'hu_{i+1}': np.nan for i in range(7)}

    # Compute raw moments
    y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
    m00 = np.sum(mask)
    m10 = np.sum(x * mask)
    m01 = np.sum(y * mask)
    cx = m10 / m00
    cy = m01 / m00

    # Central moments
    xc = x - cx
    yc = y - cy
    mu20 = np.sum(xc ** 2 * mask) / m00
    mu02 = np.sum(yc ** 2 * mask) / m00
    mu11 = np.sum(xc * yc * mask) / m00
    mu30 = np.sum(xc ** 3 * mask) / m00
    mu03 = np.sum(yc ** 3 * mask) / m00
    mu21 = np.sum(xc ** 2 * yc * mask) / m00
    mu12 = np.sum(xc * yc ** 2 * mask) / m00

    # Scale-normalized central moments
    def eta(p, q, muval):
        gamma = (p + q) / 2.0 + 1
        return muval / (m00 ** gamma)

    n20 = eta(2, 0, mu20 * m00)
    n02 = eta(0, 2, mu02 * m00)
    n11 = eta(1, 1, mu11 * m00)
    n30 = eta(3, 0, mu30 * m00)
    n03 = eta(0, 3, mu03 * m00)
    n21 = eta(2, 1, mu21 * m00)
    n12 = eta(1, 2, mu12 * m00)

    # Hu moments
    hu1 = n20 + n02
    hu2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    hu3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    hu4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    hu5 = ((n30 - 3 * n12) * (n30 + n12) *
           ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
           (3 * n21 - n03) * (n21 + n03) *
           (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    hu6 = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) +
           4 * n11 * (n30 + n12) * (n21 + n03))
    hu7 = ((3 * n21 - n03) * (n30 + n12) *
           ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
           (n30 - 3 * n12) * (n21 + n03) *
           (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))

    return {
        'hu_1': float(hu1), 'hu_2': float(hu2), 'hu_3': float(hu3),
        'hu_4': float(hu4), 'hu_5': float(hu5), 'hu_6': float(hu6),
        'hu_7': float(hu7),
    }


# ---------------------------------------------------------------------------
# BaseProcess subclass
# ---------------------------------------------------------------------------

class MorphometryAnalysis(BaseProcess):
    """morphometry_analysis(window, threshold=0.5, include_texture=False, keepSourceWindow=True)

    Morphometric measurements for labelled or binary images.

    Computes region properties (area, perimeter, circularity, eccentricity,
    etc.), optional texture features (Haralick), and shape descriptors
    (Hu moments) for each object in a labelled image.

    If the input is not a label image, it is auto-thresholded and
    connected components are labelled.

    Parameters:
        window (Window): Image window (labelled or intensity).
        threshold (float): Auto-threshold level for intensity images.
        include_texture (bool): Compute Haralick texture features.
    Returns:
        None (results stored in window.metadata['morphometry']).
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window = WindowSelector()
        threshold = SliderLabel(decimals=2)
        threshold.setRange(0.0, 65535.0)
        threshold.setValue(0.5)
        include_texture = CheckBox()
        include_texture.setChecked(False)
        self.items.append({'name': 'window', 'string': 'Window', 'object': window})
        self.items.append({'name': 'threshold', 'string': 'Threshold', 'object': threshold})
        self.items.append({'name': 'include_texture', 'string': 'Include texture', 'object': include_texture})
        super().gui()

    def __call__(self, window, threshold=0.5, include_texture=False,
                 keepSourceWindow=True):
        if window is None:
            raise MissingWindowError("A window must be selected.")

        img = window.image
        if img.ndim == 3:
            img = img[window.currentIndex]

        img = np.asarray(img, dtype=np.float64)

        # Check if label image (integer, small range)
        is_label = (img.dtype in (np.int32, np.int64, int) or
                    (np.all(img == img.astype(int)) and
                     img.max() < 10000 and img.min() >= 0))

        if is_label and img.max() > 0:
            label_img = img.astype(int)
            intensity_img = img
        else:
            # Auto-threshold
            binary = img > threshold
            label_img, n_labels = ndimage.label(binary)
            intensity_img = img

        if label_img.max() == 0:
            g.alert("No objects found. Try adjusting the threshold.")
            return

        # Compute region properties
        regions = compute_region_properties(label_img, intensity_img)

        # Add texture features if requested
        if include_texture:
            for region in regions:
                mask = label_img == region['label']
                texture = compute_texture_features(intensity_img, mask=mask)
                region.update(texture)
                shape = compute_shape_descriptors(mask)
                region.update(shape)

        # Print results
        name = window.name
        n_regions = len(regions)
        print("=" * 60)
        print(f"Morphometry Analysis: {name} ({n_regions} regions)")
        print("=" * 60)
        for r in regions[:10]:  # Print first 10
            print(f"  Region {r['label']}: area={r['area']}, "
                  f"circ={r.get('circularity', 0):.3f}, "
                  f"ecc={r.get('eccentricity', 0):.3f}, "
                  f"aspect={r.get('aspect_ratio', 0):.2f}")
        if n_regions > 10:
            print(f"  ... and {n_regions - 10} more regions")
        print("=" * 60)

        # Summary stats
        areas = [r['area'] for r in regions]
        print(f"  Total regions      = {n_regions}")
        print(f"  Mean area          = {np.mean(areas):.1f}")
        print(f"  Median area        = {np.median(areas):.1f}")
        print(f"  Total area         = {np.sum(areas)}")
        print("=" * 60)

        # Store results
        window.metadata['morphometry'] = {
            'regions': regions,
            'n_regions': n_regions,
            'include_texture': include_texture,
        }

        g.status_msg(f"Morphometry: {n_regions} regions, "
                     f"mean area={np.mean(areas):.1f}")

        return None

    def get_init_settings_dict(self):
        return {'threshold': 0.5, 'include_texture': False}


morphometry_analysis = MorphometryAnalysis()

logger.debug("Completed 'reading process/morphometry.py'")
