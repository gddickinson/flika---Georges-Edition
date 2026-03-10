# -*- coding: utf-8 -*-
"""Dimension-aware ROI statistics utilities."""
import numpy as np


def compute_roi_stats(roi, window):
    """Return a dict of statistics for the ROI on the current frame.

    Works for 2D, 3D, and 4D windows.
    """
    mask = roi.getMask()
    s1, s2 = mask
    if np.size(s1) == 0:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan,
                'max': np.nan, 'area': 0, 'integrated': np.nan}

    if window.nDims >= 3 and not window.metadata.get('is_rgb', False):
        frame = window.image[window.currentIndex]
    else:
        frame = window.image if window.nDims == 2 else window.image
        if frame.ndim == 3 and window.metadata.get('is_rgb', False):
            frame = np.mean(frame, axis=2)

    pixels = frame[s1, s2]
    stats = {
        'mean': float(np.mean(pixels)),
        'std': float(np.std(pixels)),
        'min': float(np.min(pixels)),
        'max': float(np.max(pixels)),
        'area': int(np.size(pixels)),
        'integrated': float(np.sum(pixels)),
    }

    # 4D stats
    if window.volume is not None:
        stats.update(_compute_4d_stats(mask, window))

    return stats


def _compute_4d_stats(mask, window):
    """Compute per-Z-plane stats for a 4D window."""
    vol = window.volume
    if vol is None or vol.ndim < 4:
        return {}
    s1, s2 = mask
    t = min(window.currentIndex, vol.shape[0] - 1)
    results = {}
    nz = vol.shape[3] if vol.ndim == 4 else 1
    for z in range(nz):
        plane = vol[t, :, :, z]
        pixels = plane[s1, s2]
        results[f'z{z}_mean'] = float(np.mean(pixels))
    # Mean across all Z
    all_pixels = []
    for z in range(nz):
        all_pixels.append(vol[t, :, :, z][s1, s2])
    all_pixels = np.concatenate(all_pixels)
    results['mean_all_z'] = float(np.mean(all_pixels))
    return results


def compute_shape_descriptors(roi, window):
    """Shape analysis using skimage regionprops on the ROI mask."""
    try:
        from skimage.measure import regionprops, label as sklabel
    except ImportError:
        return {}

    s1, s2 = roi.getMask()
    if np.size(s1) == 0:
        return {}

    # Build binary mask image
    if window.nDims >= 3 and not window.metadata.get('is_rgb', False):
        shape = (window.mx, window.my)
    elif window.nDims == 2:
        shape = window.image.shape
    else:
        shape = (window.mx, window.my)

    mask_img = np.zeros(shape, dtype=bool)
    mask_img[s1, s2] = True
    labeled = sklabel(mask_img)
    props = regionprops(labeled)
    if len(props) == 0:
        return {}
    p = props[0]
    area = p.area
    perimeter = p.perimeter if hasattr(p, 'perimeter') else 0
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    return {
        'perimeter': float(perimeter),
        'eccentricity': float(p.eccentricity) if hasattr(p, 'eccentricity') else 0,
        'circularity': float(circularity),
        'major_axis': float(p.axis_major_length) if hasattr(p, 'axis_major_length') else 0,
        'minor_axis': float(p.axis_minor_length) if hasattr(p, 'axis_minor_length') else 0,
        'orientation': float(p.orientation) if hasattr(p, 'orientation') else 0,
    }
