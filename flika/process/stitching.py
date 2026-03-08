# -*- coding: utf-8 -*-
"""Image stitching via phase cross-correlation.

Stitches two images (or stacks) using subpixel registration.
Supports manual grid layout and automatic overlap detection.
"""
from ..logger import logger
logger.debug("Started 'reading process/stitching.py'")

import numpy as np
from qtpy import QtWidgets
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, ComboBox, CheckBox, MissingWindowError

__all__ = ['stitch_images']


def _register_pair(img_a, img_b, overlap_fraction=0.1):
    """Find the translational offset between two images using phase correlation.

    Returns (dy, dx) shift of img_b relative to img_a.
    """
    from scipy.ndimage import shift as ndi_shift
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        from skimage.feature import register_translation as phase_cross_correlation

    # Use the overlap region for registration
    h, w = img_a.shape[:2]
    overlap_px = max(int(w * overlap_fraction), 10)

    ref_region = img_a[:, -overlap_px:]
    mov_region = img_b[:, :overlap_px]

    shift_vals, error, _ = phase_cross_correlation(ref_region, mov_region, upsample_factor=10)
    # The shift is in the overlap region; translate to full-image coordinates
    dy = shift_vals[0]
    dx = shift_vals[1] + (w - overlap_px)
    return dy, dx


def _blend_overlap(canvas, img, y0, x0, blend_width=32):
    """Place img onto canvas at (y0, x0) with linear blending in overlapping regions."""
    h, w = img.shape[:2]
    ch, cw = canvas.shape[:2]

    # Clamp destination region
    dy0 = max(0, y0)
    dx0 = max(0, x0)
    dy1 = min(ch, y0 + h)
    dx1 = min(cw, x0 + w)

    sy0 = dy0 - y0
    sx0 = dx0 - x0
    sy1 = sy0 + (dy1 - dy0)
    sx1 = sx0 + (dx1 - dx0)

    region = canvas[dy0:dy1, dx0:dx1]
    src = img[sy0:sy1, sx0:sx1]

    # Where canvas is zero, just place the image; otherwise blend
    mask = np.abs(region) > 1e-10
    if mask.ndim > 2:
        mask = mask.any(axis=-1)
    alpha = np.ones(mask.shape, dtype=np.float64) * 0.5
    # Ramp from 0 to 0.5 in the overlap
    if blend_width > 0:
        ramp = np.minimum(np.arange(alpha.shape[1], dtype=np.float64) / blend_width, 1.0) * 0.5
        alpha = np.broadcast_to(ramp[None, :], alpha.shape).copy()
    alpha[~mask] = 1.0

    if src.ndim > 2:
        alpha = alpha[..., None]
    canvas[dy0:dy1, dx0:dx1] = alpha * src + (1 - alpha) * region


class Stitch_Images(BaseProcess):
    """stitch_images(window1, window2, direction, overlap, blend, keepSourceWindow=False)

    Stitches two images together using phase cross-correlation alignment.

    Parameters:
        window1 (Window): Left/top image.
        window2 (Window): Right/bottom image.
        direction (str): 'Horizontal' or 'Vertical'.
        overlap (float): Expected overlap fraction (0.01-0.5).
        blend (bool): Enable linear blending in overlap region.
    Returns:
        newWindow with stitched image
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        direction = ComboBox()
        direction.addItems(['Horizontal', 'Vertical'])
        overlap = SliderLabel(2)
        overlap.setRange(0.01, 0.5)
        overlap.setValue(0.1)
        blend = CheckBox()
        blend.setChecked(True)
        self.items.append({'name': 'window1', 'string': 'Image 1 (Left/Top)', 'object': window1})
        self.items.append({'name': 'window2', 'string': 'Image 2 (Right/Bottom)', 'object': window2})
        self.items.append({'name': 'direction', 'string': 'Direction', 'object': direction})
        self.items.append({'name': 'overlap', 'string': 'Overlap Fraction', 'object': overlap})
        self.items.append({'name': 'blend', 'string': 'Blend Overlap', 'object': blend})
        super().gui()

    def __call__(self, window1, window2, direction='Horizontal', overlap=0.1,
                 blend=True, keepSourceWindow=False):
        self.keepSourceWindow = keepSourceWindow
        g.status_msg('Stitching images...')

        if window1 is None or window2 is None:
            raise MissingWindowError("Select two windows to stitch.")

        A = window1.image.astype(np.float64)
        B = window2.image.astype(np.float64)

        # Handle stacks: stitch frame by frame using first frame for registration
        is_stack = A.ndim >= 3 and B.ndim >= 3

        if direction == 'Vertical':
            A = np.swapaxes(A, -2, -1) if A.ndim == 2 else np.swapaxes(A, -2, -1)
            B = np.swapaxes(B, -2, -1) if B.ndim == 2 else np.swapaxes(B, -2, -1)

        if is_stack:
            ref_a = A[0]
            ref_b = B[0]
        else:
            ref_a = A
            ref_b = B

        dy, dx = _register_pair(ref_a, ref_b, overlap)
        dy, dx = int(round(dy)), int(round(dx))

        # Compute canvas size
        if is_stack:
            h_a, w_a = A.shape[1], A.shape[2]
            h_b, w_b = B.shape[1], B.shape[2]
        else:
            h_a, w_a = A.shape[0], A.shape[1]
            h_b, w_b = B.shape[0], B.shape[1]

        canvas_h = max(h_a, dy + h_b) - min(0, dy)
        canvas_w = max(w_a, dx + w_b) - min(0, dx)
        offset_y = -min(0, dy)
        offset_x = -min(0, dx)

        if is_stack:
            n_frames = min(A.shape[0], B.shape[0])
            result = np.zeros((n_frames, canvas_h, canvas_w), dtype=np.float64)
            for t in range(n_frames):
                canvas = np.zeros((canvas_h, canvas_w), dtype=np.float64)
                canvas[offset_y:offset_y+h_a, offset_x:offset_x+w_a] = A[t]
                if blend:
                    _blend_overlap(canvas, B[t], offset_y + dy, offset_x + dx)
                else:
                    by0 = max(0, offset_y + dy)
                    bx0 = max(0, offset_x + dx)
                    by1 = min(canvas_h, offset_y + dy + h_b)
                    bx1 = min(canvas_w, offset_x + dx + w_b)
                    sy0 = by0 - (offset_y + dy)
                    sx0 = bx0 - (offset_x + dx)
                    canvas[by0:by1, bx0:bx1] = B[t, sy0:sy0+(by1-by0), sx0:sx0+(bx1-bx0)]
                result[t] = canvas
        else:
            result = np.zeros((canvas_h, canvas_w), dtype=np.float64)
            result[offset_y:offset_y+h_a, offset_x:offset_x+w_a] = A
            if blend:
                _blend_overlap(result, B, offset_y + dy, offset_x + dx)
            else:
                by0 = max(0, offset_y + dy)
                bx0 = max(0, offset_x + dx)
                by1 = min(canvas_h, offset_y + dy + h_b)
                bx1 = min(canvas_w, offset_x + dx + w_b)
                sy0 = by0 - (offset_y + dy)
                sx0 = bx0 - (offset_x + dx)
                result[by0:by1, bx0:bx1] = B[sy0:sy0+(by1-by0), sx0:sx0+(bx1-bx0)]

        if direction == 'Vertical':
            result = np.swapaxes(result, -2, -1)

        self.newtif = result
        self.oldwindow = window1
        self.oldname = window1.name
        self.newname = f'{window1.name} + {window2.name} (Stitched)'

        if keepSourceWindow is False:
            window2.close()
        g.status_msg('Stitching complete. Shift: dy={}, dx={}'.format(dy, dx))
        return self.end()


stitch_images = Stitch_Images()

logger.debug("Completed 'reading process/stitching.py'")
