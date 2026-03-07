# -*- coding: utf-8 -*-
"""Background subtraction with multiple methods.

Provides ROI-based (manual or auto-detected) and statistical background
subtraction, either per-frame or for the entire stack.
"""
from ..logger import logger
logger.debug("Started 'reading process/background.py'")

import numpy as np
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import (BaseProcess, SliderLabel, CheckBox, ComboBox,
                                 MissingWindowError)

__all__ = ['background_subtract']


# ---------------------------------------------------------------------------
# Auto-detection helper (ported from spt_batch_analysis.AutoBackgroundDetector)
# ---------------------------------------------------------------------------

def _auto_detect_background_roi(image, roi_width=10, roi_height=10, stride=5,
                                top_n=10):
    """Find the darkest, most temporally stable rectangular region.

    Parameters
    ----------
    image : ndarray
        3-D array (T, Y, X).
    roi_width, roi_height : int
        Size of the candidate ROI.
    stride : int
        Step between candidate positions.
    top_n : int
        Number of lowest-intensity candidates to evaluate for temporal
        stability.

    Returns
    -------
    dict or None
        ``{'y': int, 'x': int, 'w': int, 'h': int}`` of the best ROI,
        or *None* if no valid region was found.
    """
    if image.ndim == 2:
        max_proj = image
    else:
        max_proj = np.max(image, axis=0)

    h, w = max_proj.shape
    candidates = []
    for y in range(0, h - roi_height + 1, stride):
        for x in range(0, w - roi_width + 1, stride):
            region = max_proj[y:y + roi_height, x:x + roi_width]
            candidates.append((float(np.mean(region)), y, x))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])

    # Stage 2 — temporal stability check (3-D data only)
    if image.ndim == 3 and len(candidates) > 1:
        actual_top_n = min(top_n, len(candidates))
        evaluated = []
        for mean_int, y, x in candidates[:actual_top_n]:
            roi_stack = image[:, y:y + roi_height, x:x + roi_width]
            frame_means = np.mean(roi_stack, axis=(1, 2))
            temporal_std = float(np.std(frame_means))
            evaluated.append((temporal_std, mean_int, y, x))
        evaluated.sort(key=lambda e: e[0])
        _, _, best_y, best_x = evaluated[0]
    else:
        _, best_y, best_x = candidates[0]

    return {'y': best_y, 'x': best_x, 'w': roi_width, 'h': roi_height}


def _roi_mean_trace(image, rois):
    """Compute the mean background trace from a list of ROIs.

    Parameters
    ----------
    image : ndarray
        The image array from the current window (2-D or 3-D).
    rois : list
        List of flika ROI objects that have a ``getMask()`` method.

    Returns
    -------
    ndarray
        1-D array of length ``image.shape[0]`` (for 3-D) or length 1 (2-D),
        giving the mean ROI intensity per frame.
    """
    traces = []
    for roi in rois:
        t = roi.getTrace()
        if t is not None:
            traces.append(np.asarray(t, dtype=float))
    if not traces:
        raise ValueError("No valid ROI traces could be computed.")
    return np.mean(traces, axis=0)


def _stat_background(image, method, per_frame):
    """Compute a statistical background value.

    Parameters
    ----------
    image : ndarray
        2-D or 3-D image array.
    method : str
        One of ``'mean'``, ``'median'``, ``'mode_approx'``, ``'percentile_5'``,
        ``'percentile_25'``.
    per_frame : bool
        If *True* (and image is 3-D), compute per frame; otherwise compute a
        single value for the entire stack.

    Returns
    -------
    ndarray or float
    """
    funcs = {
        'Mean': lambda a: np.mean(a),
        'Median': lambda a: np.median(a),
        'Mode (approx: 3*median - 2*mean)': lambda a: 3.0 * np.median(a) - 2.0 * np.mean(a),
        '5th Percentile': lambda a: np.percentile(a, 5),
        '25th Percentile': lambda a: np.percentile(a, 25),
    }
    fn = funcs.get(method)
    if fn is None:
        raise ValueError(f"Unknown statistical method: {method}")

    if per_frame and image.ndim == 3:
        return np.array([fn(image[t]) for t in range(image.shape[0])])
    else:
        return fn(image)


# ---------------------------------------------------------------------------
# BaseProcess class
# ---------------------------------------------------------------------------

class Background_Subtract(BaseProcess):
    """background_subtract(method, stat_method, per_frame, roi_width, roi_height,
    stride, top_n, keepSourceWindow=False)

    Subtract background from the current image.

    Methods:
        * **Manual ROI** -- uses currently drawn ROIs as background regions.
        * **Auto ROI** -- automatically finds the darkest, most stable region.
        * **Statistical** -- computes a background value using a chosen statistic.

    Parameters:
        method (str): 'Manual ROI', 'Auto ROI', or 'Statistical'.
        stat_method (str): Statistic to use (only for Statistical method).
        per_frame (bool): Compute per-frame background (True) or single value (False).
        roi_width (int): Width of auto ROI (only for Auto ROI).
        roi_height (int): Height of auto ROI (only for Auto ROI).
        stride (int): Scan stride for auto detection (only for Auto ROI).
        top_n (int): Candidates to evaluate temporally (only for Auto ROI).
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()

        # --- Method selector ---
        method = ComboBox()
        method.addItems(['Manual ROI', 'Auto ROI', 'Statistical'])

        # --- Statistical sub-options ---
        stat_method = ComboBox()
        stat_method.addItems([
            'Mean', 'Median', 'Mode (approx: 3*median - 2*mean)',
            '5th Percentile', '25th Percentile',
        ])
        stat_method.setCurrentIndex(1)

        per_frame = CheckBox()
        per_frame.setChecked(True)

        # --- Auto ROI sub-options ---
        roi_width = SliderLabel(0)
        roi_width.setRange(3, 100)
        roi_width.setValue(10)

        roi_height = SliderLabel(0)
        roi_height.setRange(3, 100)
        roi_height.setValue(10)

        stride = SliderLabel(0)
        stride.setRange(1, 50)
        stride.setValue(5)

        top_n = SliderLabel(0)
        top_n.setRange(1, 50)
        top_n.setValue(10)

        # --- Info label ---
        self._info_label = QtWidgets.QLabel("")
        self._info_label.setStyleSheet("color: gray; font-style: italic;")

        # --- Preview ---
        preview_check = CheckBox()

        self.items.append({'name': 'method', 'string': 'Method', 'object': method})
        self.items.append({'name': 'stat_method', 'string': 'Statistic', 'object': stat_method})
        self.items.append({'name': 'per_frame', 'string': 'Per-frame subtraction', 'object': per_frame})
        self.items.append({'name': 'roi_width', 'string': 'Auto ROI Width', 'object': roi_width})
        self.items.append({'name': 'roi_height', 'string': 'Auto ROI Height', 'object': roi_height})
        self.items.append({'name': 'stride', 'string': 'Scan Stride', 'object': stride})
        self.items.append({'name': 'top_n', 'string': 'Top-N Candidates', 'object': top_n})
        self.items.append({'name': 'info', 'string': '', 'object': self._info_label})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview_check})

        super().gui()

        # Wire visibility toggling
        def _update_visibility(*_):
            m = method.currentText()
            is_stat = m == 'Statistical'
            is_auto = m == 'Auto ROI'
            is_manual = m == 'Manual ROI'

            stat_method.setVisible(is_stat)
            roi_width.setVisible(is_auto)
            roi_height.setVisible(is_auto)
            stride.setVisible(is_auto)
            top_n.setVisible(is_auto)

            # Find and toggle labels in the dialog
            for item in self.items:
                name = item['name']
                if name in ('stat_method',):
                    if hasattr(item['object'], 'label'):
                        item['object'].label.setVisible(is_stat)
                if name in ('roi_width', 'roi_height', 'stride', 'top_n'):
                    if hasattr(item['object'], 'label'):
                        item['object'].label.setVisible(is_auto)

            # Update info text
            if is_manual:
                rois = g.win.rois if g.win is not None else []
                n = len(rois)
                self._info_label.setText(
                    f"{n} ROI(s) on current window" if n > 0
                    else "Draw ROI(s) on the image to define background regions"
                )
            elif is_auto:
                self._info_label.setText(
                    "Will scan for the darkest, most stable region"
                )
            else:
                self._info_label.setText("")

        method.currentTextChanged.connect(_update_visibility)
        _update_visibility()

    def __call__(self, method='Manual ROI', stat_method='Median',
                 per_frame=True, roi_width=10, roi_height=10,
                 stride=5, top_n=10, keepSourceWindow=False):
        self.start(keepSourceWindow)
        image = self.tif.astype(np.float64)

        if method == 'Manual ROI':
            rois = g.win.rois if hasattr(g, 'win') and g.win is not None else []
            if len(rois) == 0:
                g.alert("No ROIs found on the current window. "
                        "Draw one or more ROIs to define background regions.")
                return None
            bg = _roi_mean_trace(image, rois)

        elif method == 'Auto ROI':
            result = _auto_detect_background_roi(
                image, roi_width=int(roi_width), roi_height=int(roi_height),
                stride=int(stride), top_n=int(top_n),
            )
            if result is None:
                g.alert("Auto detection failed — image too small or no valid region found.")
                return None

            y, x, w, h = result['y'], result['x'], result['w'], result['h']
            g.status_msg(f"Auto background ROI at ({x}, {y}) size {w}x{h}")

            # Draw the detected ROI on the window so the user can see it
            self._draw_auto_roi(x, y, w, h)

            if per_frame and image.ndim == 3:
                bg = np.array([np.mean(image[t, y:y+h, x:x+w])
                               for t in range(image.shape[0])])
            else:
                if image.ndim == 3:
                    bg = np.mean(image[:, y:y+h, x:x+w])
                else:
                    bg = np.mean(image[y:y+h, x:x+w])

        elif method == 'Statistical':
            bg = _stat_background(image, stat_method, per_frame)

        else:
            g.alert(f"Unknown method: {method}")
            return None

        # Subtract
        if isinstance(bg, np.ndarray) and bg.ndim == 1 and image.ndim == 3:
            # Per-frame: broadcast (T,) -> (T, 1, 1)
            self.newtif = image - bg[:, np.newaxis, np.newaxis]
        else:
            self.newtif = image - bg

        # Build descriptive name
        if method == 'Manual ROI':
            n_rois = len(g.win.rois) if hasattr(g, 'win') and g.win is not None else 0
            detail = f"ROI x{n_rois}"
        elif method == 'Auto ROI':
            detail = f"auto {roi_width}x{roi_height}"
        else:
            detail = stat_method
        pf = " per-frame" if per_frame else ""
        self.newname = f"{self.oldname} - BgSub({detail}{pf})"
        return self.end()

    def _draw_auto_roi(self, x, y, w, h):
        """Draw a rectangle ROI on the current window showing the detected region."""
        try:
            from ..roi import ROI_Drawing
            win = g.win
            if win is None:
                return
            roi_drawing = ROI_Drawing(win)
            # makeROI expects (x, y, w, h) for rectangle
            roi_drawing.makeROI('rectangle', [x, y, w, h])
        except Exception:
            pass  # Non-critical — just visual feedback

    def preview(self):
        preview_on = self.getValue('preview')
        if not preview_on:
            g.win.reset()
            return

        method = self.getValue('method')
        image = g.win.image.astype(np.float64)

        try:
            if method == 'Manual ROI':
                rois = g.win.rois
                if len(rois) == 0:
                    return
                bg = _roi_mean_trace(image, rois)
            elif method == 'Auto ROI':
                roi_w = int(self.getValue('roi_width'))
                roi_h = int(self.getValue('roi_height'))
                stride_val = int(self.getValue('stride'))
                top_n_val = int(self.getValue('top_n'))
                result = _auto_detect_background_roi(
                    image, roi_w, roi_h, stride_val, top_n_val)
                if result is None:
                    return
                y, x, w, h = result['y'], result['x'], result['w'], result['h']
                per_frame = self.getValue('per_frame')
                if per_frame and image.ndim == 3:
                    bg = np.array([np.mean(image[t, y:y+h, x:x+w])
                                   for t in range(image.shape[0])])
                else:
                    bg = np.mean(image[:, y:y+h, x:x+w]) if image.ndim == 3 else np.mean(image[y:y+h, x:x+w])
            else:
                stat_method = self.getValue('stat_method')
                per_frame = self.getValue('per_frame')
                bg = _stat_background(image, stat_method, per_frame)

            idx = g.win.currentIndex
            if isinstance(bg, np.ndarray) and bg.ndim == 1 and image.ndim == 3:
                preview_img = image[idx] - bg[idx]
            else:
                preview_img = image[idx] - (bg if np.isscalar(bg) else float(bg))

            g.win.imageview.setImage(preview_img, autoLevels=False)
        except Exception:
            g.win.reset()


background_subtract = Background_Subtract()

logger.debug("Completed 'reading process/background.py'")
