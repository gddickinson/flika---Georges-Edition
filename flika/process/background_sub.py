# -*- coding: utf-8 -*-
"""Background subtraction with multiple methods.

Provides ROI-based (manual or auto-detected) and statistical background
subtraction, either per-frame or for the entire stack.
"""
import numpy as np
from qtpy import QtWidgets, QtCore
import flika.global_vars as g
from flika.utils.BaseProcess import BaseProcess
from flika.utils.custom_widgets import SliderLabel, CheckBox, ComboBox

__all__ = ['background_subtract', 'scaled_average_subtract']


# ---------------------------------------------------------------------------
# Auto-detection helper
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

    # Stage 2 -- temporal stability check (3-D data only)
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
    """Compute the mean background trace from a list of ROIs."""
    traces = []
    for roi in rois:
        t = roi.getTrace()
        if t is not None:
            traces.append(np.asarray(t, dtype=float))
    if not traces:
        raise ValueError("No valid ROI traces could be computed.")
    return np.mean(traces, axis=0)


def _stat_background(image, method, per_frame):
    """Compute a statistical background value."""
    funcs = {
        'Mean': lambda a: np.mean(a),
        'Median': lambda a: np.median(a),
        'Mode (approx: 3*median - 2*mean)': lambda a: 3.0 * np.median(a) - 2.0 * np.mean(a),
        '5th Percentile': lambda a: np.percentile(a, 5),
        '25th Percentile': lambda a: np.percentile(a, 25),
    }
    fn = funcs.get(method)
    if fn is None:
        raise ValueError("Unknown statistical method: {}".format(method))

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

        method = ComboBox()
        method.addItems(['Manual ROI', 'Auto ROI', 'Statistical'])

        stat_method = ComboBox()
        stat_method.addItems([
            'Mean', 'Median', 'Mode (approx: 3*median - 2*mean)',
            '5th Percentile', '25th Percentile',
        ])
        stat_method.setCurrentIndex(1)

        per_frame = CheckBox()
        per_frame.setChecked(True)

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

        self._info_label = QtWidgets.QLabel("")
        self._info_label.setStyleSheet("color: gray; font-style: italic;")

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

        def _update_visibility(*_):
            m = method.currentText()
            is_stat = m == 'Statistical'
            is_auto = m == 'Auto ROI'

            stat_method.setVisible(is_stat)
            roi_width.setVisible(is_auto)
            roi_height.setVisible(is_auto)
            stride.setVisible(is_auto)
            top_n.setVisible(is_auto)

            if m == 'Manual ROI':
                rois = g.win.rois if g.win is not None else []
                n = len(rois)
                self._info_label.setText(
                    "{} ROI(s) on current window".format(n) if n > 0
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
                g.alert("Auto detection failed -- image too small or no valid region found.")
                return None

            y, x, w, h = result['y'], result['x'], result['w'], result['h']
            if g.m is not None:
                g.m.statusBar().showMessage(
                    "Auto background ROI at ({}, {}) size {}x{}".format(x, y, w, h))

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
            g.alert("Unknown method: {}".format(method))
            return None

        # Subtract
        if isinstance(bg, np.ndarray) and bg.ndim == 1 and image.ndim == 3:
            self.newtif = image - bg[:, np.newaxis, np.newaxis]
        else:
            self.newtif = image - bg

        # Build descriptive name
        if method == 'Manual ROI':
            n_rois = len(g.win.rois) if hasattr(g, 'win') and g.win is not None else 0
            detail = "ROI x{}".format(n_rois)
        elif method == 'Auto ROI':
            detail = "auto {}x{}".format(roi_width, roi_height)
        else:
            detail = stat_method
        pf = " per-frame" if per_frame else ""
        self.newname = "{} - BgSub({}{})".format(self.oldname, detail, pf)
        return self.end()

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


class Scaled_Average_Subtract(BaseProcess):
    """scaled_average_subtract(window_size=50, average_size=100, keepSourceWindow=False)

    Subtracts a scaled average image of the peak response from each frame.
    Useful for isolating local calcium puffs from a global calcium wave.

    Parameters:
        window_size (int): Number of frames for the rolling average window.
        average_size (int): Number of frames around the peak to average.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'window_size': 50, 'average_size': 100}

    def gui(self):
        self.gui_reset()
        window_size = SliderLabel(0)
        window_size.setRange(1, 10000)
        window_size.setValue(50)
        average_size = SliderLabel(0)
        average_size.setRange(1, 10000)
        average_size.setValue(100)
        self.items.append({'name': 'window_size', 'string': 'Rolling Average Window',
                           'object': window_size})
        self.items.append({'name': 'average_size', 'string': 'Peak Average Frames',
                           'object': average_size})
        super().gui()

    def __call__(self, window_size=50, average_size=100, keepSourceWindow=False):
        self.start(keepSourceWindow)
        A = self.tif.astype(np.float64)
        if A.ndim < 3:
            g.alert('Scaled Average Subtract requires a 3D stack')
            return None

        frames, height, width = A.shape

        win = g.win
        if win is not None and win.currentROI is not None:
            trace = np.asarray(win.currentROI.getTrace(), dtype=np.float64)
        else:
            trace = np.mean(A.reshape(frames, -1), axis=1)

        kernel = np.ones(window_size) / window_size
        if len(trace) < window_size:
            g.alert('Stack has fewer frames than the rolling average window')
            return None
        moving_avg = np.convolve(trace, kernel, mode='valid')
        moving_avg[moving_avg <= 0] = 1e-7

        peak_frame = np.argmax(moving_avg)

        half = average_size // 2
        start = max(0, peak_frame - half)
        end = min(frames, peak_frame + half)
        average_image = np.mean(A[start:end], axis=0)

        scale = moving_avg / np.max(moving_avg)
        offset = window_size // 2
        trim_end = offset + len(scale)

        result = np.zeros_like(A)
        scaled_stack = average_image[np.newaxis, :, :] * scale[:, np.newaxis, np.newaxis]
        result[offset:trim_end] = A[offset:trim_end] - scaled_stack

        self.newtif = result
        self.newname = self.oldname + ' - Scaled Avg Subtracted'
        return self.end()


scaled_average_subtract = Scaled_Average_Subtract()
