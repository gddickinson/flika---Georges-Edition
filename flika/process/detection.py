# -*- coding: utf-8 -*-
import numpy as np
from skimage import feature, morphology
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector, MissingWindowError
from ..utils.ndim import per_plane
from ..logger import logger

logger.debug("Started 'reading process/detection.py'")

__all__ = ['blob_detection_log', 'blob_detection_doh', 'peak_local_max',
           'template_match', 'local_maxima_detect']


def _draw_blob_circles(image, blobs):
    """Draw circles at blob locations on a copy of the image.

    Parameters
    ----------
    image : 2D ndarray
        Single frame image.
    blobs : ndarray (N, 3)
        Each row is (y, x, sigma).

    Returns
    -------
    marked : 2D ndarray
        Image with blob markers drawn.
    """
    marked = np.array(image, dtype=np.float64)
    mark_val = np.max(marked) * 1.2 if np.max(marked) != 0 else 1.0
    for y, x, sigma in blobs:
        r = int(np.ceil(sigma * np.sqrt(2)))
        y, x = int(round(y)), int(round(x))
        for angle in np.linspace(0, 2 * np.pi, max(20, int(2 * np.pi * r))):
            cy = int(round(y + r * np.sin(angle)))
            cx = int(round(x + r * np.cos(angle)))
            if 0 <= cy < marked.shape[0] and 0 <= cx < marked.shape[1]:
                marked[cy, cx] = mark_val
    return marked


def _draw_peaks(image, coords):
    """Mark peak locations on a copy of the image.

    Parameters
    ----------
    image : 2D ndarray
        Single frame image.
    coords : ndarray (N, 2)
        Each row is (y, x).

    Returns
    -------
    marked : 2D ndarray
        Image with peak markers drawn.
    """
    marked = np.array(image, dtype=np.float64)
    mark_val = np.max(marked) * 1.2 if np.max(marked) != 0 else 1.0
    for y, x in coords:
        y, x = int(round(y)), int(round(x))
        # Draw a small cross
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            cy, cx = y + dy, x + dx
            if 0 <= cy < marked.shape[0] and 0 <= cx < marked.shape[1]:
                marked[cy, cx] = mark_val
    return marked


class Blob_Detection_LoG(BaseProcess):
    """blob_detection_log(min_sigma=1.0, max_sigma=10.0, num_sigma=10, threshold=0.1, keepSourceWindow=False)

    Detect blobs using the Laplacian of Gaussian (LoG) method.
    Blobs are detected in each frame and their coordinates are stored
    in window.metadata['blobs'].

    Parameters:
        min_sigma (float): Minimum sigma for Gaussian kernel
        max_sigma (float): Maximum sigma for Gaussian kernel
        num_sigma (int): Number of intermediate sigma values
        threshold (float): Detection threshold

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        min_sigma = SliderLabel(2)
        min_sigma.setRange(0.5, 20)
        min_sigma.setValue(1.0)
        max_sigma = SliderLabel(2)
        max_sigma.setRange(1, 50)
        max_sigma.setValue(10.0)
        num_sigma = SliderLabel(0)
        num_sigma.setRange(5, 50)
        num_sigma.setValue(10)
        threshold = SliderLabel(3)
        threshold.setRange(0.001, 1.0)
        threshold.setValue(0.1)
        self.items.append({'name': 'min_sigma', 'string': 'Min Sigma', 'object': min_sigma})
        self.items.append({'name': 'max_sigma', 'string': 'Max Sigma', 'object': max_sigma})
        self.items.append({'name': 'num_sigma', 'string': 'Num Sigma', 'object': num_sigma})
        self.items.append({'name': 'threshold', 'string': 'Threshold', 'object': threshold})
        super().gui()

    def __call__(self, min_sigma=1.0, max_sigma=10.0, num_sigma=10, threshold=0.1, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        all_blobs = []
        if tif.ndim == 3:
            result = np.zeros_like(tif, dtype=np.float64)
            for i in range(len(tif)):
                blobs = feature.blob_log(tif[i], min_sigma=min_sigma,
                                         max_sigma=max_sigma,
                                         num_sigma=int(num_sigma),
                                         threshold=threshold)
                result[i] = _draw_blob_circles(tif[i], blobs)
                # Store with frame index
                for b in blobs:
                    all_blobs.append([i, b[0], b[1], b[2]])
        elif tif.ndim == 2:
            blobs = feature.blob_log(tif, min_sigma=min_sigma,
                                     max_sigma=max_sigma,
                                     num_sigma=int(num_sigma),
                                     threshold=threshold)
            result = _draw_blob_circles(tif, blobs)
            for b in blobs:
                all_blobs.append([0, b[0], b[1], b[2]])
        self.newtif = result
        self.newname = self.oldname + ' - LoG Blobs'
        newWindow = self.end()
        if newWindow is not None:
            blob_array = np.array(all_blobs) if len(all_blobs) > 0 else np.empty((0, 4))
            newWindow.metadata['blobs'] = blob_array
        return newWindow

blob_detection_log = Blob_Detection_LoG()


class Blob_Detection_DoH(BaseProcess):
    """blob_detection_doh(min_sigma=1.0, max_sigma=10.0, num_sigma=10, threshold=0.01, keepSourceWindow=False)

    Detect blobs using the Determinant of Hessian (DoH) method.
    Blobs are detected in each frame and their coordinates are stored
    in window.metadata['blobs'].

    Parameters:
        min_sigma (float): Minimum sigma for Gaussian kernel
        max_sigma (float): Maximum sigma for Gaussian kernel
        num_sigma (int): Number of intermediate sigma values
        threshold (float): Detection threshold

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        min_sigma = SliderLabel(2)
        min_sigma.setRange(0.5, 20)
        min_sigma.setValue(1.0)
        max_sigma = SliderLabel(2)
        max_sigma.setRange(1, 50)
        max_sigma.setValue(10.0)
        num_sigma = SliderLabel(0)
        num_sigma.setRange(5, 50)
        num_sigma.setValue(10)
        threshold = SliderLabel(3)
        threshold.setRange(0.001, 1.0)
        threshold.setValue(0.01)
        self.items.append({'name': 'min_sigma', 'string': 'Min Sigma', 'object': min_sigma})
        self.items.append({'name': 'max_sigma', 'string': 'Max Sigma', 'object': max_sigma})
        self.items.append({'name': 'num_sigma', 'string': 'Num Sigma', 'object': num_sigma})
        self.items.append({'name': 'threshold', 'string': 'Threshold', 'object': threshold})
        super().gui()

    def __call__(self, min_sigma=1.0, max_sigma=10.0, num_sigma=10, threshold=0.01, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        all_blobs = []
        if tif.ndim == 3:
            result = np.zeros_like(tif, dtype=np.float64)
            for i in range(len(tif)):
                blobs = feature.blob_doh(tif[i], min_sigma=min_sigma,
                                         max_sigma=max_sigma,
                                         num_sigma=int(num_sigma),
                                         threshold=threshold)
                result[i] = _draw_blob_circles(tif[i], blobs)
                for b in blobs:
                    all_blobs.append([i, b[0], b[1], b[2]])
        elif tif.ndim == 2:
            blobs = feature.blob_doh(tif, min_sigma=min_sigma,
                                     max_sigma=max_sigma,
                                     num_sigma=int(num_sigma),
                                     threshold=threshold)
            result = _draw_blob_circles(tif, blobs)
            for b in blobs:
                all_blobs.append([0, b[0], b[1], b[2]])
        self.newtif = result
        self.newname = self.oldname + ' - DoH Blobs'
        newWindow = self.end()
        if newWindow is not None:
            blob_array = np.array(all_blobs) if len(all_blobs) > 0 else np.empty((0, 4))
            newWindow.metadata['blobs'] = blob_array
        return newWindow

blob_detection_doh = Blob_Detection_DoH()


class Peak_Local_Max(BaseProcess):
    """peak_local_max(min_distance=5, threshold_abs=0.0, num_peaks=0, keepSourceWindow=False)

    Detect peaks in an image using local maximum detection.
    Peak coordinates are stored in window.metadata['peaks'].

    Parameters:
        min_distance (int): Minimum number of pixels separating peaks
        threshold_abs (float): Minimum intensity of peaks (0 for auto)
        num_peaks (int): Maximum number of peaks (0 for unlimited)

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        min_distance = SliderLabel(0)
        min_distance.setRange(1, 50)
        min_distance.setValue(5)
        threshold_abs = SliderLabel(2)
        if g.win is not None:
            threshold_abs.setRange(0, np.max(g.win.image))
            threshold_abs.setValue(np.mean(g.win.image))
        else:
            threshold_abs.setRange(0, 1000)
            threshold_abs.setValue(0)
        num_peaks = SliderLabel(0)
        num_peaks.setRange(0, 10000)
        num_peaks.setValue(0)
        self.items.append({'name': 'min_distance', 'string': 'Min Distance', 'object': min_distance})
        self.items.append({'name': 'threshold_abs', 'string': 'Threshold', 'object': threshold_abs})
        self.items.append({'name': 'num_peaks', 'string': 'Num Peaks (0=all)', 'object': num_peaks})
        super().gui()

    def __call__(self, min_distance=5, threshold_abs=0.0, num_peaks=0, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        all_peaks = []
        peak_kwargs = {'min_distance': int(min_distance)}
        if threshold_abs > 0:
            peak_kwargs['threshold_abs'] = threshold_abs
        if num_peaks > 0:
            peak_kwargs['num_peaks'] = int(num_peaks)
        if tif.ndim == 3:
            result = np.zeros_like(tif, dtype=np.float64)
            for i in range(len(tif)):
                coords = feature.peak_local_max(tif[i], **peak_kwargs)
                result[i] = _draw_peaks(tif[i], coords)
                for c in coords:
                    all_peaks.append([i, c[0], c[1]])
        elif tif.ndim == 2:
            coords = feature.peak_local_max(tif, **peak_kwargs)
            result = _draw_peaks(tif, coords)
            for c in coords:
                all_peaks.append([0, c[0], c[1]])
        self.newtif = result
        self.newname = self.oldname + ' - Peak Local Max'
        newWindow = self.end()
        if newWindow is not None:
            peak_array = np.array(all_peaks) if len(all_peaks) > 0 else np.empty((0, 3))
            newWindow.metadata['peaks'] = peak_array
        return newWindow

peak_local_max = Peak_Local_Max()


class Template_Match(BaseProcess):
    """template_match(template_window, keepSourceWindow=False)

    Perform template matching using normalized cross-correlation.
    Uses another window as the template image.

    Parameters:
        template_window (Window): Window containing the template image

    Returns:
        flika.window.Window (correlation map)
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        template_window = WindowSelector()
        self.items.append({'name': 'template_window', 'string': 'Template Window', 'object': template_window})
        super().gui()

    def __call__(self, template_window, keepSourceWindow=False):
        if template_window is None:
            g.alert("You cannot execute '{}' without selecting a template window first.".format(self.__name__))
            return None
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        template = template_window.image.astype(np.float64)
        # Use 2D template (take current frame if template is a stack)
        if template.ndim == 3:
            template = template[template_window.currentIndex]
        if tif.ndim == 3:
            result = np.zeros((tif.shape[0],
                               tif.shape[1] - template.shape[0] + 1,
                               tif.shape[2] - template.shape[1] + 1), dtype=np.float64)
            for i in range(len(tif)):
                result[i] = feature.match_template(tif[i], template)
        elif tif.ndim == 2:
            result = feature.match_template(tif, template)
        self.newtif = result
        self.newname = self.oldname + ' - Template Match'
        return self.end()

template_match = Template_Match()


class Local_Maxima_Detect(BaseProcess):
    """local_maxima_detect(keepSourceWindow=False)

    Detect local maxima using morphological analysis.
    Returns a binary image marking all local maxima positions.

    Returns:
        flika.window.Window (binary image of local maxima)
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        if tif.ndim == 3:
            result = np.zeros_like(tif, dtype=np.uint8)
            for i in range(len(tif)):
                result[i] = morphology.local_maxima(tif[i]).astype(np.uint8)
        elif tif.ndim == 2:
            result = morphology.local_maxima(tif).astype(np.uint8)
        self.newtif = result
        self.newname = self.oldname + ' - Local Maxima'
        return self.end()

local_maxima_detect = Local_Maxima_Detect()


logger.debug("Completed 'reading process/detection.py'")
