# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/kymograph.py'")
import numpy as np
import scipy.ndimage
import scipy.signal
from skimage.measure import profile_line
from qtpy import QtWidgets, QtCore
import pyqtgraph as pg
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from ..utils.ndim import per_plane

__all__ = ['kymograph']


def _extract_line_profile(image_2d, roi, line_width):
    """Extract intensity profile along an ROI line from a 2D image.

    Parameters
    ----------
    image_2d : ndarray (2D)
        Single image frame.
    roi : ROI_Base
        The ROI to extract the profile along.
    line_width : int
        Width of the line for averaging perpendicular pixels.

    Returns
    -------
    profile : ndarray (1D)
        Intensity values along the line.
    """
    kind = getattr(roi, 'kind', None)

    if kind == 'line':
        pts = roi.getPoints()
        x0, y0 = pts[0][0], pts[0][1]
        x1, y1 = pts[1][0], pts[1][1]
        prof = profile_line(image_2d, (y0, x0), (y1, x1),
                            linewidth=line_width, mode='constant')
        return prof

    elif kind == 'rect_line':
        # Multi-segment line: extract along each segment and concatenate
        handle_positions = roi.getHandlePositions()
        if handle_positions is None or len(handle_positions) < 2:
            return np.array([])
        # getHandlePositions returns list of (index, Point) tuples
        points = []
        for item in handle_positions:
            if isinstance(item, tuple):
                pos = item[1]
            else:
                pos = item
            points.append((float(pos.x()), float(pos.y())))

        segments = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            seg = profile_line(image_2d, (y0, x0), (y1, x1),
                               linewidth=line_width, mode='constant')
            # Avoid duplicating the junction point
            if i > 0 and len(seg) > 0:
                seg = seg[1:]
            segments.append(seg)

        if segments:
            return np.concatenate(segments)
        return np.array([])

    elif kind == 'freehand':
        # Freehand ROI: extract along each consecutive pair of points
        pts = roi.getPoints()
        if pts is None or len(pts) < 2:
            return np.array([])
        segments = []
        for i in range(len(pts) - 1):
            x0, y0 = float(pts[i][0]), float(pts[i][1])
            x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            seg = profile_line(image_2d, (y0, x0), (y1, x1),
                               linewidth=line_width, mode='constant')
            if i > 0 and len(seg) > 0:
                seg = seg[1:]
            segments.append(seg)
        if segments:
            return np.concatenate(segments)
        return np.array([])

    else:
        # Fallback: try to use pts directly
        pts = roi.getPoints()
        if pts is None or len(pts) < 2:
            return np.array([])
        x0, y0 = float(pts[0][0]), float(pts[0][1])
        x1, y1 = float(pts[1][0]), float(pts[1][1])
        prof = profile_line(image_2d, (y0, x0), (y1, x1),
                            linewidth=line_width, mode='constant')
        return prof


class Kymograph(BaseProcess):
    """kymograph(line_width=1, temporal_binning=1, normalize=False, detrend=False, gaussian_sigma=0, keepSourceWindow=False)

    Generate a kymograph from the current window along the active line ROI.

    A kymograph is a 2D representation of intensity along a spatial line over
    time.  Each row corresponds to a time frame, and each column to a position
    along the ROI line.

    Parameters:
        line_width (int): Width of the line for averaging perpendicular pixels
            (1-50, default 1).
        temporal_binning (int): Number of consecutive frames to average
            together (1-20, default 1).
        normalize (bool): If True, normalize the result to [0, 1].
        detrend (bool): If True, remove linear trend along the time axis
            using ``scipy.signal.detrend``.
        gaussian_sigma (float): Standard deviation for Gaussian smoothing
            (0 = no smoothing, max 10).
        keepSourceWindow (bool): If False, a new Window is created with the
            result.  Otherwise the current Window is reused.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()
        self._preview_win = None

    def get_init_settings_dict(self):
        s = dict()
        s['line_width'] = 1
        s['temporal_binning'] = 1
        s['normalize'] = False
        s['detrend'] = False
        s['gaussian_sigma'] = 0
        return s

    def gui(self):
        self.gui_reset()

        line_width = SliderLabel(0)
        line_width.setRange(1, 50)
        line_width.setValue(1)

        temporal_binning = SliderLabel(0)
        temporal_binning.setRange(1, 20)
        temporal_binning.setValue(1)

        normalize = CheckBox()
        normalize.setValue(False)

        detrend = CheckBox()
        detrend.setValue(False)

        gaussian_sigma = SliderLabel(2)
        gaussian_sigma.setRange(0, 10)
        gaussian_sigma.setValue(0)

        self.items.append({'name': 'line_width', 'string': 'Line Width', 'object': line_width})
        self.items.append({'name': 'temporal_binning', 'string': 'Temporal Binning', 'object': temporal_binning})
        self.items.append({'name': 'normalize', 'string': 'Normalize', 'object': normalize})
        self.items.append({'name': 'detrend', 'string': 'Detrend', 'object': detrend})
        self.items.append({'name': 'gaussian_sigma', 'string': 'Gaussian Sigma', 'object': gaussian_sigma})
        super().gui()

    def __call__(self, line_width=1, temporal_binning=1, normalize=False,
                 detrend=False, gaussian_sigma=0, keepSourceWindow=False):
        self.start(keepSourceWindow)

        tif = self.tif
        if tif.ndim < 3:
            g.alert("Kymograph requires at least a 3D (time, x, y) image.")
            return None

        roi = g.win.currentROI
        if roi is None:
            g.alert("Draw a line ROI first.")
            return None

        # Extract intensity profiles along the ROI line for each frame
        nFrames = tif.shape[0]
        profiles = []
        for t in range(nFrames):
            if tif.ndim == 3:
                frame = tif[t]
            elif tif.ndim == 4:
                # For 4D data, use the first z-slice or channel
                frame = tif[t, 0]
            else:
                frame = tif[t]

            prof = _extract_line_profile(frame, roi, line_width)
            profiles.append(prof)

        # Ensure all profiles have the same length
        lengths = [len(p) for p in profiles]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            profiles = [p[:min_len] for p in profiles]

        kymo = np.array(profiles, dtype=np.float64)  # shape: (nFrames, nPixels)

        # Apply temporal binning
        if temporal_binning > 1:
            n = kymo.shape[0]
            n_binned = n // temporal_binning
            if n_binned > 0:
                kymo = kymo[:n_binned * temporal_binning]
                kymo = kymo.reshape(n_binned, temporal_binning, -1).mean(axis=1)

        # Apply detrend along time axis
        if detrend:
            kymo = scipy.signal.detrend(kymo, axis=0)

        # Apply Gaussian smoothing
        if gaussian_sigma > 0:
            kymo = scipy.ndimage.gaussian_filter(kymo, sigma=gaussian_sigma)

        # Normalize to [0, 1]
        if normalize:
            kymo_min = kymo.min()
            kymo_max = kymo.max()
            if kymo_max > kymo_min:
                kymo = (kymo - kymo_min) / (kymo_max - kymo_min)

        self.newtif = kymo
        self.newname = self.oldname + ' - Kymograph'
        return self.end()

    def preview(self):
        """Extract kymograph from current ROI and display in a preview window."""
        if g.win is None:
            g.alert("No window is open.")
            return

        tif = g.win.image
        if tif.ndim < 3:
            g.alert("Kymograph requires at least a 3D image.")
            return

        roi = g.win.currentROI
        if roi is None:
            g.alert("Draw a line ROI first.")
            return

        # Read current GUI values if available
        line_width = 1
        temporal_binning = 1
        do_normalize = False
        do_detrend = False
        gaussian_sigma = 0.0

        if hasattr(self, 'items') and self.items:
            for item in self.items:
                if item['name'] == 'line_width':
                    line_width = int(item['object'].value())
                elif item['name'] == 'temporal_binning':
                    temporal_binning = int(item['object'].value())
                elif item['name'] == 'normalize':
                    do_normalize = item['object'].isChecked()
                elif item['name'] == 'detrend':
                    do_detrend = item['object'].isChecked()
                elif item['name'] == 'gaussian_sigma':
                    gaussian_sigma = float(item['object'].value())

        # Build kymograph
        nFrames = tif.shape[0]
        profiles = []
        for t in range(nFrames):
            if tif.ndim == 3:
                frame = tif[t]
            elif tif.ndim == 4:
                frame = tif[t, 0]
            else:
                frame = tif[t]
            prof = _extract_line_profile(frame, roi, line_width)
            profiles.append(prof)

        lengths = [len(p) for p in profiles]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            profiles = [p[:min_len] for p in profiles]

        kymo = np.array(profiles, dtype=np.float64)

        if temporal_binning > 1:
            n = kymo.shape[0]
            n_binned = n // temporal_binning
            if n_binned > 0:
                kymo = kymo[:n_binned * temporal_binning]
                kymo = kymo.reshape(n_binned, temporal_binning, -1).mean(axis=1)

        if do_detrend:
            kymo = scipy.signal.detrend(kymo, axis=0)

        if gaussian_sigma > 0:
            kymo = scipy.ndimage.gaussian_filter(kymo, sigma=gaussian_sigma)

        if do_normalize:
            kymo_min = kymo.min()
            kymo_max = kymo.max()
            if kymo_max > kymo_min:
                kymo = (kymo - kymo_min) / (kymo_max - kymo_min)

        # Display in a pyqtgraph window
        if self._preview_win is None or not self._preview_win.isVisible():
            self._preview_win = pg.GraphicsLayoutWidget()
            self._preview_win.setWindowTitle('Kymograph Preview')
            self._preview_win.resize(600, 400)

        self._preview_win.clear()
        plot = self._preview_win.addPlot()
        img_item = pg.ImageItem(kymo.T)
        plot.addItem(img_item)
        plot.setLabel('bottom', 'Frame')
        plot.setLabel('left', 'Position (pixels)')
        plot.setAspectLocked(False)
        self._preview_win.show()


kymograph = Kymograph()


logger.debug("Completed 'reading process/kymograph.py'")
