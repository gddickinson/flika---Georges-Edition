# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/watershed.py'")
import numpy as np
import scipy.ndimage
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess, SliderLabel, WindowSelector, MissingWindowError, CheckBox, ComboBox
from ..utils.ndim import per_plane

__all__ = ['distance_transform', 'watershed_segmentation']


@per_plane
def _distance_transform_impl(tif):
    """Apply Euclidean distance transform, handling 2D and 3D arrays."""
    if tif.ndim == 2:
        return scipy.ndimage.distance_transform_edt(tif > 0).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros_like(tif, dtype=np.float64)
        for i in range(len(result)):
            result[i] = scipy.ndimage.distance_transform_edt(tif[i] > 0)
        return result
    return tif


class Distance_Transform(BaseProcess):
    """distance_transform(keepSourceWindow=False)

    Apply the Euclidean distance transform to a binary image.

    Each foreground pixel (value > 0) is replaced with its Euclidean distance
    to the nearest background pixel.  This is useful as a preprocessing step
    for watershed segmentation or for measuring object thickness.

    Parameters:
        keepSourceWindow (bool): If False, a new Window is created with the
            result.  Otherwise the current Window is reused.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Distance transform does not support float16 type arrays")
            return None
        self.newtif = _distance_transform_impl(np.copy(self.tif))
        self.newname = self.oldname + ' - Distance Transform'
        return self.end()

distance_transform = Distance_Transform()


# ---------------------------------------------------------------------------
# Watershed Segmentation
# ---------------------------------------------------------------------------

def _get_markers_auto(binary_mask, distance, min_distance):
    """Generate markers automatically from local maxima of the distance transform.

    Parameters
    ----------
    binary_mask : ndarray (2D, bool)
        Foreground mask.
    distance : ndarray (2D, float)
        Euclidean distance transform of *binary_mask*.
    min_distance : int
        Minimum number of pixels separating peaks.

    Returns
    -------
    markers : ndarray (2D, int32)
        Labelled marker image.
    """
    try:
        from skimage.feature import peak_local_max
    except ImportError:
        g.alert("scikit-image is required for watershed segmentation.  "
                "Install it with:  pip install scikit-image")
        return None

    coords = peak_local_max(distance, min_distance=min_distance,
                            labels=binary_mask)
    if coords.size == 0:
        return None
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = scipy.ndimage.label(mask)
    return markers


def _get_markers_from_points(scatter_points, shape):
    """Create a marker image from user-placed scatter points.

    Parameters
    ----------
    scatter_points : list
        List of point entries.  Each entry is ``[x, y, ...]`` where *x* and
        *y* are pixel coordinates.
    shape : tuple of int
        ``(height, width)`` of the image.

    Returns
    -------
    markers : ndarray (2D, int32) or None
        Labelled marker image, or *None* if no points are available.
    """
    if scatter_points is None or len(scatter_points) == 0:
        return None
    markers = np.zeros(shape, dtype=np.int32)
    for idx, pt in enumerate(scatter_points, start=1):
        # scatter_points entries are [x, y, color, size] – x is column, y is row
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            markers[y, x] = idx
    if markers.max() == 0:
        return None
    return markers


def _watershed_single_frame(image_2d, binary_mask, markers, connectivity_val):
    """Run watershed on a single 2D frame.

    Parameters
    ----------
    image_2d : ndarray (2D)
        Intensity image (used as the landscape for watershed; negated internally
        so that bright regions become basins).
    binary_mask : ndarray (2D, bool)
        Foreground mask.
    markers : ndarray (2D, int32)
        Labelled seed regions.
    connectivity_val : int
        1 for 4-connected neighbourhood, 2 for 8-connected.

    Returns
    -------
    labels : ndarray (2D, int32)
        Labelled segmentation result.
    """
    try:
        from skimage.segmentation import watershed
    except ImportError:
        g.alert("scikit-image is required for watershed segmentation.  "
                "Install it with:  pip install scikit-image")
        return np.zeros_like(image_2d, dtype=np.int32)

    labels = watershed(-image_2d, markers, mask=binary_mask,
                       connectivity=connectivity_val)
    return labels.astype(np.int32)


class Watershed_Segmentation(BaseProcess):
    """watershed_segmentation(method='Auto Markers', min_distance=10, connectivity='4-connected', keepSourceWindow=False)

    Marker-controlled watershed segmentation.

    Segments foreground regions of a binary (or intensity) image using the
    watershed algorithm.  Markers can be generated automatically from the
    distance transform, or placed manually as scatter points on the window.

    Parameters:
        method (str): How to generate marker seeds.

            * ``'Auto Markers'`` -- find local maxima in the distance transform.
            * ``'From Points'``  -- use scatter points placed on the window.
            * ``'From Binary'``  -- (reserved for future use; falls back to auto).

        min_distance (int): Minimum pixel distance between peaks when using
            automatic marker detection (ignored for other methods).
        connectivity (str): ``'4-connected'`` or ``'8-connected'`` neighbourhood
            used by the watershed algorithm.
        keepSourceWindow (bool): If False, a new Window is created with the
            result.  Otherwise the current Window is reused.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        method = ComboBox()
        method.addItem('Auto Markers')
        method.addItem('From Points')
        method.addItem('From Binary')

        min_distance = QtWidgets.QSpinBox()
        min_distance.setRange(1, 500)
        min_distance.setValue(10)

        connectivity = ComboBox()
        connectivity.addItem('4-connected')
        connectivity.addItem('8-connected')

        self.items.append({'name': 'method', 'string': 'Marker Method', 'object': method})
        self.items.append({'name': 'min_distance', 'string': 'Min Distance', 'object': min_distance})
        self.items.append({'name': 'connectivity', 'string': 'Connectivity', 'object': connectivity})
        super().gui()

    def __call__(self, method='Auto Markers', min_distance=10,
                 connectivity='4-connected', keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Watershed segmentation does not support float16 type arrays")
            return None

        conn_val = 1 if connectivity == '4-connected' else 2
        tif = self.tif

        # --- 2D case -----------------------------------------------------------
        if tif.ndim == 2:
            labelled = self._process_frame(tif, method, min_distance, conn_val)
            if labelled is None:
                self.oldwindow.reset()
                return None
            self.newtif = labelled.astype(np.float64)

        # --- 3D (time series) case ---------------------------------------------
        elif tif.ndim == 3:
            result = np.zeros_like(tif, dtype=np.float64)
            for t in range(tif.shape[0]):
                frame = tif[t]
                # For 'From Points', use scatter points for the specific frame
                labelled = self._process_frame(
                    frame, method, min_distance, conn_val, frame_index=t)
                if labelled is not None:
                    result[t] = labelled.astype(np.float64)
            self.newtif = result

        else:
            g.alert("Watershed segmentation requires a 2D or 3D image.")
            return None

        self.newname = self.oldname + ' - Watershed'
        return self.end()

    def _process_frame(self, frame, method, min_distance, conn_val,
                       frame_index=None):
        """Segment a single 2D frame.

        Parameters
        ----------
        frame : ndarray (2D)
            The image frame to segment.
        method : str
            Marker generation method.
        min_distance : int
            Minimum peak distance (auto markers).
        conn_val : int
            Watershed connectivity (1 or 2).
        frame_index : int or None
            Time index into ``scatterPoints`` for 'From Points' mode.
            If *None* (2D image), all scatter points are used.

        Returns
        -------
        labels : ndarray (2D, int32) or None
        """
        binary_mask = frame > 0
        if not np.any(binary_mask):
            return np.zeros(frame.shape, dtype=np.int32)

        distance = scipy.ndimage.distance_transform_edt(binary_mask)

        # Determine markers --------------------------------------------------
        markers = None
        if method == 'From Points':
            scatter_pts = self._get_scatter_points(frame_index)
            if scatter_pts is not None and len(scatter_pts) > 0:
                markers = _get_markers_from_points(scatter_pts, frame.shape)
            if markers is None:
                logger.info("No scatter points found; falling back to auto markers")

        if method == 'From Binary':
            # Reserved for future enhancement -- fall back to auto
            logger.info("'From Binary' not yet implemented; using auto markers")

        if markers is None:
            markers = _get_markers_auto(binary_mask, distance, min_distance)

        if markers is None:
            logger.warning("Could not generate markers for watershed")
            return np.zeros(frame.shape, dtype=np.int32)

        return _watershed_single_frame(frame, binary_mask, markers, conn_val)

    def _get_scatter_points(self, frame_index):
        """Retrieve scatter points from the source window.

        Parameters
        ----------
        frame_index : int or None
            If *None*, return all scatter points (2D image case).
            Otherwise, return points for the given frame.

        Returns
        -------
        list or None
        """
        win = self.oldwindow
        if win is None or win.scatterPlot is None:
            return None
        if not hasattr(win, 'scatterPoints'):
            return None

        if frame_index is not None:
            if frame_index < len(win.scatterPoints):
                return win.scatterPoints[frame_index]
            return None
        else:
            # 2D image: collect all points (there is typically only frame 0)
            all_pts = []
            for pts in win.scatterPoints:
                all_pts.extend(pts)
            return all_pts if all_pts else None

    def get_init_settings_dict(self):
        s = dict()
        s['method'] = 'Auto Markers'
        s['min_distance'] = 10
        s['connectivity'] = '4-connected'
        return s

watershed_segmentation = Watershed_Segmentation()


logger.debug("Completed 'reading process/watershed.py'")
