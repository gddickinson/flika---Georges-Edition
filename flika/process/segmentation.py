# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/segmentation.py'")
import numpy as np
import scipy.ndimage
import skimage
from skimage import measure, segmentation, morphology, filters
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector
from ..utils.ndim import per_plane

__all__ = ['connected_components', 'region_properties', 'clear_border',
           'expand_labels', 'random_walker_seg', 'slic_superpixels',
           'find_boundaries', 'find_contours_process']


# ---------------------------------------------------------------------------
# Connected Components
# ---------------------------------------------------------------------------

@per_plane
def _connected_components_impl(tif, connectivity):
    if tif.ndim == 2:
        return measure.label(tif > 0, connectivity=connectivity).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.float64)
        for i in range(len(result)):
            result[i] = measure.label(tif[i] > 0, connectivity=connectivity).astype(np.float64)
        return result
    return tif


class Connected_Components(BaseProcess):
    """connected_components(connectivity=1, keepSourceWindow=False)

    Label connected components in a binary image.

    Each connected component of foreground pixels (value > 0) receives a
    unique integer label.  The number of components found is stored in the
    result window's metadata under ``'connected_component_count'``.

    Parameters:
        connectivity (int): Maximum number of orthogonal hops to consider
            a pixel a neighbour.  1 means 4-connected (2D) or 6-connected (3D);
            2 means 8-connected (2D) or 26-connected (3D).
        keepSourceWindow (bool): If False, a new Window is created with the
            result.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        connectivity = QtWidgets.QSpinBox()
        connectivity.setRange(1, 2)
        connectivity.setValue(1)
        self.items.append({'name': 'connectivity', 'string': 'Connectivity', 'object': connectivity})
        super().gui()

    def __call__(self, connectivity=1, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Connected components does not support float16 type arrays")
            return None
        self.newtif = _connected_components_impl(np.copy(self.tif), connectivity)
        self.newname = self.oldname + ' - Connected Components'
        w = self.end()
        if w is not None:
            count = int(self.newtif.max())
            w.metadata['connected_component_count'] = count
            g.status_msg(f'Found {count} connected components')
        return w

    def get_init_settings_dict(self):
        return {'connectivity': 1}

connected_components = Connected_Components()


# ---------------------------------------------------------------------------
# Region Properties
# ---------------------------------------------------------------------------

class Region_Properties(BaseProcess):
    """region_properties(min_area=1, max_area=0, intensity_window=None, keepSourceWindow=False)

    Measure properties of labelled regions in a binary or labelled image.

    Labels connected components (if the image is binary) and computes
    geometric and intensity measurements for each region.  Results are
    stored in the window metadata under ``'region_properties'`` as a list
    of dicts.

    Parameters:
        min_area (int): Minimum region area in pixels to include.
        max_area (int): Maximum region area in pixels (0 = unlimited).
        intensity_window (Window or None): Optional window whose image is
            used to compute mean_intensity for each region.
        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        min_area = SliderLabel(0)
        min_area.setRange(1, 100000)
        min_area.setValue(1)
        max_area = SliderLabel(0)
        max_area.setRange(0, 1000000)
        max_area.setValue(0)
        intensity_window = WindowSelector()
        self.items.append({'name': 'min_area', 'string': 'Min Area (pixels)', 'object': min_area})
        self.items.append({'name': 'max_area', 'string': 'Max Area (0=no limit)', 'object': max_area})
        self.items.append({'name': 'intensity_window', 'string': 'Intensity Window', 'object': intensity_window})
        super().gui()

    def __call__(self, min_area=1, max_area=0, intensity_window=None, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif

        intensity_image = None
        if intensity_window is not None and hasattr(intensity_window, 'image'):
            intensity_image = intensity_window.image

        properties = [
            'label', 'area', 'centroid', 'eccentricity', 'perimeter',
            'solidity', 'major_axis_length', 'minor_axis_length', 'orientation'
        ]
        if intensity_image is not None:
            properties.append('mean_intensity')

        if tif.ndim == 2:
            labelled, region_list = self._analyze_frame(
                tif, min_area, max_area, properties,
                intensity_image)
            self.newtif = labelled.astype(np.float64)
        elif tif.ndim == 3:
            region_list = []
            result = np.zeros_like(tif, dtype=np.float64)
            for t in range(tif.shape[0]):
                int_frame = intensity_image[t] if (intensity_image is not None and intensity_image.ndim == 3) else (
                    intensity_image if intensity_image is not None else None)
                labelled, regions = self._analyze_frame(
                    tif[t], min_area, max_area, properties, int_frame)
                result[t] = labelled.astype(np.float64)
                for r in regions:
                    r['frame'] = t
                region_list.extend(regions)
            self.newtif = result
        else:
            g.alert('Region properties requires 2D or 3D images')
            return None

        self.newname = self.oldname + ' - Region Properties'
        w = self.end()
        if w is not None:
            w.metadata['region_properties'] = region_list
            n = len(region_list)
            g.status_msg(f'Measured {n} regions')
        return w

    @staticmethod
    def _analyze_frame(binary_frame, min_area, max_area, properties, intensity_frame):
        labelled = measure.label(binary_frame > 0)
        regions = measure.regionprops(labelled, intensity_image=intensity_frame)
        region_list = []
        filtered_label = np.zeros_like(labelled)
        new_id = 1

        for props in regions:
            area = props.area
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            r = {
                'label': new_id,
                'area': int(area),
                'centroid': tuple(float(c) for c in props.centroid),
                'eccentricity': float(props.eccentricity),
                'perimeter': float(props.perimeter),
                'solidity': float(props.solidity),
                'major_axis_length': float(props.major_axis_length),
                'minor_axis_length': float(props.minor_axis_length),
                'orientation': float(props.orientation),
            }
            if intensity_frame is not None:
                r['mean_intensity'] = float(props.mean_intensity)
            filtered_label[labelled == props.label] = new_id
            region_list.append(r)
            new_id += 1

        return filtered_label, region_list

    def get_init_settings_dict(self):
        return {'min_area': 1, 'max_area': 0}

region_properties = Region_Properties()


# ---------------------------------------------------------------------------
# Clear Border
# ---------------------------------------------------------------------------

@per_plane
def _clear_border_impl(tif, buffer_size):
    if tif.ndim == 2:
        return segmentation.clear_border(tif.astype(np.intp), buffer_size=buffer_size).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.float64)
        for i in range(len(result)):
            result[i] = segmentation.clear_border(tif[i].astype(np.intp), buffer_size=buffer_size).astype(np.float64)
        return result
    return tif


class Clear_Border(BaseProcess):
    """clear_border(buffer_size=0, keepSourceWindow=False)

    Remove labelled objects that touch the image border.

    Objects whose pixels fall within *buffer_size* pixels of the image
    edge are removed (set to 0).

    Parameters:
        buffer_size (int): Width of the border region in pixels.
            Objects touching this zone are removed.
        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        buffer_size = QtWidgets.QSpinBox()
        buffer_size.setRange(0, 20)
        buffer_size.setValue(0)
        self.items.append({'name': 'buffer_size', 'string': 'Buffer Size (px)', 'object': buffer_size})
        super().gui()

    def __call__(self, buffer_size=0, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Clear border does not support float16 type arrays")
            return None
        self.newtif = _clear_border_impl(np.copy(self.tif), buffer_size)
        self.newname = self.oldname + ' - Clear Border'
        return self.end()

    def get_init_settings_dict(self):
        return {'buffer_size': 0}

clear_border = Clear_Border()


# ---------------------------------------------------------------------------
# Expand Labels
# ---------------------------------------------------------------------------

@per_plane
def _expand_labels_impl(tif, distance):
    if tif.ndim == 2:
        return segmentation.expand_labels(tif.astype(np.intp), distance=distance).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.float64)
        for i in range(len(result)):
            result[i] = segmentation.expand_labels(tif[i].astype(np.intp), distance=distance).astype(np.float64)
        return result
    return tif


class Expand_Labels(BaseProcess):
    """expand_labels(distance=1, keepSourceWindow=False)

    Grow labelled regions by *distance* pixels without overlap.

    Each labelled region expands outward into background (label 0) pixels
    up to the given distance.  Where two expanding regions would collide,
    neither claims the contested pixels, preventing overlap.

    Parameters:
        distance (int): Maximum expansion distance in pixels.
        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        distance = SliderLabel(0)
        distance.setRange(1, 100)
        distance.setValue(1)
        self.items.append({'name': 'distance', 'string': 'Distance (px)', 'object': distance})
        super().gui()

    def __call__(self, distance=1, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Expand labels does not support float16 type arrays")
            return None
        self.newtif = _expand_labels_impl(np.copy(self.tif), distance)
        self.newname = self.oldname + ' - Expanded Labels d=' + str(distance)
        return self.end()

    def get_init_settings_dict(self):
        return {'distance': 1}

expand_labels = Expand_Labels()


# ---------------------------------------------------------------------------
# Random Walker Segmentation
# ---------------------------------------------------------------------------

@per_plane
def _random_walker_impl(tif, beta, mode):
    if tif.ndim == 2:
        return segmentation.random_walker(tif, tif.astype(np.intp), beta=beta, mode=mode).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.float64)
        for i in range(len(result)):
            result[i] = segmentation.random_walker(
                tif[i], tif[i].astype(np.intp), beta=beta, mode=mode).astype(np.float64)
        return result
    return tif


class Random_Walker_Seg(BaseProcess):
    """random_walker_seg(beta=130, mode='bf', keepSourceWindow=False)

    Random walker segmentation.

    The current window must contain a labelled seed image where background
    is 0 and each seed region has a positive integer label.  The random
    walker algorithm assigns every pixel to one of the seed regions based
    on the probability of a random walker starting at that pixel reaching
    each seed.

    Parameters:
        beta (float): Penalisation coefficient for the random walker.
            A larger *beta* results in more spatially regular regions.
        mode (str): Algorithm variant: ``'bf'`` (brute force), ``'cg'``
            (conjugate gradient), or ``'cg_j'`` (CG with Jacobi
            preconditioner).
        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        beta = SliderLabel(0)
        beta.setRange(1, 10000)
        beta.setValue(130)
        mode = ComboBox()
        mode.addItem('bf')
        mode.addItem('cg')
        mode.addItem('cg_j')
        self.items.append({'name': 'beta', 'string': 'Beta', 'object': beta})
        self.items.append({'name': 'mode', 'string': 'Mode', 'object': mode})
        super().gui()

    def __call__(self, beta=130, mode='bf', keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Random walker does not support float16 type arrays")
            return None
        tif = self.tif
        labels = tif.astype(np.intp)
        if labels.max() == 0:
            g.alert("Random walker requires seed labels (non-zero regions) in the image")
            return None

        if tif.ndim == 2:
            data = tif.astype(np.float64)
            data = (data - data.min()) / (data.max() - data.min() + 1e-12)
            self.newtif = segmentation.random_walker(data, labels, beta=beta, mode=mode).astype(np.float64)
        elif tif.ndim == 3:
            result = np.zeros_like(tif, dtype=np.float64)
            for t in range(tif.shape[0]):
                frame = tif[t].astype(np.float64)
                frame_labels = labels[t]
                if frame_labels.max() == 0:
                    result[t] = frame_labels.astype(np.float64)
                    continue
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12)
                result[t] = segmentation.random_walker(
                    frame, frame_labels, beta=beta, mode=mode).astype(np.float64)
            self.newtif = result
        else:
            g.alert("Random walker requires 2D or 3D images")
            return None

        self.newname = self.oldname + ' - Random Walker'
        return self.end()

    def get_init_settings_dict(self):
        return {'beta': 130, 'mode': 'bf'}

random_walker_seg = Random_Walker_Seg()


# ---------------------------------------------------------------------------
# SLIC Superpixels
# ---------------------------------------------------------------------------

@per_plane
def _slic_impl(tif, n_segments, compactness, sigma):
    if tif.ndim == 2:
        return segmentation.slic(tif.astype(np.float64), n_segments=n_segments,
                                 compactness=compactness, sigma=sigma,
                                 start_label=1, channel_axis=None).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.float64)
        for i in range(len(result)):
            result[i] = segmentation.slic(tif[i].astype(np.float64), n_segments=n_segments,
                                          compactness=compactness, sigma=sigma,
                                          start_label=1, channel_axis=None).astype(np.float64)
        return result
    return tif


class SLIC_Superpixels(BaseProcess):
    """slic_superpixels(n_segments=100, compactness=10.0, sigma=1.0, keepSourceWindow=False)

    Segment an image into superpixels using SLIC (Simple Linear Iterative
    Clustering).

    Parameters:
        n_segments (int): Approximate number of superpixel regions to
            generate.
        compactness (float): Balance between colour proximity and spatial
            proximity.  Higher values give more weight to spatial proximity,
            producing more regularly shaped superpixels.
        sigma (float): Width of the Gaussian smoothing kernel applied
            before segmentation.  0 means no smoothing.
        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        n_segments = SliderLabel(0)
        n_segments.setRange(10, 10000)
        n_segments.setValue(100)
        compactness = SliderLabel(2)
        compactness.setRange(0.01, 100)
        compactness.setValue(10.0)
        sigma = SliderLabel(2)
        sigma.setRange(0, 10)
        sigma.setValue(1.0)
        self.items.append({'name': 'n_segments', 'string': 'N Segments', 'object': n_segments})
        self.items.append({'name': 'compactness', 'string': 'Compactness', 'object': compactness})
        self.items.append({'name': 'sigma', 'string': 'Sigma', 'object': sigma})
        super().gui()

    def __call__(self, n_segments=100, compactness=10.0, sigma=1.0, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("SLIC superpixels does not support float16 type arrays")
            return None
        self.newtif = _slic_impl(np.copy(self.tif), n_segments, compactness, sigma)
        self.newname = self.oldname + ' - SLIC n=' + str(n_segments)
        return self.end()

    def get_init_settings_dict(self):
        return {'n_segments': 100, 'compactness': 10.0, 'sigma': 1.0}

slic_superpixels = SLIC_Superpixels()


# ---------------------------------------------------------------------------
# Find Boundaries
# ---------------------------------------------------------------------------

@per_plane
def _find_boundaries_impl(tif, mode):
    if tif.ndim == 2:
        return segmentation.find_boundaries(tif.astype(np.intp), mode=mode).astype(np.float64)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.float64)
        for i in range(len(result)):
            result[i] = segmentation.find_boundaries(tif[i].astype(np.intp), mode=mode).astype(np.float64)
        return result
    return tif


class Find_Boundaries(BaseProcess):
    """find_boundaries(mode='thick', keepSourceWindow=False)

    Find boundaries between labelled regions and return a binary image.

    Parameters:
        mode (str): Type of boundary to find.

            * ``'thick'``  -- boundaries lie inside and outside the region.
            * ``'inner'``  -- boundaries lie inside the region.
            * ``'outer'``  -- boundaries lie outside the region.

        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        mode = ComboBox()
        mode.addItem('thick')
        mode.addItem('inner')
        mode.addItem('outer')
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'mode', 'string': 'Mode', 'object': mode})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()

    def __call__(self, mode='thick', keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Find boundaries does not support float16 type arrays")
            return None
        self.newtif = _find_boundaries_impl(np.copy(self.tif), mode)
        self.newname = self.oldname + ' - Boundaries (' + mode + ')'
        return self.end()

    def preview(self):
        if g.win is None or g.win.closed:
            return
        win = g.win
        mode = self.getValue('mode')
        preview = self.getValue('preview')
        if preview:
            if win.nDims == 3:
                testimage = win.image[win.currentIndex]
            elif win.nDims == 2:
                testimage = win.image
            boundaries = segmentation.find_boundaries(testimage.astype(np.intp), mode=mode).astype(np.float64)
            win.imageview.setImage(boundaries, autoLevels=False)
            win.imageview.setLevels(-0.1, 1.1)
        else:
            win.reset()

    def get_init_settings_dict(self):
        return {'mode': 'thick'}

find_boundaries = Find_Boundaries()


# ---------------------------------------------------------------------------
# Find Contours Process
# ---------------------------------------------------------------------------

class Find_Contours_Process(BaseProcess):
    """find_contours_process(level=0.5, keepSourceWindow=False)

    Find iso-valued contours in the image and create freehand ROIs from
    them.

    Uses ``skimage.measure.find_contours`` to extract contour paths at
    the specified *level*, then converts each contour into a freehand ROI
    on the current window.

    Parameters:
        level (float): The value along which to find contours in the
            image.
        keepSourceWindow (bool): If False, a new Window is created.

    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()
        self.ROIs = []

    def gui(self):
        self.gui_reset()
        level = SliderLabel(2)
        if g.win is not None:
            image = g.win.image
            level.setRange(np.min(image), np.max(image))
            level.setValue(np.mean(image))
        else:
            level.setRange(0, 1)
            level.setValue(0.5)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'level', 'string': 'Contour Level', 'object': level})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        self.ROIs = []
        super().gui()
        self.ui.rejected.connect(self.removeROIs)

    def removeROIs(self):
        for roi in self.ROIs:
            roi.cancel()
        self.ROIs = []

    def __call__(self, level=0.5, keepSourceWindow=False):
        from ..roi import makeROI
        self.start(keepSourceWindow)
        if self.tif.dtype == np.float16:
            g.alert("Find contours does not support float16 type arrays")
            return None

        for roi in self.ROIs:
            roi.cancel()
        self.ROIs = []

        if self.tif.ndim == 2:
            im = self.tif
        elif self.tif.ndim == 3:
            im = self.tif[g.win.currentIndex] if g.win is not None else self.tif[0]
        else:
            g.alert("Find contours requires 2D or 3D images")
            return None

        contours = measure.find_contours(im, level)
        for coords in contours:
            if len(coords) >= 3:
                makeROI("freehand", coords)
        self.newtif = np.copy(self.tif)
        self.newname = self.oldname + ' - Contours'
        n = len(contours)
        g.status_msg(f'Found {n} contours, created ROIs')
        return self.end()

    def preview(self):
        from ..roi import ROI_Drawing
        if g.win is None or g.win.closed:
            return
        win = g.win
        level = self.getValue('level')
        preview = self.getValue('preview')

        for roi in self.ROIs:
            roi.cancel()
        self.ROIs = []

        if not preview:
            return

        if win.nDims == 3:
            im = win.image[win.currentIndex]
        elif win.nDims == 2:
            im = win.image
        else:
            return

        contours = measure.find_contours(im, level)
        for coords in contours:
            if len(coords) >= 3:
                self.ROIs.append(ROI_Drawing(win, coords[0][0], coords[0][1], 'freehand'))
                for p in coords[1:]:
                    self.ROIs[-1].extend(p[0], p[1])
                    QtWidgets.QApplication.processEvents()

    def get_init_settings_dict(self):
        return {'level': 0.5}

find_contours_process = Find_Contours_Process()


logger.debug("Completed 'reading process/segmentation.py'")
