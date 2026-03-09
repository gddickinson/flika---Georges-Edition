# -*- coding: utf-8 -*-
"""Structure detection: tubule/network analysis, line/circle detection,
corner detection, and texture analysis.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, transform, feature, draw
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from ..utils.ndim import per_plane
from ..utils.drawing import draw_line, draw_circle, draw_crosses
from ..logger import logger

logger.debug("Started 'reading process/structures.py'")

__all__ = [
    'frangi_vesselness', 'skeletonize_process', 'medial_axis_process',
    'skeleton_analysis', 'hough_lines', 'hough_circles',
    'corner_detection', 'local_binary_pattern_process',
    'structure_tensor_analysis',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Skeleton graph extraction (pure numpy/scipy fallback)
# ---------------------------------------------------------------------------

_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.uint8)
_NEIGHBOR_KERNEL[1, 1] = 0


def _classify_skeleton_pixels(skeleton):
    """Return (endpoints, branch_points) as (N,2) arrays of (y,x) coords."""
    skel = skeleton.astype(np.uint8)
    neighbor_count = ndimage.convolve(skel, _NEIGHBOR_KERNEL, mode='constant', cval=0)
    skel_mask = skel > 0
    endpoints = np.argwhere(skel_mask & (neighbor_count == 1))
    branch_points = np.argwhere(skel_mask & (neighbor_count >= 3))
    return endpoints, branch_points


def _trace_segments(skeleton, branch_points, endpoints):
    """Trace skeleton segments between junctions/endpoints.

    Returns list of dicts with 'start', 'end', 'path', 'length' keys.
    """
    skel = skeleton.astype(np.uint8).copy()
    h, w = skel.shape

    # Mark junction pixels so we can stop tracing at them
    bp_set = set(map(tuple, branch_points)) if len(branch_points) > 0 else set()
    ep_set = set(map(tuple, endpoints)) if len(endpoints) > 0 else set()
    seed_points = list(bp_set | ep_set)

    # If there are no seeds (e.g. a closed loop), pick any skeleton pixel
    if not seed_points:
        skel_pixels = np.argwhere(skel > 0)
        if len(skel_pixels) == 0:
            return []
        seed_points = [tuple(skel_pixels[0])]

    visited_edges = set()  # frozenset pairs of (start, end) to avoid duplicates
    segments = []
    visited_pixels = np.zeros_like(skel, dtype=bool)

    for seed in seed_points:
        sy, sx = seed
        # Look at all neighbors of this seed
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = sy + dy, sx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if skel[ny, nx] == 0:
                    continue
                if visited_pixels[ny, nx] and (ny, nx) not in bp_set and (ny, nx) not in ep_set:
                    continue

                # Trace from seed through (ny, nx) until we hit another
                # junction/endpoint or dead end
                path = [(sy, sx), (ny, nx)]
                cur_y, cur_x = ny, nx
                prev_y, prev_x = sy, sx

                while True:
                    if (cur_y, cur_x) in bp_set and (cur_y, cur_x) != (sy, sx):
                        break
                    if (cur_y, cur_x) in ep_set and (cur_y, cur_x) != (sy, sx):
                        break

                    # Find next unvisited neighbor (not prev)
                    found = False
                    for ddy in (-1, 0, 1):
                        for ddx in (-1, 0, 1):
                            if ddy == 0 and ddx == 0:
                                continue
                            nny, nnx = cur_y + ddy, cur_x + ddx
                            if nny < 0 or nny >= h or nnx < 0 or nnx >= w:
                                continue
                            if skel[nny, nnx] == 0:
                                continue
                            if (nny, nnx) == (prev_y, prev_x):
                                continue
                            # Found next pixel
                            path.append((nny, nnx))
                            prev_y, prev_x = cur_y, cur_x
                            cur_y, cur_x = nny, nnx
                            found = True
                            break
                        if found:
                            break

                    if not found:
                        break

                # Compute edge key to avoid duplicates
                start = path[0]
                end = path[-1]
                edge_key = frozenset([start, end, len(path)])
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                # Compute length (sum of Euclidean distances between consecutive pixels)
                path_arr = np.array(path, dtype=np.float64)
                diffs = np.diff(path_arr, axis=0)
                length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

                # Mark interior pixels as visited
                for py, px in path[1:-1]:
                    visited_pixels[py, px] = True

                segments.append({
                    'start': start,
                    'end': end,
                    'path': path_arr,
                    'length': length,
                })

    return segments


# ===================================================================
# Priority 1: Tubule / Network Analysis
# ===================================================================

@per_plane(expects_2d=True)
def _frangi_2d(image, sigmas, black_ridges):
    """Apply Frangi vesselness filter to a single 2D frame."""
    return filters.frangi(image, sigmas=sigmas, black_ridges=black_ridges)


@per_plane(expects_2d=True)
def _skeletonize_2d(image):
    """Skeletonize a single 2D binary frame."""
    return morphology.skeletonize(image > 0).astype(np.float64)


@per_plane(expects_2d=True)
def _lbp_2d(image, P, R, method):
    """Compute LBP on a single 2D frame."""
    return feature.local_binary_pattern(image, P, R, method=method)


class Frangi_Vesselness(BaseProcess):
    """frangi_vesselness(sigma_min=1.0, sigma_max=5.0, black_ridges=False, keepSourceWindow=False)

    Apply the Frangi vesselness filter to enhance tubular structures
    (blood vessels, microtubules, ER tubules, axons).

    Parameters:
        sigma_min (float): Minimum sigma for scale range
        sigma_max (float): Maximum sigma for scale range
        black_ridges (bool): Detect dark ridges on light background

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma_min = SliderLabel(2)
        sigma_min.setRange(0.5, 50)
        sigma_min.setValue(1.0)
        sigma_max = SliderLabel(2)
        sigma_max.setRange(0.5, 50)
        sigma_max.setValue(5.0)
        black_ridges = CheckBox()
        black_ridges.setChecked(False)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({'name': 'sigma_min', 'string': 'Sigma Min', 'object': sigma_min})
        self.items.append({'name': 'sigma_max', 'string': 'Sigma Max', 'object': sigma_max})
        self.items.append({'name': 'black_ridges', 'string': 'Black Ridges', 'object': black_ridges})
        self.items.append({'name': 'preview', 'string': 'Preview', 'object': preview})
        super().gui()
        self.preview()

    def __call__(self, sigma_min=1.0, sigma_max=5.0, black_ridges=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
        result = _frangi_2d(tif, sigmas, black_ridges)
        self.newtif = result.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Frangi Vesselness'
        return self.end()

    def preview(self):
        sigma_min = self.getValue('sigma_min')
        sigma_max = self.getValue('sigma_max')
        black_ridges = self.getValue('black_ridges')
        preview = self.getValue('preview')
        if preview:
            if len(g.win.image.shape) == 3:
                testimage = g.win.image[g.win.currentIndex].astype(np.float64)
            elif len(g.win.image.shape) == 2:
                testimage = g.win.image.astype(np.float64)
            sigmas = np.linspace(sigma_min, sigma_max, max(int(sigma_max - sigma_min) + 1, 2))
            testimage = filters.frangi(testimage, sigmas=sigmas, black_ridges=black_ridges)
            g.win.imageview.setImage(testimage, autoLevels=False)
        else:
            g.win.reset()

frangi_vesselness = Frangi_Vesselness()


class Skeletonize_Process(BaseProcess):
    """skeletonize_process(keepSourceWindow=False)

    Reduce binary structures to 1-pixel-wide centerlines (skeletons).
    Input should be a binary (thresholded) image.

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = _skeletonize_2d(self.tif)
        self.newname = self.oldname + ' - Skeleton'
        return self.end()

skeletonize_process = Skeletonize_Process()


class Medial_Axis_Process(BaseProcess):
    """medial_axis_process(return_distance=True, keepSourceWindow=False)

    Compute the medial axis (morphological skeleton) of a binary image.
    Optionally returns a distance map giving the local thickness (radius)
    at each skeleton pixel.

    Parameters:
        return_distance (bool): If True, output distance map; otherwise binary skeleton

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        return_distance = CheckBox()
        return_distance.setChecked(True)
        self.items.append({'name': 'return_distance', 'string': 'Return Distance Map', 'object': return_distance})
        super().gui()

    def __call__(self, return_distance=True, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif
        if tif.ndim == 3:
            result = np.zeros(tif.shape, dtype=np.float64)
            skeletons = np.zeros(tif.shape, dtype=bool)
            for i in range(len(tif)):
                skel, dist = morphology.medial_axis(tif[i] > 0, return_distance=True)
                skeletons[i] = skel
                result[i] = dist if return_distance else skel.astype(np.float64)
        elif tif.ndim == 2:
            skel, dist = morphology.medial_axis(tif > 0, return_distance=True)
            skeletons = skel
            result = dist if return_distance else skel.astype(np.float64)
        self.newtif = result.astype(np.float64)
        suffix = ' - Medial Axis (Distance)' if return_distance else ' - Medial Axis'
        self.newname = self.oldname + suffix
        newWindow = self.end()
        if newWindow is not None:
            newWindow.metadata['medial_axis_skeleton'] = skeletons
        return newWindow

medial_axis_process = Medial_Axis_Process()


class Skeleton_Analysis(BaseProcess):
    """skeleton_analysis(pixel_size=1.0, keepSourceWindow=False)

    Analyze a skeleton image to extract network topology: branch points,
    endpoints, segment lengths, junction count, and total network length.
    Input should be a binary skeleton (from Skeletonize or Medial Axis).

    Results are stored in window.metadata['skeleton_analysis'].

    Parameters:
        pixel_size (float): Physical pixel size for length measurements

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        pixel_size = SliderLabel(3)
        pixel_size.setRange(0.001, 100)
        pixel_size.setValue(1.0)
        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size', 'object': pixel_size})
        super().gui()

    def __call__(self, pixel_size=1.0, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif

        def _analyze_frame(frame_2d, frame_idx):
            skel = frame_2d > 0
            endpoints, branch_pts = _classify_skeleton_pixels(skel)
            segments = _trace_segments(skel, branch_pts, endpoints)

            # Scale lengths by pixel_size
            for seg in segments:
                seg['length'] *= pixel_size

            total_length = sum(s['length'] for s in segments)
            n_junctions = len(branch_pts)
            n_endpoints = len(endpoints)
            mean_seg_len = total_length / len(segments) if segments else 0.0
            area = float(frame_2d.shape[0] * frame_2d.shape[1]) * pixel_size ** 2
            mesh_size = np.sqrt(area / n_junctions) if n_junctions > 0 else 0.0

            return {
                'frame': frame_idx,
                'branch_points': branch_pts.tolist(),
                'endpoints': endpoints.tolist(),
                'segments': segments,
                'total_length': total_length,
                'junction_count': n_junctions,
                'endpoint_count': n_endpoints,
                'mean_segment_length': mean_seg_len,
                'mesh_size': mesh_size,
                'pixel_size': pixel_size,
            }

        if tif.ndim == 3:
            result = np.zeros(tif.shape, dtype=np.float64)
            all_analyses = []
            for i in range(len(tif)):
                analysis = _analyze_frame(tif[i], i)
                all_analyses.append(analysis)
                # Draw overlay: branch points bright, endpoints medium
                marked = tif[i].astype(np.float64).copy()
                mark_val = max(np.max(marked) * 1.5, 1.0)
                if analysis['branch_points']:
                    draw_crosses(marked, analysis['branch_points'], size=2, value=mark_val)
                if analysis['endpoints']:
                    draw_crosses(marked, analysis['endpoints'], size=1, value=mark_val * 0.7)
                result[i] = marked
        elif tif.ndim == 2:
            analysis = _analyze_frame(tif, 0)
            all_analyses = [analysis]
            marked = tif.astype(np.float64).copy()
            mark_val = max(np.max(marked) * 1.5, 1.0)
            if analysis['branch_points']:
                draw_crosses(marked, analysis['branch_points'], size=2, value=mark_val)
            if analysis['endpoints']:
                draw_crosses(marked, analysis['endpoints'], size=1, value=mark_val * 0.7)
            result = marked

        self.newtif = result
        self.newname = self.oldname + ' - Skeleton Analysis'
        newWindow = self.end()
        if newWindow is not None:
            if len(all_analyses) == 1:
                newWindow.metadata['skeleton_analysis'] = all_analyses[0]
            else:
                newWindow.metadata['skeleton_analysis'] = all_analyses
            # Log summary
            a = all_analyses[0]
            logger.info('Skeleton Analysis: %d junctions, %d endpoints, '
                        '%.1f total length, %.1f mean segment length',
                        a['junction_count'], a['endpoint_count'],
                        a['total_length'], a['mean_segment_length'])
        return newWindow

skeleton_analysis = Skeleton_Analysis()


# ===================================================================
# Priority 2: Line / Circle Detection
# ===================================================================

class Hough_Lines(BaseProcess):
    """hough_lines(threshold=10, line_length=50, line_gap=10, keepSourceWindow=False)

    Detect line segments using the Probabilistic Hough Transform.
    Input should be an edge image (e.g. from Canny edge detector).
    Detected lines are stored in window.metadata['lines'].

    Parameters:
        threshold (int): Accumulator threshold for line detection
        line_length (int): Minimum accepted length of detected lines
        line_gap (int): Maximum gap between points on the same line

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        threshold = SliderLabel(0)
        threshold.setRange(1, 500)
        threshold.setValue(10)
        line_length = SliderLabel(0)
        line_length.setRange(1, 500)
        line_length.setValue(50)
        line_gap = SliderLabel(0)
        line_gap.setRange(1, 100)
        line_gap.setValue(10)
        self.items.append({'name': 'threshold', 'string': 'Threshold', 'object': threshold})
        self.items.append({'name': 'line_length', 'string': 'Min Line Length', 'object': line_length})
        self.items.append({'name': 'line_gap', 'string': 'Max Line Gap', 'object': line_gap})
        super().gui()

    def __call__(self, threshold=10, line_length=50, line_gap=10, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        all_lines = []

        def _detect_frame(frame_2d, frame_idx):
            binary = frame_2d > 0
            lines = transform.probabilistic_hough_line(
                binary, threshold=int(threshold),
                line_length=int(line_length), line_gap=int(line_gap))
            marked = frame_2d.copy()
            mark_val = max(np.max(marked) * 1.2, 1.0)
            for (x0, y0), (x1, y1) in lines:
                draw_line(marked, y0, x0, y1, x1, mark_val)
                length = np.hypot(x1 - x0, y1 - y0)
                angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
                all_lines.append({
                    'frame': frame_idx,
                    'start': (int(y0), int(x0)),
                    'end': (int(y1), int(x1)),
                    'length': float(length),
                    'angle': float(angle),
                })
            return marked

        if tif.ndim == 3:
            result = np.zeros_like(tif)
            for i in range(len(tif)):
                result[i] = _detect_frame(tif[i], i)
        elif tif.ndim == 2:
            result = _detect_frame(tif, 0)

        self.newtif = result
        self.newname = self.oldname + ' - Hough Lines'
        newWindow = self.end()
        if newWindow is not None:
            newWindow.metadata['lines'] = all_lines
            logger.info('Hough Lines: detected %d line segments', len(all_lines))
        return newWindow

hough_lines = Hough_Lines()


class Hough_Circles(BaseProcess):
    """hough_circles(min_radius=10, max_radius=50, num_peaks=10, min_distance=20, keepSourceWindow=False)

    Detect circles using the Hough Transform.
    Input should be an edge image (e.g. from Canny edge detector).
    Detected circles are stored in window.metadata['circles'].

    Parameters:
        min_radius (int): Minimum circle radius to detect
        max_radius (int): Maximum circle radius to detect
        num_peaks (int): Maximum number of circles to detect
        min_distance (int): Minimum distance between detected circle centers

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        min_radius = SliderLabel(0)
        min_radius.setRange(1, 200)
        min_radius.setValue(10)
        max_radius = SliderLabel(0)
        max_radius.setRange(2, 500)
        max_radius.setValue(50)
        num_peaks = SliderLabel(0)
        num_peaks.setRange(1, 1000)
        num_peaks.setValue(10)
        min_distance = SliderLabel(0)
        min_distance.setRange(1, 100)
        min_distance.setValue(20)
        self.items.append({'name': 'min_radius', 'string': 'Min Radius', 'object': min_radius})
        self.items.append({'name': 'max_radius', 'string': 'Max Radius', 'object': max_radius})
        self.items.append({'name': 'num_peaks', 'string': 'Max Circles', 'object': num_peaks})
        self.items.append({'name': 'min_distance', 'string': 'Min Distance', 'object': min_distance})
        super().gui()

    def __call__(self, min_radius=10, max_radius=50, num_peaks=10, min_distance=20, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        all_circles = []
        radii = np.arange(int(min_radius), int(max_radius) + 1)
        if len(radii) == 0:
            radii = np.array([int(min_radius)])

        def _detect_frame(frame_2d, frame_idx):
            binary = (frame_2d > 0).astype(np.uint8)
            hough_res = transform.hough_circle(binary, radii)
            accums, cx, cy, found_radii = transform.hough_circle_peaks(
                hough_res, radii,
                min_xdistance=int(min_distance),
                min_ydistance=int(min_distance),
                total_num_peaks=int(num_peaks))
            marked = frame_2d.copy()
            mark_val = max(np.max(marked) * 1.2, 1.0)
            for acc, x, y, r in zip(accums, cx, cy, found_radii):
                draw_circle(marked, y, x, r, mark_val)
                all_circles.append({
                    'frame': frame_idx,
                    'center_y': int(y),
                    'center_x': int(x),
                    'radius': int(r),
                    'accumulator': float(acc),
                })
            return marked

        if tif.ndim == 3:
            result = np.zeros_like(tif)
            for i in range(len(tif)):
                result[i] = _detect_frame(tif[i], i)
        elif tif.ndim == 2:
            result = _detect_frame(tif, 0)

        self.newtif = result
        self.newname = self.oldname + ' - Hough Circles'
        newWindow = self.end()
        if newWindow is not None:
            newWindow.metadata['circles'] = all_circles
            logger.info('Hough Circles: detected %d circles', len(all_circles))
        return newWindow

hough_circles = Hough_Circles()


# ===================================================================
# Priority 3: Corner / Junction Detection
# ===================================================================

class Corner_Detection(BaseProcess):
    """corner_detection(method='Harris', min_distance=5, sigma=1.0, keepSourceWindow=False)

    Detect corners/junctions using Harris or Shi-Tomasi response.
    Detected corners are stored in window.metadata['corners'].

    Parameters:
        method (str): 'Harris' or 'Shi-Tomasi'
        min_distance (int): Minimum pixels between detected corners
        sigma (float): Gaussian smoothing sigma

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        method = ComboBox()
        method.addItem('Harris')
        method.addItem('Shi-Tomasi')
        min_distance = SliderLabel(0)
        min_distance.setRange(1, 50)
        min_distance.setValue(5)
        sigma = SliderLabel(2)
        sigma.setRange(0.5, 20)
        sigma.setValue(1.0)
        self.items.append({'name': 'method', 'string': 'Method', 'object': method})
        self.items.append({'name': 'min_distance', 'string': 'Min Distance', 'object': min_distance})
        self.items.append({'name': 'sigma', 'string': 'Sigma', 'object': sigma})
        super().gui()

    def __call__(self, method='Harris', min_distance=5, sigma=1.0, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        all_corners = []

        def _detect_frame(frame_2d, frame_idx):
            if method == 'Harris':
                response = feature.corner_harris(frame_2d, sigma=sigma)
            else:
                response = feature.corner_shi_tomasi(frame_2d, sigma=sigma)
            coords = feature.corner_peaks(response, min_distance=int(min_distance))
            marked = frame_2d.copy()
            mark_val = max(np.max(marked) * 1.2, 1.0)
            draw_crosses(marked, coords, size=2, value=mark_val)
            for c in coords:
                all_corners.append([frame_idx, int(c[0]), int(c[1])])
            return marked

        if tif.ndim == 3:
            result = np.zeros_like(tif)
            for i in range(len(tif)):
                result[i] = _detect_frame(tif[i], i)
        elif tif.ndim == 2:
            result = _detect_frame(tif, 0)

        self.newtif = result
        self.newname = self.oldname + ' - Corners ({})'.format(method)
        newWindow = self.end()
        if newWindow is not None:
            corner_array = np.array(all_corners) if all_corners else np.empty((0, 3))
            newWindow.metadata['corners'] = corner_array
            logger.info('Corner Detection (%s): found %d corners', method, len(all_corners))
        return newWindow

corner_detection = Corner_Detection()


# ===================================================================
# Priority 4: Texture / Orientation
# ===================================================================

class Local_Binary_Pattern_Process(BaseProcess):
    """local_binary_pattern_process(P=8, R=1.0, method='uniform', keepSourceWindow=False)

    Compute Local Binary Pattern texture descriptor.

    Parameters:
        P (int): Number of circularly symmetric neighbor points
        R (float): Radius of circle
        method (str): 'default', 'ror', 'uniform', or 'var'

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        P = SliderLabel(0)
        P.setRange(4, 24)
        P.setValue(8)
        R = SliderLabel(1)
        R.setRange(0.5, 10)
        R.setValue(1.0)
        method = ComboBox()
        for m in ('default', 'ror', 'uniform', 'var'):
            method.addItem(m)
        method.setCurrentIndex(2)  # 'uniform'
        self.items.append({'name': 'P', 'string': 'Neighbor Points (P)', 'object': P})
        self.items.append({'name': 'R', 'string': 'Radius (R)', 'object': R})
        self.items.append({'name': 'method', 'string': 'Method', 'object': method})
        super().gui()

    def __call__(self, P=8, R=1.0, method='uniform', keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)
        P_int = int(P)
        result = _lbp_2d(tif, P_int, R, method)
        self.newtif = result.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - LBP P={} R={}'.format(P_int, R)
        return self.end()

local_binary_pattern_process = Local_Binary_Pattern_Process()


class Structure_Tensor_Analysis(BaseProcess):
    """structure_tensor_analysis(sigma=1.0, output_type='Coherency', keepSourceWindow=False)

    Compute structure tensor to analyze local orientation and anisotropy.
    Useful for detecting oriented structures (fibers, stress fibers,
    actin filaments) and measuring their coherency and orientation.

    Results stored in window.metadata['structure_tensor'].

    Parameters:
        sigma (float): Gaussian smoothing sigma for structure tensor
        output_type (str): 'Coherency' or 'Orientation'

    Returns:
        flika.window.Window
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        sigma = SliderLabel(2)
        sigma.setRange(0.5, 20)
        sigma.setValue(1.0)
        output_type = ComboBox()
        output_type.addItem('Coherency')
        output_type.addItem('Orientation')
        self.items.append({'name': 'sigma', 'string': 'Sigma', 'object': sigma})
        self.items.append({'name': 'output_type', 'string': 'Output', 'object': output_type})
        super().gui()

    def __call__(self, sigma=1.0, output_type='Coherency', keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)

        def _compute_frame(frame_2d):
            A_elems = feature.structure_tensor(frame_2d, sigma=sigma)
            eigenvalues = feature.structure_tensor_eigenvalues(A_elems)
            # eigenvalues shape: (2, H, W), sorted descending
            l1 = eigenvalues[0]
            l2 = eigenvalues[1]
            eps = 1e-10
            coherency = (l1 - l2) / (l1 + l2 + eps)
            # Orientation from tensor elements: 0.5 * arctan2(2*Axy, Axx - Ayy)
            orientation = 0.5 * np.arctan2(2 * A_elems[1], A_elems[0] - A_elems[2])
            return coherency, orientation

        if tif.ndim == 3:
            coherency_stack = np.zeros(tif.shape, dtype=np.float64)
            orientation_stack = np.zeros(tif.shape, dtype=np.float64)
            for i in range(len(tif)):
                c, o = _compute_frame(tif[i])
                coherency_stack[i] = c
                orientation_stack[i] = o
        elif tif.ndim == 2:
            coherency_stack, orientation_stack = _compute_frame(tif)

        if output_type == 'Coherency':
            self.newtif = coherency_stack.astype(g.settings['internal_data_type'])
        else:
            self.newtif = orientation_stack.astype(g.settings['internal_data_type'])

        self.newname = self.oldname + ' - Structure Tensor ({})'.format(output_type)
        newWindow = self.end()
        if newWindow is not None:
            newWindow.metadata['structure_tensor'] = {
                'coherency': coherency_stack,
                'orientation': orientation_stack,
                'sigma': sigma,
            }
        return newWindow

structure_tensor_analysis = Structure_Tensor_Analysis()


logger.debug("Completed 'reading process/structures.py'")
