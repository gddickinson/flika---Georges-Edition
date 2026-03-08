# -*- coding: utf-8 -*-
"""Tests for process/structures.py — structure detection module.

Standalone tests (no Qt app needed) for pure functions,
plus Window-based integration tests using the ``fa`` fixture.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: synthetic images
# ---------------------------------------------------------------------------

def _make_tube_image(shape=(100, 100), center_y=50, width=3, intensity=200):
    """Horizontal bright tube on dark background."""
    img = np.zeros(shape, dtype=np.float64)
    half = width // 2
    for dy in range(-half, half + 1):
        y = center_y + dy
        if 0 <= y < shape[0]:
            img[y, :] = intensity * np.exp(-0.5 * (dy / max(width / 4, 0.5)) ** 2)
    return img


def _make_binary_rect(shape=(100, 100), top=20, left=20, bottom=80, right=80):
    """Binary image with a filled rectangle."""
    img = np.zeros(shape, dtype=np.float64)
    img[top:bottom, left:right] = 1.0
    return img


def _make_T_skeleton():
    """Binary T-shaped skeleton: 1 branch point, 3 endpoints."""
    img = np.zeros((50, 50), dtype=np.float64)
    # Vertical bar
    img[10:40, 25] = 1.0
    # Horizontal bar at y=10
    img[10, 10:40] = 1.0
    return img


def _make_circle_edge(shape=(100, 100), cy=50, cx=50, radius=20):
    """Binary circle perimeter."""
    from skimage.draw import circle_perimeter
    img = np.zeros(shape, dtype=np.float64)
    rr, cc = circle_perimeter(cy, cx, radius, shape=shape)
    img[rr, cc] = 1.0
    return img


def _make_line_image(shape=(100, 100), y0=50, x0=10, y1=50, x1=90):
    """Binary image with a single line."""
    from skimage.draw import line
    img = np.zeros(shape, dtype=np.float64)
    rr, cc = line(y0, x0, y1, x1)
    img[rr, cc] = 1.0
    return img


def _make_corner_image(shape=(100, 100)):
    """Binary L-shape with a sharp corner at (50, 50)."""
    img = np.zeros(shape, dtype=np.float64)
    # Horizontal arm
    img[50, 50:90] = 255.0
    # Vertical arm
    img[10:50, 50] = 255.0
    return img


def _make_striped_image(shape=(100, 100), period=10):
    """Horizontal stripes for structure tensor testing."""
    img = np.zeros(shape, dtype=np.float64)
    for y in range(shape[0]):
        if (y // period) % 2 == 0:
            img[y, :] = 255.0
    return img


# ===========================================================================
# Pure function / standalone tests
# ===========================================================================

class TestClassifySkeletonPixels:
    def test_T_shape(self):
        from ..process.structures import _classify_skeleton_pixels
        skel = _make_T_skeleton()
        endpoints, branch_pts = _classify_skeleton_pixels(skel)
        # T has 3 endpoints and 1 branch point
        assert len(endpoints) == 3
        assert len(branch_pts) >= 1

    def test_empty_skeleton(self):
        from ..process.structures import _classify_skeleton_pixels
        skel = np.zeros((20, 20), dtype=np.float64)
        endpoints, branch_pts = _classify_skeleton_pixels(skel)
        assert len(endpoints) == 0
        assert len(branch_pts) == 0

    def test_single_line(self):
        from ..process.structures import _classify_skeleton_pixels
        skel = np.zeros((20, 20), dtype=np.float64)
        skel[10, 5:15] = 1.0  # horizontal line, 10 pixels
        endpoints, branch_pts = _classify_skeleton_pixels(skel)
        assert len(endpoints) == 2
        assert len(branch_pts) == 0


class TestTraceSegments:
    def test_single_line_one_segment(self):
        from ..process.structures import _classify_skeleton_pixels, _trace_segments
        skel = np.zeros((20, 20), dtype=np.float64)
        skel[10, 5:15] = 1.0
        endpoints, branch_pts = _classify_skeleton_pixels(skel)
        segments = _trace_segments(skel, branch_pts, endpoints)
        assert len(segments) >= 1
        total = sum(s['length'] for s in segments)
        assert total > 0

    def test_T_shape_multiple_segments(self):
        from ..process.structures import _classify_skeleton_pixels, _trace_segments
        skel = _make_T_skeleton()
        endpoints, branch_pts = _classify_skeleton_pixels(skel)
        segments = _trace_segments(skel, branch_pts, endpoints)
        # T-shape should yield at least 3 segments (junction pixels may split)
        assert len(segments) >= 3
        total = sum(s['length'] for s in segments)
        assert total > 0


# ===========================================================================
# BaseProcess integration tests (require Qt app via ``fa`` fixture)
# ===========================================================================

class TestFrangiVesselness:
    def test_enhances_tube(self, fa):
        from ..window import Window
        img = _make_tube_image()
        w = Window(img)
        from ..process.structures import frangi_vesselness
        result_w = frangi_vesselness(sigma_min=1.0, sigma_max=3.0,
                                     keepSourceWindow=True)
        result = result_w.image
        # Frangi response should be higher along the tube center
        tube_val = result[50, 50] if result.ndim == 2 else result[0, 50, 50]
        bg_val = result[5, 5] if result.ndim == 2 else result[0, 5, 5]
        assert tube_val > bg_val
        result_w.close()
        w.close()

    def test_2d_output_shape(self, fa):
        from ..window import Window
        img = np.random.rand(64, 64).astype(np.float64)
        w = Window(img)
        from ..process.structures import frangi_vesselness
        result_w = frangi_vesselness(sigma_min=1.0, sigma_max=3.0,
                                     keepSourceWindow=True)
        # Output shape should match input
        assert result_w.image.shape[-2:] == (64, 64)
        result_w.close()
        w.close()

    def test_3d_stack(self, fa):
        from ..window import Window
        img = np.random.rand(5, 64, 64).astype(np.float64)
        w = Window(img)
        from ..process.structures import frangi_vesselness
        result_w = frangi_vesselness(sigma_min=1.0, sigma_max=3.0,
                                     keepSourceWindow=True)
        assert result_w.image.shape == (5, 64, 64)
        result_w.close()
        w.close()


class TestSkeletonize:
    def test_thins_binary(self, fa):
        from ..window import Window
        img = _make_binary_rect()
        w = Window(img)
        from ..process.structures import skeletonize_process
        result_w = skeletonize_process(keepSourceWindow=True)
        result = result_w.image if result_w.image.ndim == 2 else result_w.image[0]
        # Skeleton should have fewer nonzero pixels than filled rect
        assert np.sum(result > 0) < np.sum(img > 0)
        assert np.sum(result > 0) > 0
        result_w.close()
        w.close()

    def test_skeleton_subset_of_input(self, fa):
        from ..window import Window
        img = _make_binary_rect()
        w = Window(img)
        from ..process.structures import skeletonize_process
        result_w = skeletonize_process(keepSourceWindow=True)
        result = result_w.image if result_w.image.ndim == 2 else result_w.image[0]
        # All skeleton pixels should be within the original shape
        skel_coords = np.argwhere(result > 0)
        for y, x in skel_coords:
            assert img[y, x] > 0
        result_w.close()
        w.close()


class TestMedialAxis:
    def test_distance_map(self, fa):
        from ..window import Window
        # Binary circle
        img = np.zeros((100, 100), dtype=np.float64)
        yy, xx = np.ogrid[:100, :100]
        img[((yy - 50) ** 2 + (xx - 50) ** 2) < 30 ** 2] = 1.0
        w = Window(img)
        from ..process.structures import medial_axis_process
        result_w = medial_axis_process(return_distance=True, keepSourceWindow=True)
        result = result_w.image if result_w.image.ndim == 2 else result_w.image[0]
        # Distance should be highest near center
        assert result[50, 50] > result[35, 50]
        assert 'medial_axis_skeleton' in result_w.metadata
        result_w.close()
        w.close()

    def test_binary_output(self, fa):
        from ..window import Window
        img = _make_binary_rect()
        w = Window(img)
        from ..process.structures import medial_axis_process
        result_w = medial_axis_process(return_distance=False, keepSourceWindow=True)
        result = result_w.image if result_w.image.ndim == 2 else result_w.image[0]
        # Binary: values should be 0 or 1
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})
        result_w.close()
        w.close()


class TestSkeletonAnalysis:
    def test_T_shape_metadata(self, fa):
        from ..window import Window
        skel = _make_T_skeleton()
        w = Window(skel)
        from ..process.structures import skeleton_analysis
        result_w = skeleton_analysis(pixel_size=1.0, keepSourceWindow=True)
        assert 'skeleton_analysis' in result_w.metadata
        sa = result_w.metadata['skeleton_analysis']
        assert sa['junction_count'] >= 1
        assert sa['endpoint_count'] == 3
        assert sa['total_length'] > 0
        assert len(sa['segments']) >= 3
        result_w.close()
        w.close()

    def test_pixel_size_scaling(self, fa):
        from ..window import Window
        skel = _make_T_skeleton()
        w = Window(skel)
        from ..process.structures import skeleton_analysis
        result1 = skeleton_analysis(pixel_size=1.0, keepSourceWindow=True)
        len1 = result1.metadata['skeleton_analysis']['total_length']
        result1.close()

        w2 = Window(skel)
        result2 = skeleton_analysis(pixel_size=2.0, keepSourceWindow=True)
        len2 = result2.metadata['skeleton_analysis']['total_length']
        result2.close()
        w2.close()
        w.close()

        assert abs(len2 - 2.0 * len1) < 1e-6

    def test_empty_skeleton(self, fa):
        from ..window import Window
        img = np.zeros((30, 30), dtype=np.float64)
        w = Window(img)
        from ..process.structures import skeleton_analysis
        result_w = skeleton_analysis(pixel_size=1.0, keepSourceWindow=True)
        sa = result_w.metadata['skeleton_analysis']
        assert sa['junction_count'] == 0
        assert sa['endpoint_count'] == 0
        assert sa['total_length'] == 0
        result_w.close()
        w.close()


class TestHoughLines:
    def test_detects_horizontal_line(self, fa):
        from ..window import Window
        img = _make_line_image(y0=50, x0=10, y1=50, x1=90)
        w = Window(img)
        from ..process.structures import hough_lines
        result_w = hough_lines(threshold=5, line_length=20, line_gap=5,
                               keepSourceWindow=True)
        assert 'lines' in result_w.metadata
        lines = result_w.metadata['lines']
        assert len(lines) >= 1
        # At least one line should be roughly horizontal (angle near 0 or ±180)
        for l in lines:
            a = abs(l['angle'])
            if a > 90:
                a = 180 - a
            if a < 15:
                break
        else:
            pytest.fail('No horizontal line detected')
        result_w.close()
        w.close()

    def test_no_lines_in_blank(self, fa):
        from ..window import Window
        img = np.zeros((50, 50), dtype=np.float64)
        w = Window(img)
        from ..process.structures import hough_lines
        result_w = hough_lines(threshold=10, line_length=20, line_gap=5,
                               keepSourceWindow=True)
        assert result_w.metadata['lines'] == []
        result_w.close()
        w.close()


class TestHoughCircles:
    def test_detects_circle(self, fa):
        from ..window import Window
        img = _make_circle_edge(cy=50, cx=50, radius=20)
        w = Window(img)
        from ..process.structures import hough_circles
        result_w = hough_circles(min_radius=15, max_radius=25, num_peaks=5,
                                 min_distance=10, keepSourceWindow=True)
        assert 'circles' in result_w.metadata
        circles = result_w.metadata['circles']
        assert len(circles) >= 1
        # Best circle should be near (50, 50) with radius ~20
        c = circles[0]
        assert abs(c['center_y'] - 50) < 5
        assert abs(c['center_x'] - 50) < 5
        assert abs(c['radius'] - 20) < 5
        result_w.close()
        w.close()


class TestCornerDetection:
    def test_harris_detects_corner(self, fa):
        from ..window import Window
        img = _make_corner_image()
        w = Window(img)
        from ..process.structures import corner_detection
        result_w = corner_detection(method='Harris', min_distance=5, sigma=1.0,
                                    keepSourceWindow=True)
        assert 'corners' in result_w.metadata
        corners = result_w.metadata['corners']
        assert len(corners) >= 1
        result_w.close()
        w.close()

    def test_shi_tomasi(self, fa):
        from ..window import Window
        img = _make_corner_image()
        w = Window(img)
        from ..process.structures import corner_detection
        result_w = corner_detection(method='Shi-Tomasi', min_distance=5, sigma=1.0,
                                    keepSourceWindow=True)
        assert 'corners' in result_w.metadata
        assert len(result_w.metadata['corners']) >= 1
        result_w.close()
        w.close()


class TestLocalBinaryPattern:
    def test_output_shape(self, fa):
        from ..window import Window
        img = np.random.rand(64, 64).astype(np.float64) * 255
        w = Window(img)
        from ..process.structures import local_binary_pattern_process
        result_w = local_binary_pattern_process(P=8, R=1.0, method='uniform',
                                                keepSourceWindow=True)
        assert result_w.image.shape[-2:] == (64, 64)
        result_w.close()
        w.close()

    def test_nonnegative(self, fa):
        from ..window import Window
        img = np.random.rand(64, 64).astype(np.float64) * 255
        w = Window(img)
        from ..process.structures import local_binary_pattern_process
        result_w = local_binary_pattern_process(P=8, R=1.0, method='uniform',
                                                keepSourceWindow=True)
        result = result_w.image if result_w.image.ndim == 2 else result_w.image[0]
        assert np.all(result >= 0)
        result_w.close()
        w.close()

    def test_3d_stack(self, fa):
        from ..window import Window
        img = np.random.rand(3, 32, 32).astype(np.float64) * 255
        w = Window(img)
        from ..process.structures import local_binary_pattern_process
        result_w = local_binary_pattern_process(P=8, R=1.0, method='default',
                                                keepSourceWindow=True)
        assert result_w.image.shape == (3, 32, 32)
        result_w.close()
        w.close()


class TestStructureTensor:
    def test_horizontal_stripes_orientation(self, fa):
        from ..window import Window
        img = _make_striped_image(period=10)
        w = Window(img)
        from ..process.structures import structure_tensor_analysis
        result_w = structure_tensor_analysis(sigma=2.0, output_type='Orientation',
                                             keepSourceWindow=True)
        assert 'structure_tensor' in result_w.metadata
        st = result_w.metadata['structure_tensor']
        assert 'coherency' in st
        assert 'orientation' in st
        result_w.close()
        w.close()

    def test_coherency_output(self, fa):
        from ..window import Window
        img = _make_striped_image(period=10)
        w = Window(img)
        from ..process.structures import structure_tensor_analysis
        result_w = structure_tensor_analysis(sigma=2.0, output_type='Coherency',
                                             keepSourceWindow=True)
        result = result_w.image if result_w.image.ndim == 2 else result_w.image[0]
        # Coherency should be high in striped regions (between 0 and 1)
        center_val = result[25, 50]
        assert 0 <= center_val <= 1.0
        result_w.close()
        w.close()

    def test_3d_stack(self, fa):
        from ..window import Window
        img = np.random.rand(3, 32, 32).astype(np.float64) * 255
        w = Window(img)
        from ..process.structures import structure_tensor_analysis
        result_w = structure_tensor_analysis(sigma=1.0, output_type='Coherency',
                                             keepSourceWindow=True)
        assert result_w.image.shape == (3, 32, 32)
        result_w.close()
        w.close()
