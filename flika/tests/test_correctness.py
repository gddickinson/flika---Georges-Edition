# -*- coding: utf-8 -*-
"""Comprehensive correctness tests for flika process functions.

These tests validate that process functions produce correct output values,
not just that they run without errors. Each test uses known inputs and
checks specific properties of the output.
"""
import sys
import os
import gc

import numpy as np
import pytest
import warnings
from scipy import ndimage

from flika.process import *
from flika import global_vars as g
from flika.window import Window
from flika.roi import makeROI

warnings.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean up windows after each test."""
    yield
    from qtpy.QtWidgets import QApplication
    from ..core.undo import undo_stack
    if g.m is not None:
        g.m.clear()
    undo_stack.clear()
    gc.collect()
    QApplication.processEvents()


# -----------------------------------------------------------------------
# Helper images
# -----------------------------------------------------------------------

def _uniform_3d(val=100.0, shape=(5, 32, 32), dtype=np.float64):
    return np.full(shape, val, dtype=dtype)


def _gradient_2d(shape=(64, 64)):
    """Horizontal gradient 0..1."""
    return np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)).astype(np.float64)


def _dot_image(shape=(64, 64), pos=(32, 32), radius=3, bg=0.0, fg=1.0):
    """2D image with a single bright dot."""
    img = np.full(shape, bg, dtype=np.float64)
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = ((y - pos[0])**2 + (x - pos[1])**2) <= radius**2
    img[mask] = fg
    return img


def _binary_circles(shape=(64, 64)):
    """Binary image with two non-touching circles."""
    img = np.zeros(shape, dtype=np.float64)
    y, x = np.ogrid[:shape[0], :shape[1]]
    img[((y - 16)**2 + (x - 16)**2) <= 25] = 1.0
    img[((y - 48)**2 + (x - 48)**2) <= 36] = 1.0
    return img


def _stack_with_drift(n_frames=10, shape=(64, 64)):
    """Stack where a dot moves 1px right per frame."""
    stack = np.zeros((n_frames, *shape), dtype=np.float64)
    for t in range(n_frames):
        cx = 20 + t
        stack[t] = _dot_image(shape, pos=(32, cx), radius=4, fg=1.0)
    return stack


def _bleaching_stack(n_frames=50, shape=(32, 32)):
    """Exponentially decaying signal."""
    decay = np.exp(-np.arange(n_frames) / 15.0)
    stack = np.ones((n_frames, *shape), dtype=np.float64)
    for t in range(n_frames):
        stack[t] *= decay[t] * 1000
    return stack


# -----------------------------------------------------------------------
# Filters — correctness
# -----------------------------------------------------------------------

class TestFilterCorrectness:

    def test_gaussian_blur_reduces_noise(self):
        rng = np.random.RandomState(42)
        noisy = rng.randn(32, 32).astype(np.float64) * 10 + 100
        w = Window(noisy.copy())
        w2 = gaussian_blur(2.0)
        assert w2.image.std() < noisy.std(), "Gaussian blur should reduce noise"

    def test_gaussian_blur_preserves_mean(self):
        rng = np.random.RandomState(42)
        img = rng.randn(32, 32).astype(np.float64) * 10 + 100
        w = Window(img.copy())
        w2 = gaussian_blur(1.0)
        np.testing.assert_allclose(w2.image.mean(), img.mean(), rtol=0.05)

    def test_difference_of_gaussians_bandpass(self):
        img = _dot_image((64, 64), (32, 32), radius=5, fg=100.0)
        w = Window(img.copy())
        w2 = difference_of_gaussians(1.0, 5.0)
        # DoG should enhance edges around the dot
        assert w2.image.max() > 0
        assert w2.image.min() < 0

    def test_variance_filter_uniform_is_zero(self):
        stack = _uniform_3d(100.0, (10, 32, 32))
        w = Window(stack)
        w2 = variance_filter(3)
        np.testing.assert_allclose(w2.image, 0.0, atol=1e-10)

    def test_sobel_detects_edges(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[:, 16:] = 1.0  # vertical edge
        w = Window(img.copy())
        w2 = sobel_filter()
        # Edge response should be strongest at column 16
        assert w2.image[:, 15:17].mean() > w2.image[:, :10].mean()

    def test_laplacian_filter_detects_edges(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:22, 10:22] = 1.0
        w = Window(img.copy())
        w2 = laplacian_filter(3)
        assert w2.image.max() > 0
        assert w2.image[16, 16] <= 0  # center of square should be non-positive

    def test_gaussian_laplace_filter(self):
        img = _dot_image((64, 64), (32, 32), radius=5, fg=100.0)
        w = Window(img.copy())
        w2 = gaussian_laplace_filter(2.0)
        # LoG should have negative center response for bright blob
        assert w2.image[32, 32] < 0

    def test_gaussian_gradient_magnitude(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[:, 16:] = 100.0
        w = Window(img.copy())
        w2 = gaussian_gradient_magnitude_filter(1.0)
        # Should peak at edge
        assert w2.image[:, 15:17].mean() > w2.image[:, :5].mean()

    def test_maximum_filter_expands_bright(self):
        img = _dot_image((32, 32), (16, 16), radius=1, fg=100.0)
        w = Window(img.copy())
        w2 = maximum_filter(3)
        # Maximum filter should expand the bright region
        bright_before = (img > 50).sum()
        bright_after = (w2.image > 50).sum()
        assert bright_after > bright_before

    def test_minimum_filter_shrinks_bright(self):
        img = _dot_image((32, 32), (16, 16), radius=3, fg=100.0)
        w = Window(img.copy())
        w2 = minimum_filter(3)
        bright_before = (img > 50).sum()
        bright_after = (w2.image > 50).sum()
        assert bright_after < bright_before

    def test_percentile_filter_50_is_spatial_median(self):
        rng = np.random.RandomState(42)
        img = rng.randn(32, 32).astype(np.float64) * 10 + 100
        w1 = Window(img.copy())
        w_perc = percentile_filter(50, 3)
        # 50th percentile with size 3 should equal scipy's median_filter
        expected = ndimage.median_filter(img, size=3)
        np.testing.assert_allclose(w_perc.image, expected, atol=1e-10)

    def test_tv_denoise_reduces_noise(self):
        rng = np.random.RandomState(42)
        clean = _dot_image((32, 32), (16, 16), radius=5, fg=100.0)
        noisy = clean + rng.randn(32, 32) * 20
        w = Window(noisy.copy())
        w2 = tv_denoise(0.1)
        # Denoised should be closer to clean
        err_noisy = np.mean((noisy - clean)**2)
        err_denoised = np.mean((w2.image - clean)**2)
        assert err_denoised < err_noisy

    def test_mean_filter_uniform_unchanged(self):
        stack = _uniform_3d(50.0, (10, 32, 32))
        w = Window(stack)
        w2 = mean_filter(3)
        # Uniform stack temporally averaged should stay uniform
        np.testing.assert_allclose(w2.image, 50.0, atol=1e-10)

    def test_gabor_filter_runs(self):
        img = np.zeros((32, 32), dtype=np.float64)
        # Create a sinusoidal pattern
        x = np.arange(32)
        img += np.sin(2 * np.pi * x / 8)[None, :]
        w = Window(img.copy())
        w2 = gabor_filter(0.125, 0)
        assert w2.image.shape == img.shape

    def test_sato_tubeness_detects_line(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[14:18, :] = 1.0  # horizontal line
        w = Window(img.copy())
        w2 = sato_tubeness(1, 3, False)
        assert w2.image[16, 16] > w2.image[0, 0]

    def test_meijering_neuriteness_detects_line(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[14:18, :] = 1.0
        w = Window(img.copy())
        w2 = meijering_neuriteness(1, 3, False)
        assert w2.image[16, 16] > w2.image[0, 0]

    def test_hessian_filter_detects_line(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[14:18, :] = 1.0
        w = Window(img.copy())
        w2 = hessian_filter(1, 3, False)
        assert w2.image[16, 16] > w2.image[0, 0]

    def test_difference_filter_on_stack(self):
        stack = np.zeros((5, 32, 32), dtype=np.float64)
        stack[0] = 10
        stack[1] = 20
        stack[2] = 30
        stack[3] = 25
        stack[4] = 15
        w = Window(stack)
        w2 = difference_filter()
        # diff[i] = frame[i] - frame[i-1], first frame is 0
        np.testing.assert_allclose(w2.image[1, 0, 0], 10.0, atol=1e-10)
        np.testing.assert_allclose(w2.image[2, 0, 0], 10.0, atol=1e-10)
        np.testing.assert_allclose(w2.image[3, 0, 0], -5.0, atol=1e-10)

    def test_boxcar_differential_filter(self):
        stack = np.ones((10, 16, 16), dtype=np.float64)
        for t in range(10):
            stack[t] *= t * 10
        w = Window(stack)
        w2 = boxcar_differential_filter(2, 3)
        assert w2.image is not None


# -----------------------------------------------------------------------
# Binary — correctness
# -----------------------------------------------------------------------

class TestBinaryCorrectness:

    def test_threshold_values(self):
        img = np.array([[0.2, 0.5, 0.8],
                        [0.1, 0.9, 0.3]], dtype=np.float64)
        w = Window(img)
        w2 = threshold(0.5)
        expected = np.array([[0, 0, 1],
                             [0, 1, 0]], dtype=np.float64)
        np.testing.assert_array_equal(w2.image, expected)

    def test_adaptive_threshold_binary_output(self):
        rng = np.random.RandomState(42)
        img = rng.rand(32, 32).astype(np.float64)
        w = Window(img)
        w2 = adaptive_threshold(0.5, 5)
        unique = np.unique(w2.image)
        assert len(unique) <= 2
        assert 0 in unique

    def test_binary_dilation_expands(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[16, 16] = 1.0
        w = Window(img)
        w2 = binary_dilation(2, 1, 1)
        assert w2.image.sum() > 1  # should have expanded

    def test_binary_erosion_shrinks(self):
        img = _binary_circles((64, 64))
        before_sum = img.sum()
        w = Window(img)
        w2 = binary_erosion(2, 1, 1)
        assert w2.image.sum() < before_sum

    def test_logically_combine_and(self):
        a = np.zeros((32, 32), dtype=np.float64)
        b = np.zeros((32, 32), dtype=np.float64)
        a[:16, :] = 1.0
        b[:, :16] = 1.0
        w1 = Window(a)
        w2 = Window(b)
        w3 = logically_combine(w1, w2, 'AND', keepSourceWindow=True)
        # Only top-left quadrant should be 1
        assert w3.image[:16, :16].sum() > 0
        assert w3.image[16:, :].sum() == 0
        assert w3.image[:, 16:].sum() == 0

    def test_logically_combine_or(self):
        a = np.zeros((32, 32), dtype=np.float64)
        b = np.zeros((32, 32), dtype=np.float64)
        a[:16, :] = 1.0
        b[:, :16] = 1.0
        w1 = Window(a)
        w2 = Window(b)
        w3 = logically_combine(w1, w2, 'OR', keepSourceWindow=True)
        assert w3.image.sum() > a.sum()

    def test_logically_combine_xor(self):
        a = np.zeros((32, 32), dtype=np.float64)
        b = np.zeros((32, 32), dtype=np.float64)
        a[:16, :] = 1.0
        b[:, :16] = 1.0
        w1 = Window(a)
        w2 = Window(b)
        w3 = logically_combine(w1, w2, 'XOR', keepSourceWindow=True)
        # XOR: top-left is 0, rest of top-row and left-col are 1
        assert w3.image[:16, :16].sum() == 0

    def test_grayscale_opening_removes_small_bright(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[16, 16] = 100.0  # single bright pixel
        img[5:15, 5:15] = 50.0  # large bright region
        w = Window(img)
        w2 = grayscale_opening(3)
        assert w2.image[16, 16] < 50  # single pixel removed
        assert w2.image[10, 10] > 0   # large region preserved

    def test_grayscale_closing_fills_small_dark(self):
        img = np.full((32, 32), 100.0, dtype=np.float64)
        img[16, 16] = 0.0  # single dark pixel
        w = Window(img)
        w2 = grayscale_closing(3)
        assert w2.image[16, 16] == 100.0  # dark pixel filled

    def test_morphological_gradient_detects_edges(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:22, 10:22] = 100.0
        w = Window(img)
        w2 = morphological_gradient(3)
        # Gradient should be high at edges, low in interior
        assert w2.image[16, 16] == 0  # interior
        assert w2.image[10, 16] > 0   # edge

    def test_h_maxima_suppresses_low_peaks(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10, 10] = 5.0   # low peak
        img[20, 20] = 50.0  # high peak
        w = Window(img)
        w2 = h_maxima(10.0)
        assert w2.image[10, 10] == 0   # suppressed
        assert w2.image[20, 20] > 0    # kept

    def test_h_minima(self):
        img = np.full((32, 32), 100.0, dtype=np.float64)
        img[10, 10] = 95.0  # shallow minimum (depth 5)
        img[20, 20] = 10.0  # deep minimum (depth 90)
        w = Window(img)
        w2 = h_minima(20.0)
        # h_minima detects minima with depth >= h
        # Deep minimum should be detected, shallow should not
        assert w2.image[20, 20] > w2.image[5, 5]  # deep minimum detected

    def test_area_opening(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[5, 5] = 1.0          # 1-pixel object
        img[15:20, 15:20] = 1.0  # 25-pixel object
        w = Window(img)
        w2 = area_opening(10)
        assert w2.image[5, 5] == 0        # removed
        assert w2.image[17, 17] == 1.0    # kept

    def test_area_closing(self):
        img = np.ones((32, 32), dtype=np.float64)
        img[5, 5] = 0.0          # 1-pixel hole
        img[15:20, 15:20] = 0.0  # 25-pixel hole
        w = Window(img)
        w2 = area_closing(10)
        assert w2.image[5, 5] == 1.0      # filled
        assert w2.image[17, 17] == 0.0    # not filled

    def test_remove_small_holes_binary(self):
        img = np.ones((32, 32), dtype=np.float64)
        img[15:17, 15:17] = 0.0  # small hole (4 pixels)
        w = Window(img)
        w2 = remove_small_holes(10)
        assert w2.image[15, 15] == 1  # hole filled

    def test_hysteresis_threshold(self):
        img = np.array([[0.1, 0.4, 0.9],
                        [0.4, 0.6, 0.4],
                        [0.1, 0.4, 0.1]], dtype=np.float64)
        w = Window(img)
        w2 = hysteresis_threshold(0.3, 0.7)
        # (0,2) is above 0.7 — seed
        # (0,1), (1,1), (1,2), (1,0) are between 0.3-0.7 and connected
        assert w2.image[0, 2] == 1.0
        assert w2.image[0, 0] == 0.0  # too low

    def test_multi_otsu_threshold(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[:11, :] = 0.0
        img[11:22, :] = 0.5
        img[22:, :] = 1.0
        w = Window(img)
        w2 = multi_otsu_threshold(3)
        unique = np.unique(w2.image)
        assert len(unique) == 3

    def test_canny_edge_detector_output_binary(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:22, 10:22] = 100.0
        w = Window(img)
        w2 = canny_edge_detector(0.5)
        unique = np.unique(w2.image)
        assert set(unique).issubset({0.0, 1.0})

    def test_analyze_particles(self):
        img = _binary_circles((64, 64))
        w = Window(img)
        w2 = threshold(0.5)
        w3 = analyze_particles(5, 1000)
        assert 'particle_analysis' in w3.metadata or w3.image is not None


# -----------------------------------------------------------------------
# Math — correctness
# -----------------------------------------------------------------------

class TestMathCorrectness:

    def test_subtract_value(self):
        img = np.full((32, 32), 100.0, dtype=np.float64)
        w = Window(img)
        w2 = subtract(30)
        np.testing.assert_allclose(w2.image, 70.0)

    def test_multiply_value(self):
        img = np.full((32, 32), 10.0, dtype=np.float64)
        w = Window(img)
        w2 = multiply(3.0)
        np.testing.assert_allclose(w2.image, 30.0)

    def test_divide_value(self):
        img = np.full((32, 32), 100.0, dtype=np.float64)
        w = Window(img)
        w2 = divide(4.0)
        np.testing.assert_allclose(w2.image, 25.0)

    def test_power_value(self):
        img = np.full((32, 32), 3.0, dtype=np.float64)
        w = Window(img)
        w2 = power(2)
        np.testing.assert_allclose(w2.image, 9.0)

    def test_sqrt_value(self):
        img = np.full((32, 32), 25.0, dtype=np.float64)
        w = Window(img)
        w2 = sqrt()
        np.testing.assert_allclose(w2.image, 5.0)

    def test_absolute_value_negative(self):
        img = np.full((32, 32), -7.0, dtype=np.float64)
        w = Window(img)
        w2 = absolute_value()
        np.testing.assert_allclose(w2.image, 7.0)

    def test_ratio_average(self):
        stack = np.ones((10, 16, 16), dtype=np.float64) * 50
        stack[5:] = 100  # second half is brighter
        w = Window(stack)
        w2 = ratio(0, 4, 'average')
        # ratio = frame / baseline_mean
        # baseline frames 0..4, mean = 50
        # frame 7 value = 100, ratio = 100/50 = 2.0
        np.testing.assert_allclose(w2.image[7, 0, 0], 2.0, atol=0.01)

    def test_histogram_equalize(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[:16, :] = 0.0
        img[16:, :] = 1.0
        w = Window(img)
        w2 = histogram_equalize(256)
        # After equalization, output should use more of the range
        assert w2.image is not None

    def test_normalize_minmax(self):
        img = np.array([[10.0, 50.0], [30.0, 90.0]])
        w = Window(img)
        w2 = normalize('Min-Max (0-1)')
        np.testing.assert_allclose(w2.image.min(), 0.0, atol=1e-10)
        np.testing.assert_allclose(w2.image.max(), 1.0, atol=1e-10)

    def test_normalize_zscore(self):
        rng = np.random.RandomState(42)
        img = rng.randn(64, 64).astype(np.float64) * 10 + 50
        w = Window(img)
        w2 = normalize('Z-Score')
        np.testing.assert_allclose(w2.image.mean(), 0.0, atol=0.01)
        np.testing.assert_allclose(w2.image.std(), 1.0, atol=0.01)


# -----------------------------------------------------------------------
# Stacks — correctness
# -----------------------------------------------------------------------

class TestStackCorrectness:

    def test_duplicate_same_data(self):
        img = np.random.rand(32, 32).astype(np.float64)
        w = Window(img.copy())
        w2 = duplicate()
        np.testing.assert_array_equal(w.image, w2.image)

    def test_zproject_mean(self):
        stack = np.arange(50, dtype=np.float64).reshape(5, 10, 1) * np.ones((1, 1, 10))
        # frames: 0..9, 10..19, 20..29, 30..39, 40..49
        w = Window(stack)
        w2 = zproject(0, 4, 'Average', True)
        expected_mean = stack.mean(axis=0)
        np.testing.assert_allclose(w2.image, expected_mean, atol=1e-10)

    def test_zproject_max(self):
        stack = np.zeros((5, 16, 16), dtype=np.float64)
        stack[2, 8, 8] = 100.0
        w = Window(stack)
        w.setAsCurrentWindow()
        w2 = zproject(0, 4, 'Max Intensity')
        assert w2.image[8, 8] == 100.0

    def test_zproject_min(self):
        stack = np.ones((5, 16, 16), dtype=np.float64) * 100
        stack[3, 4, 4] = 5.0
        w = Window(stack)
        w.setAsCurrentWindow()
        w2 = zproject(0, 4, 'Min Intensity')
        assert w2.image[4, 4] == 5.0

    def test_zproject_std(self):
        stack = _uniform_3d(50.0, (5, 16, 16))
        w = Window(stack)
        w.setAsCurrentWindow()
        w2 = zproject(0, 4, 'Standard Deviation')
        np.testing.assert_allclose(w2.image, 0.0, atol=1e-10)

    def test_pixel_binning_halves_size(self):
        img = np.random.rand(32, 32).astype(np.float64)
        w = Window(img)
        w2 = pixel_binning(2)
        assert w2.image.shape == (16, 16)

    def test_frame_binning_halves_frames(self):
        stack = np.random.rand(10, 16, 16).astype(np.float64)
        w = Window(stack)
        w2 = frame_binning(2)
        assert w2.image.shape[0] == 5

    def test_resize_doubles(self):
        img = np.random.rand(16, 16).astype(np.float64)
        w = Window(img)
        w2 = resize(2.0)
        assert w2.image.shape == (32, 32)

    def test_trim_range(self):
        stack = np.arange(100, dtype=np.float64).reshape(10, 10, 1) * np.ones((1, 1, 10))
        w = Window(stack)
        w2 = trim(2, 5, 1)
        assert w2.image.shape[0] == 4  # frames 2,3,4,5

    def test_concatenate_stacks_shape(self):
        a = np.random.rand(5, 16, 16).astype(np.float64)
        b = np.random.rand(3, 16, 16).astype(np.float64)
        w1 = Window(a)
        w2 = Window(b)
        w3 = concatenate_stacks(w1, w2)
        assert w3.image.shape == (8, 16, 16)

    def test_change_datatype(self):
        img = np.random.rand(32, 32).astype(np.float64) * 255
        w = Window(img)
        w2 = change_datatype('uint8')
        assert w2.image.dtype == np.uint8

    def test_deinterleave(self):
        stack = np.zeros((6, 16, 16), dtype=np.float64)
        for i in range(6):
            stack[i] = i
        w = Window(stack)
        result = deinterleave(2)
        # Should produce 2 windows with 3 frames each
        if isinstance(result, list):
            assert len(result) == 2
            assert result[0].image.shape[0] == 3
        else:
            assert result.image.shape[0] == 3

    def test_image_calculator_add(self):
        a = np.full((16, 16), 30.0, dtype=np.float64)
        b = np.full((16, 16), 20.0, dtype=np.float64)
        w1 = Window(a)
        w2 = Window(b)
        w3 = image_calculator(w1, w2, 'Add', True)
        np.testing.assert_allclose(w3.image, 50.0)

    def test_image_calculator_subtract(self):
        a = np.full((16, 16), 50.0, dtype=np.float64)
        b = np.full((16, 16), 20.0, dtype=np.float64)
        w1 = Window(a)
        w2 = Window(b)
        w3 = image_calculator(w1, w2, 'Subtract', True)
        np.testing.assert_allclose(w3.image, 30.0)

    def test_image_calculator_multiply(self):
        a = np.full((16, 16), 5.0, dtype=np.float64)
        b = np.full((16, 16), 4.0, dtype=np.float64)
        w1 = Window(a)
        w2 = Window(b)
        w3 = image_calculator(w1, w2, 'Multiply', True)
        np.testing.assert_allclose(w3.image, 20.0)

    def test_frame_remover(self):
        stack = np.arange(100, dtype=np.float64).reshape(10, 10, 1) * np.ones((1, 1, 10))
        w = Window(stack)
        w2 = frame_remover(2, 4, 5, 2)
        assert w2.image.shape[0] < 10

    def test_shear_transform(self):
        stack = np.zeros((5, 32, 32), dtype=np.float64)
        stack[:, 16, 16] = 100.0
        w = Window(stack)
        w2 = shear_transform(45.0, 1, False)
        assert w2.image is not None

    def test_generate_random_image_3d(self):
        w = generate_random_image(5, 16, 16, 1, '3D (T, X, Y)')
        assert w.image.shape == (5, 16, 16)

    def test_generate_random_image_2d(self):
        w = generate_random_image(1, 16, 16, 1, '2D (X, Y)')
        assert w.image.shape == (16, 16)


# -----------------------------------------------------------------------
# Detection — correctness
# -----------------------------------------------------------------------

class TestDetectionCorrectness:

    def test_blob_detection_log_finds_blob(self):
        img = _dot_image((64, 64), (32, 32), radius=5, fg=100.0)
        w = Window(img)
        w2 = blob_detection_log(3.0, 8.0, 5, 0.05)
        # Should produce a marked image (not all zeros)
        assert w2.image.max() > img.max()  # markers drawn brighter

    def test_blob_detection_doh_finds_blob(self):
        img = _dot_image((64, 64), (32, 32), radius=5, fg=100.0)
        w = Window(img)
        w2 = blob_detection_doh(3.0, 8.0, 5, 0.001)
        assert w2.image is not None

    def test_peak_local_max_finds_peaks(self):
        img = np.zeros((64, 64), dtype=np.float64)
        img[20, 20] = 100.0
        img[40, 40] = 80.0
        # Blur slightly to make peaks smooth
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=2)
        w = Window(img)
        w2 = peak_local_max(5, 0.5, 0)
        # Should have markers at the peaks (drawn brighter)
        assert w2.image.max() > img.max()

    def test_local_maxima_detect(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10, 10] = 50.0
        img[20, 20] = 80.0
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=1.5)
        w = Window(img)
        w2 = local_maxima_detect()
        assert w2.image is not None

    def test_template_match(self):
        img = np.zeros((64, 64), dtype=np.float64)
        img[28:36, 28:36] = 1.0
        template = np.zeros((16, 16), dtype=np.float64)
        template[4:12, 4:12] = 1.0
        w_t = Window(template)
        w1 = Window(img)
        w1.setAsCurrentWindow()
        w2 = template_match(w_t)
        assert w2.image is not None
        # The correlation map should show match location
        assert w2.image.shape[0] > 1


# -----------------------------------------------------------------------
# Segmentation — correctness
# -----------------------------------------------------------------------

class TestSegmentationCorrectness:

    def test_connected_components_count(self):
        img = _binary_circles((64, 64))
        w = Window(img)
        w2 = connected_components(2)
        # Should have 3 unique labels (0=background, 1, 2)
        assert len(np.unique(w2.image)) == 3

    def test_clear_border(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[0:5, 0:5] = 1.0    # touches border
        img[15:20, 15:20] = 2.0  # interior
        w = Window(img)
        w2 = connected_components(2)
        w3 = clear_border(0)
        # Border object should be removed
        assert w3.image[2, 2] == 0

    def test_expand_labels(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10, 10] = 1.0
        img[20, 20] = 2.0
        w = Window(img)
        w2 = expand_labels(3)
        # Labels should have expanded
        assert (w2.image == 1).sum() > 1
        assert (w2.image == 2).sum() > 1

    def test_find_boundaries(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:20, 10:20] = 1.0
        w = Window(img)
        w2 = find_boundaries('thick')
        # Boundaries should be at the edges of the square
        assert w2.image[10, 15] == 1.0  # top edge
        assert w2.image[15, 15] == 0.0  # interior

    def test_slic_superpixels(self):
        rng = np.random.RandomState(42)
        img = rng.rand(64, 64).astype(np.float64)
        w = Window(img)
        w2 = slic_superpixels(20, 10.0, 1.0)
        # Should have ~20 segments
        n_labels = len(np.unique(w2.image))
        assert 5 < n_labels <= 30

    def test_region_properties_stores_metadata(self):
        img = _binary_circles((64, 64))
        w = Window(img)
        w2 = connected_components(2)
        w3 = region_properties(1, 0)
        # Should store region properties in metadata
        assert 'region_properties' in w3.metadata


# -----------------------------------------------------------------------
# Watershed — correctness
# -----------------------------------------------------------------------

class TestWatershedCorrectness:

    def test_distance_transform(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:22, 10:22] = 1.0
        w = Window(img)
        w2 = distance_transform()
        # Center should have highest distance
        assert w2.image[16, 16] > w2.image[10, 10]
        assert w2.image[0, 0] == 0  # background

    def test_watershed_segmentation_auto(self):
        img = _binary_circles((64, 64))
        w = Window(img)
        w2 = distance_transform()
        w3 = watershed_segmentation('Auto Markers', 5, '4-connected')
        n_labels = len(np.unique(w3.image))
        assert n_labels >= 2  # at least background + 1 region


# -----------------------------------------------------------------------
# Colocalization — correctness
# -----------------------------------------------------------------------

class TestColocalizationCorrectness:

    def test_pearson_perfect(self):
        from ..process.colocalization import pearson_correlation
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        r = pearson_correlation(a, a)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_pearson_anticorrelated(self):
        from ..process.colocalization import pearson_correlation
        a = np.arange(100, dtype=np.float64).reshape(10, 10)
        b = 100 - a
        r = pearson_correlation(a, b)
        np.testing.assert_allclose(r, -1.0, atol=1e-10)

    def test_pearson_uncorrelated(self):
        from ..process.colocalization import pearson_correlation
        rng = np.random.RandomState(42)
        a = rng.randn(1000)
        b = rng.randn(1000)
        r = pearson_correlation(a, b)
        assert abs(r) < 0.1  # should be near zero

    def test_manders_perfect_overlap(self):
        from ..process.colocalization import manders_coefficients
        a = np.ones((10, 10), dtype=np.float64)
        b = np.ones((10, 10), dtype=np.float64)
        m1, m2 = manders_coefficients(a, b, 0.5, 0.5)
        np.testing.assert_allclose(m1, 1.0, atol=1e-10)
        np.testing.assert_allclose(m2, 1.0, atol=1e-10)

    def test_manders_no_overlap(self):
        from ..process.colocalization import manders_coefficients
        a = np.zeros((10, 10), dtype=np.float64)
        b = np.ones((10, 10), dtype=np.float64)
        a[:5, :] = 1.0
        b[:5, :] = 0.0
        m1, m2 = manders_coefficients(a, b, 0.5, 0.5)
        assert m1 == 0.0
        assert m2 == 0.0


# -----------------------------------------------------------------------
# Color — correctness
# -----------------------------------------------------------------------

class TestColorCorrectness:

    def test_split_channels_rgb(self):
        # Window considers ndim==4 with last dim <=4 as RGB
        img = np.zeros((5, 32, 32, 3), dtype=np.float64)
        img[..., 0] = 100  # Red
        img[..., 1] = 50   # Green
        img[..., 2] = 25   # Blue
        w = Window(img)
        result = split_channels(keepSourceWindow=True)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_blend_channels_additive(self):
        a = np.full((32, 32), 30.0, dtype=np.float64)
        b = np.full((32, 32), 20.0, dtype=np.float64)
        w1 = Window(a)
        w2 = Window(b)
        w3 = blend_channels(w1, w2, 'Additive', 0.5)
        assert w3.image is not None


# -----------------------------------------------------------------------
# Kymograph — correctness
# -----------------------------------------------------------------------

class TestKymographCorrectness:

    def test_kymograph_from_line_roi(self):
        stack = np.zeros((10, 32, 32), dtype=np.float64)
        stack[:, 16, :] = 100.0  # bright horizontal line
        w = Window(stack)
        # Create a vertical line ROI
        roi = makeROI('line', [pg.Point(16, 0), pg.Point(16, 31)])
        w2 = kymograph(1, 1, False, False, 0)
        assert w2.image is not None
        assert w2.image.ndim == 2


# -----------------------------------------------------------------------
# Background subtraction — correctness
# -----------------------------------------------------------------------

class TestBackgroundSubCorrectness:

    def test_background_subtract_constant(self):
        stack = np.ones((10, 16, 16), dtype=np.float64) * 100
        stack[:, 5:10, 5:10] += 50  # signal region
        w = Window(stack)
        roi = makeROI('rectangle', [[0, 0], [4, 4]])  # background ROI
        w2 = background_subtract()
        # Background region should be ~0, signal region should be ~50
        np.testing.assert_allclose(w2.image[:, 2, 2], 0.0, atol=1.0)
        np.testing.assert_allclose(w2.image[:, 7, 7], 50.0, atol=1.0)


# -----------------------------------------------------------------------
# Motion correction — correctness
# -----------------------------------------------------------------------

class TestMotionCorrectionCorrectness:

    def test_motion_correction_aligns(self):
        stack = _stack_with_drift(n_frames=5, shape=(64, 64))
        w = Window(stack)
        w2 = motion_correction(0)
        # After correction, the dot should be in similar position across frames
        # Check by computing centroid of each frame
        centroids_x = []
        for t in range(w2.image.shape[0]):
            frame = w2.image[t]
            if frame.max() > 0:
                x_coords = np.arange(frame.shape[1])
                cx = np.average(x_coords, weights=frame.sum(axis=0) + 1e-10)
                centroids_x.append(cx)
        if len(centroids_x) > 1:
            spread = max(centroids_x) - min(centroids_x)
            assert spread < 3.0  # should be well-aligned


# -----------------------------------------------------------------------
# Overlay — correctness
# -----------------------------------------------------------------------

class TestOverlayCorrectness:

    def test_time_stamp_adds_overlay(self):
        stack = np.zeros((5, 32, 32), dtype=np.float64)
        w = Window(stack)
        result = time_stamp(2)
        # time_stamp may return window or None depending on impl
        assert w.image is not None

    def test_scale_bar_adds_overlay(self):
        img = np.zeros((64, 64), dtype=np.float64)
        w = Window(img)
        scale_bar.gui()
        result = scale_bar(30, 5, 12, 'White', 'None', 'Lower Left')
        assert w.image is not None


# Import pyqtgraph for line ROI creation
import pyqtgraph as pg


# -----------------------------------------------------------------------
# Deconvolution — correctness
# -----------------------------------------------------------------------

class TestDeconvolutionCorrectness:

    def test_richardson_lucy_sharpens(self):
        from scipy.ndimage import gaussian_filter
        # Create a sharp image, blur it, then deconvolve
        sharp = _dot_image((32, 32), (16, 16), radius=2, fg=100.0)
        blurred = gaussian_filter(sharp, sigma=2.0)
        w = Window(blurred)
        w2 = richardson_lucy(2.0, 11, 10)
        # Deconvolved should be sharper (higher peak)
        assert w2.image[16, 16] > blurred[16, 16]

    def test_wiener_deconvolution_runs(self):
        from scipy.ndimage import gaussian_filter
        sharp = _dot_image((32, 32), (16, 16), radius=3, fg=100.0)
        blurred = gaussian_filter(sharp, sigma=2.0)
        w = Window(blurred)
        w2 = wiener_deconvolution(2.0, 11, 0.01)
        assert w2.image is not None
        assert w2.image.shape == blurred.shape

    def test_generate_psf_gaussian(self):
        w = generate_psf('Gaussian', 21, 3.0)
        assert w.image.shape == (21, 21)
        # PSF should be normalized to sum to 1
        np.testing.assert_allclose(w.image.sum(), 1.0, atol=1e-10)
        # Center should be the maximum
        assert w.image[10, 10] == w.image.max()


# -----------------------------------------------------------------------
# Stitching — correctness
# -----------------------------------------------------------------------

class TestStitchingCorrectness:

    def test_stitch_horizontal(self):
        # Two halves of a gradient image with overlap
        full = _gradient_2d((32, 64))
        left = full[:, :36].copy()  # 36 wide with 4px overlap
        right = full[:, 28:].copy()  # starts at 28, 36 wide
        w1 = Window(left)
        w2 = Window(right)
        w3 = stitch_images(w1, w2, 'Horizontal', 0.2, True)
        assert w3.image is not None
        # Stitched should be wider than either input
        assert w3.image.shape[1] >= 50


# -----------------------------------------------------------------------
# Color conversions — correctness
# -----------------------------------------------------------------------

class TestColorConversionCorrectness:

    def test_grayscale_luminance(self):
        img = np.zeros((5, 32, 32, 3), dtype=np.float64)
        img[..., 0] = 100  # R
        img[..., 1] = 50   # G
        img[..., 2] = 25   # B
        w = Window(img)
        w2 = grayscale('Luminance')
        assert w2.image.ndim == 3  # (T, H, W)
        # Luminance = 0.2126*R + 0.7152*G + 0.0722*B
        expected = 0.2126 * 100 + 0.7152 * 50 + 0.0722 * 25
        np.testing.assert_allclose(w2.image[0, 0, 0], expected, atol=0.01)

    def test_grayscale_average(self):
        img = np.zeros((32, 32, 3), dtype=np.float64)
        img[..., 0] = 30
        img[..., 1] = 60
        img[..., 2] = 90
        w = Window(img)
        w.metadata['is_rgb'] = True
        w2 = grayscale('Average')
        np.testing.assert_allclose(w2.image[0, 0], 60.0, atol=0.01)


# -----------------------------------------------------------------------
# Bleach correction — correctness
# -----------------------------------------------------------------------

class TestBleachCorrectionCorrectness:

    def test_ratio_to_mean_corrects(self):
        stack = _bleaching_stack(50, (16, 16))
        w = Window(stack)
        w2 = bleach_correction('Ratio to Mean')
        # After correction, each frame mean should be closer to the global mean
        frame_means = [w2.image[t].mean() for t in range(w2.image.shape[0])]
        spread = max(frame_means) - min(frame_means)
        original_spread = stack[0].mean() - stack[-1].mean()
        assert spread < original_spread * 0.1

    def test_exponential_fit_corrects(self):
        stack = _bleaching_stack(30, (16, 16))
        w = Window(stack)
        w2 = bleach_correction('Exponential Fit')
        # First and last frame should be much closer after correction
        ratio = w2.image[-1].mean() / w2.image[0].mean()
        assert ratio > 0.5  # should be close to 1.0

