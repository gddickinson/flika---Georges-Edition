"""Tests for acceleration utilities and vectorized filter correctness."""
import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt


# ---- Device detection tests ----

def test_detect_devices_returns_info():
    from flika.utils.accel import detect_devices, AccelerationInfo
    info = detect_devices(force_refresh=True)
    assert isinstance(info, AccelerationInfo)
    assert len(info.devices) >= 1  # at least CPU
    assert info.devices[0].name == 'CPU'
    assert info.devices[0].usable is True
    assert info.best_device  # not empty


def test_detect_devices_caching():
    from flika.utils.accel import detect_devices
    info1 = detect_devices(force_refresh=True)
    info2 = detect_devices()
    assert info1 is info2
    info3 = detect_devices(force_refresh=True)
    assert info3 is not info1


def test_status_report_format():
    from flika.utils.accel import detect_devices
    info = detect_devices(force_refresh=True)
    report = info.status_report()
    assert 'CPU' in report
    assert 'Recommended device:' in report


def test_get_torch_device_cpu():
    from flika.utils.accel import get_torch_device
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    device = get_torch_device('CPU')
    assert str(device) == 'cpu'


def test_should_use_gpu_small_array():
    from flika.utils.accel import should_use_gpu
    small = np.zeros((10, 10))
    assert should_use_gpu(small) is False


# ---- Vectorized filter correctness tests ----

def test_variance_filter_vectorized_correct():
    """Verify vectorized variance filter computes correct rolling variance."""
    np.random.seed(42)
    tif = np.random.randn(50, 4, 4).astype(np.float64)
    nFrames = 7

    # Vectorized: Var(X) = E[X²] - E[X]²
    tif_f = tif.astype(np.float64)
    mean = uniform_filter1d(tif_f, size=nFrames, axis=0, mode='nearest')
    mean_sq = uniform_filter1d(tif_f ** 2, size=nFrames, axis=0, mode='nearest')
    result = np.maximum(mean_sq - mean ** 2, 0)

    # Verify by manually computing variance for a few interior pixels
    half = nFrames // 2
    for t in range(half, tif.shape[0] - half):
        for x in range(2):
            for y in range(2):
                window = tif[t - half:t + half + 1, x, y]
                expected_var = np.var(window)
                np.testing.assert_allclose(result[t, x, y], expected_var, atol=1e-10,
                    err_msg=f"Mismatch at t={t}, x={x}, y={y}")


def test_median_filter_vectorized():
    """Verify vectorized median filter computes correct rolling median."""
    from scipy.ndimage import median_filter as nd_median_filter

    np.random.seed(42)
    tif = np.random.randn(30, 4, 4).astype(np.float64)
    nFrames = 5

    # Vectorized result
    result = nd_median_filter(tif, size=(nFrames, 1, 1))

    # Verify interior values manually (where full window is available)
    half = nFrames // 2
    for t in range(half, tif.shape[0] - half):
        for x in range(2):
            for y in range(2):
                window = tif[t - half:t + half + 1, x, y]
                expected_med = np.median(window)
                np.testing.assert_allclose(result[t, x, y], expected_med, atol=1e-10,
                    err_msg=f"Mismatch at t={t}, x={x}, y={y}")


def test_fourier_filter_vectorized():
    """Verify batch FFT matches per-pixel FFT."""
    from scipy.fft import fft, ifft, fftfreq

    np.random.seed(42)
    tif = np.random.randn(64, 4, 4).astype(np.float64)
    frame_rate = 100.0
    low = 5.0
    high = 40.0

    mt = tif.shape[0]
    W = fftfreq(mt, d=1.0 / frame_rate)
    filt = np.ones(mt)
    filt[np.abs(W) < low] = 0
    filt[np.abs(W) > high] = 0

    # Per-pixel (original)
    expected = np.zeros(tif.shape)
    for i in range(tif.shape[2]):
        for j in range(tif.shape[1]):
            f_signal = fft(tif[:, j, i])
            f_signal *= filt
            expected[:, j, i] = np.real(ifft(f_signal))

    # Vectorized batch
    filt_3d = filt[:, np.newaxis, np.newaxis]
    f_signal = fft(tif, axis=0)
    result = np.real(ifft(f_signal * filt_3d, axis=0))

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_butterworth_vectorized():
    """Verify axis-based filtfilt matches per-pixel filtfilt."""
    from scipy.signal import butter, filtfilt

    np.random.seed(42)
    tif = np.random.randn(100, 4, 4).astype(np.float64)
    [b, a] = butter(2, 0.3, btype='lowpass')
    padlen = 6

    # Per-pixel (original)
    expected = np.zeros(tif.shape)
    for i in range(tif.shape[1]):
        for j in range(tif.shape[2]):
            expected[:, i, j] = filtfilt(b, a, tif[:, i, j], padlen=padlen)

    # Vectorized
    result = filtfilt(b, a, tif, axis=0, padlen=padlen)

    np.testing.assert_allclose(result, expected, atol=1e-10)


# ---- Numba JIT tests ----

def test_clustering_numba_matches_pure():
    """Verify numba-accelerated clustering matches pure Python."""
    from flika.utils.clustering import (
        local_density, _higher_density_distance_pure,
        _higher_density_distance_jit, higher_density_distance
    )

    np.random.seed(42)
    points = np.random.randn(50, 2).astype(np.float64)
    rho = local_density(points, radius=2.0)

    # Get result from the public API (uses numba if available)
    delta, nearest = higher_density_distance(points, rho)

    # Also run pure Python
    order = np.argsort(-rho)
    delta_pure, nearest_pure = _higher_density_distance_pure(points, rho, order, len(points))

    # Fix the top point for comparison
    top = order[0]
    delta_pure[top] = np.max(delta_pure[delta_pure < np.inf]) if len(points) > 1 else 0.0
    nearest_pure[top] = top

    np.testing.assert_allclose(delta, delta_pure, atol=1e-10)
    np.testing.assert_array_equal(nearest, nearest_pure)


def test_tracking_iterative_no_overflow():
    """Verify iterative _extend_track works on long sequences (no stack overflow)."""
    from flika.utils.tracking import link_particles

    # Create a simple linear track: particle moves +1 in x each frame
    n_frames = 500
    locs = np.column_stack([
        np.arange(n_frames, dtype=float),  # frame
        np.arange(n_frames, dtype=float),  # x (moves +1 per frame)
        np.zeros(n_frames),               # y
    ])

    tracks = link_particles(locs, max_distance=2.0, max_gap=1)
    # Should produce a single track with all points
    assert len(tracks) == 1
    assert len(tracks[0]) == n_frames


# ---- per_plane parallel tests ----

def test_per_plane_parallel():
    """Verify parallel per_plane gives same results as sequential."""
    from flika.utils.ndim import per_plane

    @per_plane(parallel=True)
    def test_filter(img):
        return img * 2.0

    @per_plane
    def test_filter_seq(img):
        return img * 2.0

    np.random.seed(42)
    data_4d = np.random.randn(10, 8, 8, 3).astype(np.float64)

    result_par = test_filter(data_4d)
    result_seq = test_filter_seq(data_4d)

    np.testing.assert_allclose(result_par, result_seq)
