"""Tests for SPT detection modules."""
import pytest
import numpy as np


class TestUTrackDetector:
    def test_detect_frame_synthetic(self):
        """Detect known bright spots on dark background."""
        from flika.spt.detection.utrack_detector import UTrackDetector

        frame = np.zeros((100, 100), dtype=np.float64)
        # Place 3 known spots at (25,25), (50,50), (75,75) with Gaussian profile
        for cx, cy in [(25, 25), (50, 50), (75, 75)]:
            yy, xx = np.mgrid[-5:6, -5:6]
            spot = 1000 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
            frame[cy - 5:cy + 6, cx - 5:cx + 6] += spot
        frame += np.random.normal(100, 10, frame.shape)  # background

        det = UTrackDetector(psf_sigma=1.5, alpha=0.05)
        locs = det.detect_frame(frame)
        assert locs.ndim == 2
        assert locs.shape[1] == 3  # x, y, intensity
        assert len(locs) >= 2  # should find at least 2 of 3 spots

    def test_detect_stack(self):
        """Detect in a 3-frame stack."""
        from flika.spt.detection.utrack_detector import UTrackDetector

        stack = np.random.normal(100, 10, (3, 50, 50))
        for f in range(3):
            stack[f, 25, 25] += 500
        det = UTrackDetector(psf_sigma=1.5, alpha=0.1)
        locs = det.detect_stack(stack)
        assert locs.ndim == 2
        assert locs.shape[1] == 4  # frame, x, y, intensity

    def test_detect_empty_frame(self):
        """Detection on a uniform noise frame should return a 2D array."""
        from flika.spt.detection.utrack_detector import UTrackDetector

        frame = np.random.normal(100, 10, (50, 50))
        det = UTrackDetector(psf_sigma=1.5, alpha=0.001)  # strict
        locs = det.detect_frame(frame)
        assert locs.ndim == 2
        assert locs.shape[1] == 3

    def test_detect_frame_returns_empty_on_constant(self):
        """A constant frame should yield zero detections."""
        from flika.spt.detection.utrack_detector import UTrackDetector

        frame = np.full((50, 50), 100.0)
        det = UTrackDetector(psf_sigma=1.5, alpha=0.05)
        locs = det.detect_frame(frame)
        assert locs.ndim == 2
        assert len(locs) == 0

    def test_subpixel_accuracy(self):
        """Sub-pixel refinement should place detections near known centres."""
        from flika.spt.detection.utrack_detector import UTrackDetector

        frame = np.zeros((60, 60), dtype=np.float64)
        # Single bright spot at centre
        cx, cy = 30, 30
        yy, xx = np.mgrid[-5:6, -5:6]
        spot = 2000 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
        frame[cy - 5:cy + 6, cx - 5:cx + 6] += spot
        frame += np.random.normal(50, 5, frame.shape)

        det = UTrackDetector(psf_sigma=1.5, alpha=0.05)
        locs = det.detect_frame(frame)
        assert len(locs) >= 1
        # Check the closest detection is within 3 pixels of known centre
        distances = np.sqrt((locs[:, 0] - cx)**2 + (locs[:, 1] - cy)**2)
        assert np.min(distances) < 3.0


class TestThunderSTORMDetector:
    def test_detect_frame(self):
        """Detect bright spots with ThunderSTORM pipeline."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        frame = np.zeros((100, 100), dtype=np.float64)
        for cx, cy in [(30, 30), (60, 60)]:
            yy, xx = np.mgrid[-5:6, -5:6]
            spot = 800 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
            frame[cy - 5:cy + 6, cx - 5:cx + 6] += spot
        frame += np.random.normal(100, 10, frame.shape)

        det = ThunderSTORMDetector(filter_type='gaussian', detector_type='local_max',
                                   threshold=2.0)
        locs = det.detect_frame(frame)
        assert locs.ndim == 2
        if len(locs) > 0:
            assert locs.shape[1] == 7  # x, y, intensity, sigma_x, sigma_y, background, uncertainty

    def test_detect_stack(self):
        """Detect in a multi-frame stack."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        stack = np.random.normal(100, 10, (3, 50, 50))
        stack[:, 25, 25] += 500
        det = ThunderSTORMDetector(threshold=2.0)
        locs = det.detect_stack(stack)
        assert locs.ndim == 2
        if len(locs) > 0:
            assert locs.shape[1] == 8  # frame, x, y, intensity, sigma_x, sigma_y, background, uncertainty

    def test_detect_frame_empty_returns_correct_shape(self):
        """Empty detection result should have shape (0, 7)."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        frame = np.full((20, 20), 100.0, dtype=np.float64)
        det = ThunderSTORMDetector(threshold=100.0)  # extremely strict
        locs = det.detect_frame(frame)
        assert locs.shape == (0, 7)

    def test_detect_stack_empty_returns_correct_shape(self):
        """Empty stack detection result should have shape (0, 8)."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        stack = np.full((2, 20, 20), 100.0, dtype=np.float64)
        det = ThunderSTORMDetector(threshold=100.0)
        locs = det.detect_stack(stack)
        assert locs.shape == (0, 8)

    def test_invalid_filter_type_raises(self):
        """Invalid filter_type should raise ValueError."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        with pytest.raises(ValueError, match="Unknown filter"):
            ThunderSTORMDetector(filter_type='invalid')

    def test_invalid_detector_type_raises(self):
        """Invalid detector_type should raise ValueError."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        with pytest.raises(ValueError, match="Unknown detector"):
            ThunderSTORMDetector(detector_type='invalid')

    def test_invalid_fitter_type_raises(self):
        """Invalid fitter_type should raise ValueError."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        with pytest.raises(ValueError, match="Unknown fitter_type"):
            ThunderSTORMDetector(fitter_type='invalid')

    def test_wavelet_filter(self):
        """Detection with wavelet filter should work."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        frame = np.zeros((80, 80), dtype=np.float64)
        yy, xx = np.mgrid[-5:6, -5:6]
        spot = 1000 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
        frame[40 - 5:40 + 6, 40 - 5:40 + 6] += spot
        frame += np.random.normal(100, 10, frame.shape)

        det = ThunderSTORMDetector(filter_type='wavelet', threshold=1.5)
        locs = det.detect_frame(frame)
        assert locs.ndim == 2

    def test_dog_filter(self):
        """Detection with Difference of Gaussians filter should work."""
        from flika.spt.detection.thunderstorm import ThunderSTORMDetector

        frame = np.zeros((80, 80), dtype=np.float64)
        yy, xx = np.mgrid[-5:6, -5:6]
        spot = 1000 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
        frame[40 - 5:40 + 6, 40 - 5:40 + 6] += spot
        frame += np.random.normal(100, 10, frame.shape)

        det = ThunderSTORMDetector(filter_type='dog', threshold=1.5)
        locs = det.detect_frame(frame)
        assert locs.ndim == 2
