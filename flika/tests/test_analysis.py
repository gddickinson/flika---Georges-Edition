import numpy as np
import pytest
from .. import global_vars as g
from ..window import Window


class TestPearson:
    def test_perfect_positive(self):
        from ..process.colocalization import pearson_correlation
        ch1 = np.arange(100, dtype=np.float64).reshape(10, 10)
        ch2 = ch1 * 2.0 + 5.0
        r = pearson_correlation(ch1, ch2)
        assert abs(r - 1.0) < 1e-10

    def test_perfect_negative(self):
        from ..process.colocalization import pearson_correlation
        ch1 = np.arange(100, dtype=np.float64).reshape(10, 10)
        ch2 = -ch1
        r = pearson_correlation(ch1, ch2)
        assert abs(r - (-1.0)) < 1e-10

    def test_uncorrelated(self):
        from ..process.colocalization import pearson_correlation
        rng = np.random.default_rng(42)
        ch1 = rng.random((100, 100))
        ch2 = rng.random((100, 100))
        r = pearson_correlation(ch1, ch2)
        assert abs(r) < 0.1  # should be near zero for random data

    def test_with_mask(self):
        from ..process.colocalization import pearson_correlation
        ch1 = np.arange(100, dtype=np.float64).reshape(10, 10)
        ch2 = ch1.copy()
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        r = pearson_correlation(ch1, ch2, mask=mask)
        assert abs(r - 1.0) < 1e-10


class TestManders:
    def test_full_overlap(self):
        from ..process.colocalization import manders_coefficients
        ch1 = np.ones((10, 10), dtype=np.float64)
        ch2 = np.ones((10, 10), dtype=np.float64)
        m1, m2 = manders_coefficients(ch1, ch2, 0.5, 0.5)
        assert abs(m1 - 1.0) < 1e-10
        assert abs(m2 - 1.0) < 1e-10

    def test_no_overlap(self):
        from ..process.colocalization import manders_coefficients
        ch1 = np.zeros((10, 10), dtype=np.float64)
        ch2 = np.ones((10, 10), dtype=np.float64)
        ch1[:5, :] = 1.0
        ch2[:5, :] = 0.0
        m1, m2 = manders_coefficients(ch1, ch2, 0.5, 0.5)
        assert m1 == 0.0  # no ch1 where ch2 > thresh
        assert m2 == 0.0  # no ch2 where ch1 > thresh


class TestLiICQ:
    def test_positive_correlation(self):
        from ..process.colocalization import li_icq
        ch1 = np.arange(100, dtype=np.float64).reshape(10, 10)
        ch2 = ch1 * 2.0
        icq = li_icq(ch1, ch2)
        assert icq > 0  # positively correlated

    def test_independent(self):
        from ..process.colocalization import li_icq
        rng = np.random.default_rng(42)
        ch1 = rng.random((200, 200))
        ch2 = rng.random((200, 200))
        icq = li_icq(ch1, ch2)
        assert abs(icq) < 0.1  # should be near zero


class TestWatershed:
    def setup_method(self):
        # Create synthetic binary image with 3 circles
        self.im = np.zeros((100, 100), dtype=np.float32)
        for cx, cy, r in [(25, 25, 10), (75, 25, 10), (50, 75, 10)]:
            yy, xx = np.ogrid[:100, :100]
            mask = ((xx - cx)**2 + (yy - cy)**2) <= r**2
            self.im[mask] = 1.0
        self.win = Window(self.im)

    def teardown_method(self):
        if not self.win.closed:
            self.win.close()
        # Close any other windows created during tests
        while g.windows:
            g.windows[-1].close()

    def test_distance_transform(self):
        from ..process.watershed import distance_transform
        g.win = self.win
        result_win = distance_transform()
        assert result_win is not None
        assert result_win.image.shape == (100, 100)
        # Center of circles should have highest distance values
        assert result_win.image[25, 25] > 0
        result_win.close()

    def test_distance_transform_binary(self):
        from ..process.watershed import distance_transform
        import scipy.ndimage
        g.win = self.win
        result_win = distance_transform()
        expected = scipy.ndimage.distance_transform_edt(self.im > 0)
        np.testing.assert_allclose(result_win.image, expected, rtol=1e-5)
        result_win.close()
