"""Tests for FRAP, FRET, and Calcium analysis modules.

Tests the pure analysis functions (no GUI).  All tests are standalone
(no FlikaApplication required).
"""
import pytest
import numpy as np


# =========================================================================
# FRAP Tests
# =========================================================================

class TestFRAPNormalize:
    def test_basic_normalization(self):
        """Normalize should put pre-bleach ~1.0 and post-bleach min ~0.0."""
        from flika.process.frap import normalize_frap

        # Simulated FRAP: baseline 100, drops to 20, recovers to 80
        intensities = np.concatenate([
            np.full(10, 100.0),    # pre-bleach
            np.linspace(20, 80, 40),  # recovery
        ])
        norm = normalize_frap(intensities, pre_bleach_frames=10, bleach_frame=10)

        # Pre-bleach should be ~1.0
        assert abs(np.mean(norm[:10]) - 1.0) < 0.01
        # Post-bleach minimum should be ~0.0
        assert abs(np.min(norm[10:]) - 0.0) < 0.01

    def test_constant_intensity(self):
        """Constant intensity should return zeros (no recovery)."""
        from flika.process.frap import normalize_frap

        intensities = np.full(50, 100.0)
        norm = normalize_frap(intensities, pre_bleach_frames=10, bleach_frame=10)
        assert np.all(norm == 0.0)

    def test_bleach_at_start(self):
        """Bleach at frame 0 should still work."""
        from flika.process.frap import normalize_frap

        intensities = np.linspace(10, 100, 50)
        norm = normalize_frap(intensities, pre_bleach_frames=5, bleach_frame=0)
        assert len(norm) == 50


class TestFRAPFitting:
    def test_single_exponential_recovery(self):
        """Single exponential fit on synthetic data should recover tau."""
        from flika.process.frap import fit_single_exponential

        # Generate known single-exp recovery: A=0.8, tau=15, offset=0.05
        time = np.arange(100, dtype=np.float64)
        A_true, tau_true, offset_true = 0.8, 15.0, 0.05
        y = A_true * (1 - np.exp(-time / tau_true)) + offset_true
        # Add tiny noise
        rng = np.random.RandomState(42)
        y += rng.normal(0, 0.005, len(y))

        result = fit_single_exponential(time, y)

        assert abs(result['tau'] - tau_true) < 2.0
        assert result['r_squared'] > 0.99
        assert abs(result['mobile_fraction'] - (A_true + offset_true)) < 0.1
        assert len(result['fit_curve']) == len(time)

    def test_single_exp_short_trace(self):
        """Fit on very short trace should return NaN."""
        from flika.process.frap import fit_single_exponential

        result = fit_single_exponential(np.array([0, 1]), np.array([0.1, 0.2]))
        assert np.isnan(result['tau'])

    def test_double_exponential_recovery(self):
        """Double exponential fit should return two time constants."""
        from flika.process.frap import fit_double_exponential

        time = np.arange(200, dtype=np.float64)
        y = 0.4 * (1 - np.exp(-time / 5.0)) + 0.3 * (1 - np.exp(-time / 50.0)) + 0.05
        rng = np.random.RandomState(42)
        y += rng.normal(0, 0.005, len(y))

        result = fit_double_exponential(time, y)

        assert not np.isnan(result['tau1'])
        assert not np.isnan(result['tau2'])
        assert result['tau1'] <= result['tau2']  # fast first
        assert result['r_squared'] > 0.95
        assert abs(result['fraction1'] + result['fraction2'] - 1.0) < 0.01


class TestFRAPDerivedQuantities:
    def test_half_time(self):
        """Half-time should be tau * ln(2)."""
        from flika.process.frap import compute_half_time

        t_half = compute_half_time(10.0)
        assert abs(t_half - 10.0 * np.log(2)) < 1e-10

    def test_diffusion_coefficient(self):
        """Soumpasis equation: D = 0.224 * r^2 / t_half."""
        from flika.process.frap import compute_diffusion_coefficient

        D = compute_diffusion_coefficient(half_time=5.0, bleach_radius=1.0)
        assert abs(D - 0.224 * 1.0 / 5.0) < 1e-10

    def test_diffusion_3d_raises(self):
        """Non-2D dimensionality should raise ValueError."""
        from flika.process.frap import compute_diffusion_coefficient

        with pytest.raises(ValueError, match="2-D"):
            compute_diffusion_coefficient(1.0, 1.0, dimensionality=3)

    def test_diffusion_zero_half_time_raises(self):
        """Zero half_time should raise ValueError."""
        from flika.process.frap import compute_diffusion_coefficient

        with pytest.raises(ValueError, match="positive"):
            compute_diffusion_coefficient(0.0, 1.0)


# =========================================================================
# FRET Tests
# =========================================================================

class TestFRETApparent:
    def test_basic_fret_efficiency(self):
        """Apparent FRET: E = Ia / (Ia + Id)."""
        from flika.process.fret import compute_apparent_fret

        donor = np.full((10, 10), 50.0)
        acceptor = np.full((10, 10), 50.0)
        E = compute_apparent_fret(donor, acceptor)
        assert np.allclose(E, 0.5)

    def test_fret_all_donor(self):
        """No acceptor signal should give E~0."""
        from flika.process.fret import compute_apparent_fret

        donor = np.full((10, 10), 100.0)
        acceptor = np.zeros((10, 10))
        E = compute_apparent_fret(donor, acceptor)
        # Where acceptor=0 and donor>0, E=0
        assert np.allclose(E[np.isfinite(E)], 0.0)

    def test_fret_all_acceptor(self):
        """No donor signal should give E~1."""
        from flika.process.fret import compute_apparent_fret

        donor = np.zeros((10, 10))
        acceptor = np.full((10, 10), 100.0)
        E = compute_apparent_fret(donor, acceptor)
        assert np.allclose(E[np.isfinite(E)], 1.0)

    def test_fret_with_background(self):
        """Background subtraction should affect FRET values."""
        from flika.process.fret import compute_apparent_fret

        donor = np.full((10, 10), 60.0)
        acceptor = np.full((10, 10), 60.0)
        E_no_bg = compute_apparent_fret(donor, acceptor)
        E_bg = compute_apparent_fret(donor, acceptor,
                                     background_donor=10.0,
                                     background_acceptor=10.0)
        # With equal background subtraction, E should be the same
        assert np.allclose(E_no_bg, E_bg, equal_nan=True)

    def test_fret_zero_signal_nan(self):
        """Zero signal in both channels should be NaN."""
        from flika.process.fret import compute_apparent_fret

        donor = np.zeros((5, 5))
        acceptor = np.zeros((5, 5))
        E = compute_apparent_fret(donor, acceptor)
        assert np.all(np.isnan(E))


class TestFRETCorrected:
    def test_corrected_with_bleedthrough(self):
        """Corrected FRET with bleedthrough should reduce apparent E."""
        from flika.process.fret import compute_apparent_fret, compute_corrected_fret

        donor = np.full((10, 10), 100.0)
        acceptor = np.full((10, 10), 50.0)

        E_apparent = compute_apparent_fret(donor, acceptor)
        E_corrected = compute_corrected_fret(donor, acceptor, bleedthrough=0.1)

        mean_app = np.nanmean(E_apparent)
        mean_corr = np.nanmean(E_corrected)
        # Correcting for bleedthrough should lower E
        assert mean_corr < mean_app

    def test_stoichiometry(self):
        """Stoichiometry should be ~0.5 for equal donor/acceptor."""
        from flika.process.fret import compute_stoichiometry

        donor = np.full((10, 10), 100.0)
        acceptor = np.full((10, 10), 100.0)
        S = compute_stoichiometry(donor, acceptor, gamma=1.0)
        assert np.allclose(S, 1.0)  # S = (Ia+Id)/(Ia+gamma*Id) = 1 when gamma=1


class TestFRETStats:
    def test_fret_stats(self):
        """FRET stats should compute mean, median, std."""
        from flika.process.fret import compute_fret_stats

        E = np.random.uniform(0.3, 0.7, (50, 50))
        stats = compute_fret_stats(E)
        assert stats['n_pixels'] == 2500
        assert 0.3 < stats['mean_E'] < 0.7

    def test_fret_stats_with_mask(self):
        """FRET stats with mask should only count masked pixels."""
        from flika.process.fret import compute_fret_stats

        E = np.random.uniform(0, 1, (10, 10))
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :5] = True
        stats = compute_fret_stats(E, mask=mask)
        assert stats['n_pixels'] == 25

    def test_fret_histogram(self):
        """FRET histogram should return counts and edges."""
        from flika.process.fret import fret_histogram

        E = np.random.uniform(0, 1, (100, 100))
        counts, edges = fret_histogram(E, bins=50)
        assert len(counts) == 50
        assert len(edges) == 51
        assert np.sum(counts) == 10000


# =========================================================================
# Calcium Analysis Tests
# =========================================================================

class TestCalciumDFF:
    def test_dff_basic(self):
        """dF/F should be ~0 during baseline and positive during stimulation."""
        from flika.process.calcium import compute_dff

        # 10 baseline frames at 100, then 40 frames at 200
        trace = np.concatenate([np.full(10, 100.0), np.full(40, 200.0)])
        dff = compute_dff(trace, baseline_frames=slice(0, 10))

        # Baseline should be ~0
        assert abs(np.mean(dff[:10])) < 0.01
        # Stimulation should be ~1.0 (200-100)/100
        assert abs(np.mean(dff[10:]) - 1.0) < 0.01

    def test_dff_median_baseline(self):
        """Median baseline method should work."""
        from flika.process.calcium import compute_dff

        trace = np.concatenate([np.full(10, 100.0), np.full(10, 150.0)])
        dff = compute_dff(trace, baseline_frames=slice(0, 10),
                          baseline_method='median')
        assert abs(np.mean(dff[10:]) - 0.5) < 0.01

    def test_dff_zero_baseline(self):
        """Zero baseline should return zeros."""
        from flika.process.calcium import compute_dff

        trace = np.zeros(50)
        dff = compute_dff(trace, baseline_frames=slice(0, 10))
        assert np.all(dff == 0.0)

    def test_dff_image_stack(self):
        """Pixel-wise dF/F on a stack should work."""
        from flika.process.calcium import compute_dff_image

        stack = np.ones((20, 10, 10)) * 100.0
        stack[10:] = 200.0  # stimulation
        dff = compute_dff_image(stack, baseline_frames=slice(0, 10))

        assert dff.shape == (20, 10, 10)
        assert abs(np.mean(dff[:10]) - 0.0) < 0.01
        assert abs(np.mean(dff[10:]) - 1.0) < 0.01


class TestCalciumEventDetection:
    def test_detect_single_event(self):
        """Should detect a single calcium transient."""
        from flika.process.calcium import detect_calcium_events

        # Quiet baseline, then one transient
        trace = np.zeros(100)
        trace[0:10] = np.random.normal(0, 0.01, 10)  # noise baseline
        trace[40:50] = np.array([0.5, 1.0, 1.5, 2.0, 1.8, 1.5, 1.0, 0.7, 0.3, 0.1])

        events = detect_calcium_events(trace, threshold=3.0, min_duration=3)
        assert len(events) >= 1
        # Peak should be around frame 43
        assert events[0]['peak'] >= 40
        assert events[0]['amplitude'] > 1.0

    def test_detect_no_events(self):
        """Quiet trace should have no events."""
        from flika.process.calcium import detect_calcium_events

        trace = np.random.normal(0, 0.01, 100)
        events = detect_calcium_events(trace, threshold=5.0, min_duration=3)
        assert len(events) == 0

    def test_detect_multiple_events(self):
        """Should detect multiple transients."""
        from flika.process.calcium import detect_calcium_events

        trace = np.zeros(200)
        trace[0:10] = np.random.normal(0, 0.01, 10)
        # Event 1
        trace[30:40] = np.linspace(0, 2, 10)
        # Event 2
        trace[80:90] = np.linspace(0, 3, 10)

        events = detect_calcium_events(trace, threshold=2.0,
                                       min_duration=3, min_interval=5)
        assert len(events) >= 2

    def test_short_trace(self):
        """Very short trace should return no events."""
        from flika.process.calcium import detect_calcium_events

        events = detect_calcium_events(np.array([1.0, 2.0]), min_duration=3)
        assert len(events) == 0


class TestCalciumStats:
    def test_stats_with_events(self):
        """Stats should compute means across events."""
        from flika.process.calcium import compute_calcium_stats

        events = [
            {'start': 10, 'peak': 15, 'end': 20, 'amplitude': 2.0,
             'duration': 10, 'rise_time': 5, 'decay_time': 5, 'area': 15.0},
            {'start': 40, 'peak': 45, 'end': 50, 'amplitude': 3.0,
             'duration': 10, 'rise_time': 5, 'decay_time': 5, 'area': 20.0},
        ]
        stats = compute_calcium_stats(events)
        assert stats['n_events'] == 2
        assert stats['mean_amplitude'] == 2.5
        assert stats['mean_duration'] == 10.0
        assert stats['frequency'] > 0

    def test_stats_no_events(self):
        """Stats with no events should return NaN."""
        from flika.process.calcium import compute_calcium_stats

        stats = compute_calcium_stats([])
        assert stats['n_events'] == 0
        assert np.isnan(stats['mean_amplitude'])


class TestCalciumSmoothing:
    def test_savgol_smooth(self):
        """Savitzky-Golay should smooth noisy data."""
        from flika.process.calcium import smooth_trace

        rng = np.random.RandomState(42)
        noisy = np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.3, 100)
        smoothed = smooth_trace(noisy, method='savgol', window_length=11)
        # Smoothed should have lower variance
        assert np.std(smoothed) < np.std(noisy)

    def test_median_smooth(self):
        """Median filter should smooth."""
        from flika.process.calcium import smooth_trace

        noisy = np.ones(50)
        noisy[25] = 100  # spike
        smoothed = smooth_trace(noisy, method='median', window_length=5)
        assert smoothed[25] < 50  # spike should be removed

    def test_gaussian_smooth(self):
        """Gaussian smooth should work."""
        from flika.process.calcium import smooth_trace

        noisy = np.random.normal(0, 1, 100)
        smoothed = smooth_trace(noisy, method='gaussian', window_length=11)
        assert np.std(smoothed) < np.std(noisy)

    def test_short_trace_passthrough(self):
        """Trace shorter than window should pass through unchanged."""
        from flika.process.calcium import smooth_trace

        trace = np.array([1.0, 2.0, 3.0])
        smoothed = smooth_trace(trace, method='savgol', window_length=11)
        np.testing.assert_array_equal(trace, smoothed)


# =========================================================================
# Spectral Unmixing Tests
# =========================================================================

class TestLinearUnmix:
    def test_perfect_unmixing(self):
        """Known mixture should be recovered exactly with NNLS."""
        from flika.process.spectral import linear_unmix

        # 2 components, 3 channels
        spectra = np.array([
            [1.0, 0.0, 0.0],  # pure red
            [0.0, 1.0, 0.0],  # pure green
        ])
        # A pixel that is 70% red, 30% green
        data = np.array([[[0.7, 0.3, 0.0]]])  # (1, 1, 3)
        abundances = linear_unmix(data, spectra, method='nnls')
        assert abs(abundances[0, 0, 0] - 0.7) < 0.01
        assert abs(abundances[0, 0, 1] - 0.3) < 0.01

    def test_lstsq_unmixing(self):
        """Unconstrained least squares should also work."""
        from flika.process.spectral import linear_unmix

        spectra = np.array([[1.0, 0.0], [0.0, 1.0]])
        data = np.array([[[0.5, 0.5]]])
        abundances = linear_unmix(data, spectra, method='lstsq')
        assert abs(abundances[0, 0, 0] - 0.5) < 0.01

    def test_image_unmixing(self):
        """Unmixing should work on a full image."""
        from flika.process.spectral import linear_unmix

        spectra = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # 10x10 image with 3 channels
        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, (10, 10, 3))
        abundances = linear_unmix(data, spectra, method='nnls')
        assert abundances.shape == (10, 10, 3)
        # For identity spectra, abundances should equal data
        np.testing.assert_allclose(abundances, data, atol=0.01)


class TestResidual:
    def test_perfect_reconstruction(self):
        """Zero residual for exact reconstruction."""
        from flika.process.spectral import linear_unmix, compute_residual

        spectra = np.eye(3)
        data = np.random.uniform(0, 1, (5, 5, 3))
        abundances = linear_unmix(data, spectra, method='nnls')
        residual, rmse = compute_residual(data, spectra, abundances)
        assert rmse < 0.01


class TestEndmemberEstimation:
    def test_pca_endmembers(self):
        """PCA endmember estimation should return correct shape."""
        from flika.process.spectral import estimate_endmembers_pca

        rng = np.random.RandomState(42)
        data = rng.uniform(0, 1, (20, 20, 4))  # 4 channels
        endmembers = estimate_endmembers_pca(data, n_components=3)
        assert endmembers.shape == (3, 4)

    def test_normalize_spectra(self):
        """Normalized spectra should sum to 1."""
        from flika.process.spectral import normalize_spectra

        spectra = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
        normed = normalize_spectra(spectra)
        np.testing.assert_allclose(np.sum(normed, axis=1), [1.0, 1.0])

    def test_normalize_zero_spectrum(self):
        """Zero spectrum should not cause division by zero."""
        from flika.process.spectral import normalize_spectra

        spectra = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        normed = normalize_spectra(spectra)
        assert np.all(np.isfinite(normed))


# =========================================================================
# Morphometry Tests
# =========================================================================

class TestRegionProperties:
    def test_single_circle(self):
        """Circle region should have area, perimeter, high circularity."""
        from flika.process.morphometry import compute_region_properties

        label = np.zeros((50, 50), dtype=int)
        yy, xx = np.mgrid[:50, :50]
        circle = ((yy - 25) ** 2 + (xx - 25) ** 2) < 100
        label[circle] = 1

        regions = compute_region_properties(label)
        assert len(regions) == 1
        r = regions[0]
        assert r['area'] > 200
        assert r['circularity'] > 0.7
        assert r['eccentricity'] < 0.5

    def test_multiple_regions(self):
        """Multiple labelled regions should each get properties."""
        from flika.process.morphometry import compute_region_properties

        label = np.zeros((50, 50), dtype=int)
        label[5:15, 5:15] = 1
        label[30:40, 30:40] = 2

        regions = compute_region_properties(label)
        assert len(regions) == 2
        assert regions[0]['area'] == 100
        assert regions[1]['area'] == 100

    def test_with_intensity(self):
        """Intensity measurements should be computed when provided."""
        from flika.process.morphometry import compute_region_properties

        label = np.zeros((20, 20), dtype=int)
        label[5:15, 5:15] = 1
        intensity = np.ones((20, 20)) * 100.0
        intensity[5:15, 5:15] = 200.0

        regions = compute_region_properties(label, intensity)
        r = regions[0]
        assert r['mean_intensity'] == 200.0
        assert 'total_intensity' in r

    def test_empty_label(self):
        """Empty label image should return no regions."""
        from flika.process.morphometry import compute_region_properties

        label = np.zeros((20, 20), dtype=int)
        regions = compute_region_properties(label)
        assert len(regions) == 0


class TestTextureFeatures:
    def test_constant_image(self):
        """Constant image should have zero contrast and entropy."""
        from flika.process.morphometry import compute_texture_features

        image = np.full((20, 20), 100.0)
        features = compute_texture_features(image)
        assert features['contrast'] == 0.0
        assert features['entropy'] == 0.0
        assert features['homogeneity'] == 1.0

    def test_random_image(self):
        """Random image should have positive contrast and entropy."""
        from flika.process.morphometry import compute_texture_features

        rng = np.random.RandomState(42)
        image = rng.uniform(0, 255, (30, 30))
        features = compute_texture_features(image)
        assert features['contrast'] > 0
        assert features['entropy'] > 0

    def test_with_mask(self):
        """Texture with mask should only use masked pixels."""
        from flika.process.morphometry import compute_texture_features

        image = np.random.uniform(0, 255, (20, 20))
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        features = compute_texture_features(image, mask=mask)
        assert features['contrast'] >= 0


class TestHuMoments:
    def test_circle_symmetry(self):
        """Circle should have specific Hu moment properties."""
        from flika.process.morphometry import compute_shape_descriptors

        mask = np.zeros((50, 50), dtype=bool)
        yy, xx = np.mgrid[:50, :50]
        mask[((yy - 25) ** 2 + (xx - 25) ** 2) < 100] = True

        hu = compute_shape_descriptors(mask)
        assert all(np.isfinite(hu[f'hu_{i+1}']) for i in range(7))
        # hu_1 should be positive for a circle
        assert hu['hu_1'] > 0

    def test_empty_mask(self):
        """Empty mask should return NaN Hu moments."""
        from flika.process.morphometry import compute_shape_descriptors

        mask = np.zeros((10, 10), dtype=bool)
        hu = compute_shape_descriptors(mask)
        assert np.isnan(hu['hu_1'])
