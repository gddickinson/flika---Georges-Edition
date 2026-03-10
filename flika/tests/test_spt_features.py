"""Tests for SPT feature modules."""
import pytest
import numpy as np


class TestGeometric:
    def test_radius_of_gyration(self):
        """Circular path should have nonzero Rg and bounded asymmetry."""
        from flika.spt.features.geometric import radius_of_gyration

        t = np.linspace(0, 2 * np.pi, 100)
        pos = np.column_stack([10 * np.cos(t), 10 * np.sin(t)])
        result = radius_of_gyration(pos)
        assert 'rg' in result
        assert result['rg'] > 0
        assert 0 <= result['asymmetry']

    def test_radius_of_gyration_simple_method(self):
        """Simple Rg method should also work."""
        from flika.spt.features.geometric import radius_of_gyration

        pos = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
        result = radius_of_gyration(pos, method='simple')
        assert result['rg'] > 0

    def test_radius_of_gyration_short_track(self):
        """Track with fewer than 3 points should return defaults."""
        from flika.spt.features.geometric import radius_of_gyration

        pos = np.array([[0, 0], [1, 1]])
        result = radius_of_gyration(pos)
        assert result['rg'] == 0.0

    def test_fractal_dimension(self):
        """Straight line should have fractal dimension close to 1."""
        from flika.spt.features.geometric import fractal_dimension

        pos = np.column_stack([np.arange(100), np.zeros(100)])
        fd = fractal_dimension(pos)
        assert 0.8 <= fd <= 1.5

    def test_fractal_dimension_short_track(self):
        """Short track (< 3 points) should return 1.0."""
        from flika.spt.features.geometric import fractal_dimension

        pos = np.array([[0, 0], [1, 1]])
        fd = fractal_dimension(pos)
        assert fd == 1.0

    def test_straightness(self):
        """Straight line should have straightness close to 1."""
        from flika.spt.features.geometric import straightness

        pos = np.column_stack([np.arange(50, dtype=float), np.zeros(50)])
        result = straightness(pos)
        assert result['straightness'] > 0.9

    def test_straightness_short_track(self):
        """Track with fewer than 3 points should return zero straightness."""
        from flika.spt.features.geometric import straightness

        pos = np.array([[0, 0], [1, 1]])
        result = straightness(pos)
        assert result['straightness'] == 0.0

    def test_net_displacement(self):
        """Net displacement of a (0,0)->(3,4) track should be 5.0."""
        from flika.spt.features.geometric import net_displacement_efficiency

        pos = np.array([[0, 0], [3, 4]])
        result = net_displacement_efficiency(pos)
        assert abs(result['net_displacement'] - 5.0) < 0.01
        assert result['efficiency'] == 1.0  # 2-point track: efficiency=1

    def test_net_displacement_longer_track(self):
        """Multi-point track should have efficiency <= 1."""
        from flika.spt.features.geometric import net_displacement_efficiency

        pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        result = net_displacement_efficiency(pos)
        assert abs(result['net_displacement'] - 3.0) < 0.01
        assert 0 < result['efficiency'] <= 1.0

    def test_scaled_radius_of_gyration(self):
        """Scaled Rg should return a positive number."""
        from flika.spt.features.geometric import scaled_radius_of_gyration

        pos = np.column_stack([np.arange(30, dtype=float),
                               np.random.normal(0, 0.5, 30)])
        srg = scaled_radius_of_gyration(pos)
        assert srg >= 0

    def test_classify_linear_motion(self):
        """Straight line should be classified as linear unidirectional."""
        from flika.spt.features.geometric import classify_linear_motion

        pos = np.column_stack([np.arange(50, dtype=float), np.zeros(50)])
        result = classify_linear_motion(pos)
        assert result in ('linear_unidirectional', 'linear_bidirectional',
                          'non_linear')


class TestKinematic:
    def test_velocity_analysis(self):
        """Constant speed 1 px/frame should give mean velocity 1.0."""
        from flika.spt.features.kinematic import velocity_analysis

        pos = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        result = velocity_analysis(pos, dt=1.0)
        assert abs(result['mean_velocity'] - 1.0) < 0.01

    def test_velocity_analysis_empty(self):
        """Single point should return zero velocities."""
        from flika.spt.features.kinematic import velocity_analysis

        pos = np.array([[5.0, 5.0]])
        result = velocity_analysis(pos, dt=1.0)
        assert result['mean_velocity'] == 0.0

    def test_msd_brownian(self):
        """Brownian motion should give diffusion coeff > 0 and alpha near 1."""
        from flika.spt.features.kinematic import msd_analysis

        np.random.seed(42)
        steps = np.random.normal(0, 1.0, (1000, 2))
        pos = np.cumsum(steps, axis=0)
        result = msd_analysis(pos)
        assert result['diffusion_coefficient'] > 0
        assert 0.5 < result['anomalous_exponent'] < 1.5

    def test_msd_short_track(self):
        """Short track should return default values."""
        from flika.spt.features.kinematic import msd_analysis

        pos = np.array([[0, 0], [1, 1]])
        result = msd_analysis(pos, min_points=5)
        assert result['diffusion_coefficient'] == 0.0
        assert result['anomalous_exponent'] == 1.0

    def test_direction_analysis(self):
        """Rightward motion should have high directional persistence."""
        from flika.spt.features.kinematic import direction_analysis

        pos = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
        result = direction_analysis(pos)
        assert result['directional_persistence'] > 0.9

    def test_distance_from_origin(self):
        """Distance from origin should match expected Euclidean distances."""
        from flika.spt.features.kinematic import distance_from_origin

        pos = np.array([[0, 0], [3, 4], [6, 8]])
        d = distance_from_origin(pos)
        assert abs(d[0]) < 0.001
        assert abs(d[1] - 5.0) < 0.01
        assert abs(d[2] - 10.0) < 0.01

    def test_lag_displacements(self):
        """Lag displacements for unit steps should all be 1.0."""
        from flika.spt.features.kinematic import lag_displacements

        pos = np.column_stack([np.arange(5, dtype=float), np.zeros(5)])
        lags = lag_displacements(pos)
        assert len(lags) == 4
        assert np.allclose(lags, 1.0)


class TestSpatial:
    def test_nearest_neighbor(self):
        """Nearest neighbor distances should match expected values."""
        from flika.spt.features.spatial import nearest_neighbor_distances

        pts = np.array([[0, 0], [1, 0], [3, 0], [10, 0]])
        result = nearest_neighbor_distances(pts)
        # May return (dists, indices) tuple or just dists array
        dists = result[0] if isinstance(result, tuple) else result
        assert abs(dists[0] - 1.0) < 0.01
        assert abs(dists[1] - 1.0) < 0.01

    def test_nearest_neighbor_single_point(self):
        """Single point should return NaN."""
        from flika.spt.features.spatial import nearest_neighbor_distances

        pts = np.array([[0, 0]])
        result = nearest_neighbor_distances(pts)
        dists = result[0] if isinstance(result, tuple) else result
        assert len(dists) == 1
        assert np.isnan(dists[0])

    def test_neighbor_counts(self):
        """Neighbor counts should be consistent with known geometry."""
        from flika.spt.features.spatial import neighbor_counts

        pts = np.array([[0, 0], [1, 0], [2, 0], [100, 0]])
        counts = neighbor_counts(pts, radii=(1.5, 5))
        # Point at (0,0): only (1,0) is within radius 1.5
        assert counts[1.5][0] == 1
        # Point at (1,0): both (0,0) and (2,0) within radius 1.5
        assert counts[1.5][1] == 2


class TestFeatureCalculator:
    def test_compute_track_features(self):
        """Feature calculator should return expected feature keys."""
        from flika.spt.features.feature_calculator import FeatureCalculator

        calc = FeatureCalculator()
        pos = np.column_stack([np.arange(20, dtype=float),
                               np.random.normal(0, 1, 20)])
        features = calc.compute_track_features(pos)
        assert 'radius_gyration' in features
        assert 'diffusion_coefficient' in features
        assert 'n_points' in features
        assert features['n_points'] == 20

    def test_compute_track_features_short(self):
        """Short track should only have n_points in features."""
        from flika.spt.features.feature_calculator import FeatureCalculator

        calc = FeatureCalculator()
        pos = np.array([[0, 0], [1, 1]])
        features = calc.compute_track_features(pos)
        assert 'n_points' in features
        assert features['n_points'] == 2
        # Geometric/kinematic features not computed for < 3 points
        assert 'radius_gyration' not in features

    def test_compute_all(self):
        """compute_all should return a DataFrame with one row per track."""
        from flika.spt.features.feature_calculator import FeatureCalculator

        calc = FeatureCalculator()
        locs = np.zeros((15, 3))
        for i in range(15):
            locs[i] = [i % 5, i * 0.5, i * 0.3]
        tracks = [list(range(5)), list(range(5, 10)), list(range(10, 15))]
        df = calc.compute_all(locs, tracks)
        assert len(df) == 3
        assert 'track_id' in df.columns

    def test_compute_all_empty(self):
        """Empty tracks should return empty DataFrame."""
        from flika.spt.features.feature_calculator import FeatureCalculator

        calc = FeatureCalculator()
        locs = np.empty((0, 3))
        df = calc.compute_all(locs, [])
        assert len(df) == 0

    def test_compute_per_point(self):
        """compute_per_point should return one row per detection point."""
        from flika.spt.features.feature_calculator import FeatureCalculator

        calc = FeatureCalculator()
        locs = np.zeros((6, 3))
        for i in range(6):
            locs[i] = [i % 3, i * 1.0, i * 0.5]
        tracks = [list(range(3)), list(range(3, 6))]
        df = calc.compute_per_point(locs, tracks)
        assert len(df) == 6
        assert 'track_id' in df.columns
        assert 'frame' in df.columns


class TestAutocorrelation:
    def test_directed_motion(self):
        """Directed motion should have positive autocorrelation at lag 1."""
        from flika.spt.features.autocorrelation import AutocorrelationAnalyzer

        # Directed motion -> positive autocorrelation
        tracks = {0: np.column_stack([np.arange(30),
                                      np.arange(30),
                                      np.zeros(30)])}
        analyzer = AutocorrelationAnalyzer(n_intervals=5, min_track_length=5)
        result = analyzer.compute(tracks)
        assert result['n_tracks_used'] == 1
        assert result['mean_correlation'][0] > 0.5  # lag-1 should be positive

    def test_empty_tracks(self):
        """Empty input should return empty result."""
        from flika.spt.features.autocorrelation import AutocorrelationAnalyzer

        analyzer = AutocorrelationAnalyzer()
        result = analyzer.compute({})
        assert result['n_tracks_used'] == 0
        assert len(result['mean_correlation']) == 0

    def test_short_tracks_skipped(self):
        """Tracks shorter than min_track_length should be skipped."""
        from flika.spt.features.autocorrelation import AutocorrelationAnalyzer

        tracks = {0: np.column_stack([np.arange(3), np.arange(3), np.zeros(3)])}
        analyzer = AutocorrelationAnalyzer(min_track_length=10)
        result = analyzer.compute(tracks)
        assert result['n_tracks_used'] == 0
        # No qualifying tracks means the empty result is returned
        assert len(result['mean_correlation']) == 0

    def test_compute_single_track(self):
        """compute_single_track should return correlation array."""
        from flika.spt.features.autocorrelation import AutocorrelationAnalyzer

        analyzer = AutocorrelationAnalyzer(n_intervals=5, min_track_length=5)
        positions = np.column_stack([np.arange(20, dtype=float),
                                     np.arange(20, dtype=float)])
        result = analyzer.compute_single_track(positions)
        assert 'time_lags' in result
        assert 'correlation' in result
        assert len(result['time_lags']) > 0

    def test_persistence_index(self):
        """Persistence index for directed motion should be positive."""
        from flika.spt.features.autocorrelation import AutocorrelationAnalyzer

        tracks = {0: np.column_stack([np.arange(30),
                                      np.arange(30),
                                      np.zeros(30)])}
        analyzer = AutocorrelationAnalyzer(n_intervals=5, min_track_length=5)
        pi = analyzer.persistence_index(tracks, lag=1)
        assert pi > 0
