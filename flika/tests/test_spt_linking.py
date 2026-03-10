"""Tests for SPT linking modules."""
import pytest
import numpy as np


class TestGreedyLinker:
    def test_basic_linking(self):
        """Two particles moving in straight lines should produce 2 tracks."""
        from flika.spt.linking.greedy_linker import link_particles, tracks_to_dict

        # 2 particles moving in straight lines
        locs = np.array([
            [0, 10, 10], [0, 50, 50],
            [1, 11, 10], [1, 51, 50],
            [2, 12, 10], [2, 52, 50],
        ], dtype=float)
        tracks, stats = link_particles(locs, max_distance=5)
        assert stats['num_tracks'] == 2
        assert stats['mean_track_length'] == 3.0

    def test_gap_closing(self):
        """Greedy linker should close a 1-frame gap."""
        from flika.spt.linking.greedy_linker import link_particles

        locs = np.array([
            [0, 10, 10], [1, 11, 10], [3, 13, 10],  # gap at frame 2
        ], dtype=float)
        tracks, stats = link_particles(locs, max_distance=5, max_gap=2)
        assert stats['num_tracks'] == 1  # should close the gap

    def test_tracks_to_dict(self):
        """tracks_to_dict should return a dict of sorted track arrays."""
        from flika.spt.linking.greedy_linker import link_particles, tracks_to_dict

        locs = np.array([[0, 10, 10], [1, 11, 10], [2, 12, 10]], dtype=float)
        tracks, _ = link_particles(locs, max_distance=5)
        d = tracks_to_dict(locs, tracks)
        assert len(d) == 1
        assert d[0].shape == (3, 3)

    def test_tracks_to_array(self):
        """tracks_to_array should append a track_id column."""
        from flika.spt.linking.greedy_linker import link_particles, tracks_to_array

        locs = np.array([
            [0, 10, 10], [0, 50, 50],
            [1, 11, 10], [1, 51, 50],
        ], dtype=float)
        tracks, _ = link_particles(locs, max_distance=5)
        arr = tracks_to_array(locs, tracks)
        assert arr.shape[1] == locs.shape[1] + 1  # original cols + track_id

    def test_empty_input(self):
        """Empty localizations should return empty tracks."""
        from flika.spt.linking.greedy_linker import link_particles

        locs = np.empty((0, 3), dtype=float)
        tracks, stats = link_particles(locs, max_distance=5)
        assert tracks == []
        assert stats['num_tracks'] == 0

    def test_min_track_length_filter(self):
        """Tracks shorter than min_track_length should be filtered."""
        from flika.spt.linking.greedy_linker import link_particles

        locs = np.array([
            [0, 10, 10],  # single point = track of length 1
            [0, 50, 50], [1, 51, 50], [2, 52, 50],  # track of length 3
        ], dtype=float)
        tracks, stats = link_particles(locs, max_distance=5, min_track_length=2)
        assert stats['num_tracks'] == 1
        assert stats['mean_track_length'] == 3.0

    def test_max_distance_constraint(self):
        """Particles too far apart should not be linked."""
        from flika.spt.linking.greedy_linker import link_particles

        locs = np.array([
            [0, 10, 10],
            [1, 100, 100],  # very far away
        ], dtype=float)
        tracks, stats = link_particles(locs, max_distance=5, min_track_length=1)
        assert stats['num_tracks'] == 2  # two separate single-point tracks


class TestUTrackLinker:
    def test_basic_lap_linking(self):
        """LAP-based linking should find at least 1 track."""
        from flika.spt.linking.utrack_linker import UTrackLinker

        locs = np.array([
            [0, 10, 10], [0, 50, 50],
            [1, 11, 10], [1, 51, 50],
            [2, 12, 10], [2, 52, 50],
        ], dtype=float)
        linker = UTrackLinker(max_distance=10, max_gap=1, min_track_length=2)
        tracks, stats = linker.link(locs)
        assert stats['num_tracks'] >= 1

    def test_empty_input(self):
        """Empty localizations should return empty tracks."""
        from flika.spt.linking.utrack_linker import UTrackLinker

        locs = np.empty((0, 3), dtype=float)
        linker = UTrackLinker(max_distance=10)
        tracks, stats = linker.link(locs)
        assert tracks == []
        assert stats['num_tracks'] == 0

    def test_motion_model_brownian(self):
        """Brownian motion model should be accepted."""
        from flika.spt.linking.utrack_linker import UTrackLinker

        locs = np.array([
            [0, 10, 10], [1, 11, 11], [2, 12, 12],
        ], dtype=float)
        linker = UTrackLinker(max_distance=10, motion_model='brownian',
                              min_track_length=2)
        tracks, stats = linker.link(locs)
        assert stats['num_tracks'] >= 1

    def test_motion_model_mixed(self):
        """Mixed motion model should be accepted."""
        from flika.spt.linking.utrack_linker import UTrackLinker

        locs = np.array([
            [0, 10, 10], [1, 11, 11], [2, 12, 12],
        ], dtype=float)
        linker = UTrackLinker(max_distance=10, motion_model='mixed',
                              min_track_length=2)
        tracks, stats = linker.link(locs)
        assert stats['num_tracks'] >= 1


class TestKalman:
    def test_kalman_predict_update(self):
        """Kalman filter predict/update cycle should return correct shapes."""
        from flika.spt.linking.kalman import KalmanFilter2D

        kf = KalmanFilter2D()
        kf.update(np.array([10.0, 20.0]))
        pos, cov = kf.predict()
        assert pos.shape == (4,)
        assert cov.shape == (4, 4)

    def test_kalman_initialize(self):
        """KalmanFilter2D initialize should set position."""
        from flika.spt.linking.kalman import KalmanFilter2D

        kf = KalmanFilter2D()
        kf.initialize(np.array([5.0, 10.0]))
        assert np.allclose(kf.position, [5.0, 10.0])

    def test_kalman_search_radius(self):
        """get_search_radius should return a positive float."""
        from flika.spt.linking.kalman import KalmanFilter2D

        kf = KalmanFilter2D()
        kf.initialize(np.array([0.0, 0.0]))
        r = kf.get_search_radius()
        assert isinstance(r, float)
        assert r > 0

    def test_motion_regime_detector(self):
        """MotionRegimeDetector should return regime segments."""
        from flika.spt.linking.kalman import MotionRegimeDetector

        det = MotionRegimeDetector()
        positions = np.column_stack([np.arange(20), np.arange(20)])
        regimes = det.detect_regimes(positions)
        # Returns a list of segment dicts or per-point labels
        assert len(regimes) >= 1
        if isinstance(regimes[0], dict):
            # Segment format: [{'start': 0, 'end': N, 'type': '...'}]
            assert all(r['type'] in ('brownian', 'linear', 'confined')
                       for r in regimes)
            # Segments should cover all positions
            assert regimes[0]['start'] == 0
            assert regimes[-1]['end'] == 20
        else:
            # Per-point format
            assert all(r in ('brownian', 'linear', 'confined')
                       for r in regimes)

    def test_motion_regime_short_track(self):
        """Short track should still return valid results."""
        from flika.spt.linking.kalman import MotionRegimeDetector

        det = MotionRegimeDetector(min_regime_length=5)
        positions = np.array([[0, 0], [1, 1]])
        regimes = det.detect_regimes(positions)
        assert len(regimes) >= 1
        # Regimes may be segment dicts or per-point strings
        if isinstance(regimes[0], dict):
            assert all(r['type'] in ('brownian', 'linear', 'confined')
                       for r in regimes)
        else:
            assert all(r == 'brownian' for r in regimes)

    def test_mixed_motion_predictor(self):
        """MixedMotionPredictor predict should return a dict with position."""
        from flika.spt.linking.kalman import MixedMotionPredictor

        pred = MixedMotionPredictor()
        pred.initialize(np.array([10.0, 20.0]))
        result = pred.predict()
        assert 'position' in result
        assert 'search_radius' in result
        assert result['position'].shape == (2,)
        assert result['search_radius'] > 0

    def test_mixed_motion_update(self):
        """MixedMotionPredictor update should return a position array."""
        from flika.spt.linking.kalman import MixedMotionPredictor

        pred = MixedMotionPredictor()
        pred.initialize(np.array([10.0, 20.0]))
        pred.predict()
        pos = pred.update(np.array([11.0, 21.0]))
        assert pos.shape == (2,)
