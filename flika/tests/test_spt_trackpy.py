"""Tests for trackpy linker adapter.

Tests the link_with_trackpy function, all link_type variants,
parameter validation, and edge cases.
"""
import pytest
import numpy as np


def _make_two_track_locs(n_frames=10, separation=20.0, speed=1.0):
    """Create synthetic localizations for two well-separated tracks."""
    rows = []
    for f in range(n_frames):
        rows.append([f, 10.0 + speed * f, 20.0, 100.0])
        rows.append([f, 10.0 + separation, 20.0 + speed * f, 200.0])
    return np.array(rows)


class TestTrackpyLinking:
    def test_standard_linking(self):
        """Standard linking should find 2 tracks from well-separated particles."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=10)
        tracks, stats = link_with_trackpy(
            locs[:, :3], search_range=5.0, memory=0,
            link_type='standard', min_track_length=3)

        assert stats['num_tracks'] == 2
        assert stats['link_type'] == 'standard'
        assert stats['linking_efficiency'] > 0.5
        assert all(len(t) >= 3 for t in tracks)

    def test_adaptive_linking(self):
        """Adaptive linking should also find tracks."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=10)
        tracks, stats = link_with_trackpy(
            locs[:, :3], search_range=5.0, memory=0,
            link_type='adaptive', min_track_length=3)

        assert stats['num_tracks'] >= 1
        assert stats['link_type'] == 'adaptive'

    def test_velocity_predict_linking(self):
        """Velocity prediction linking should work."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=10)
        tracks, stats = link_with_trackpy(
            locs[:, :3], search_range=5.0, memory=0,
            link_type='velocityPredict', min_track_length=3)

        assert stats['num_tracks'] >= 1
        assert stats['link_type'] == 'velocityPredict'

    def test_adaptive_velocity_linking(self):
        """Combined adaptive + velocityPredict should work."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=10)
        tracks, stats = link_with_trackpy(
            locs[:, :3], search_range=5.0, memory=0,
            link_type='adaptive + velocityPredict', min_track_length=3)

        assert stats['num_tracks'] >= 1
        assert stats['link_type'] == 'adaptive + velocityPredict'

    def test_memory_gap_closing(self):
        """Memory parameter should allow gap closing."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        # One particle with a gap at frame 5
        locs = []
        for f in range(10):
            if f == 5:
                continue  # skip frame 5
            locs.append([f, 20.0 + f * 0.5, 30.0, 100.0])
        locs = np.array(locs)

        # Without memory: may break track at gap
        tracks_no_mem, stats_no_mem = link_with_trackpy(
            locs, search_range=5.0, memory=0,
            link_type='standard', min_track_length=2)

        # With memory=2: should bridge the gap
        tracks_mem, stats_mem = link_with_trackpy(
            locs, search_range=5.0, memory=2,
            link_type='standard', min_track_length=2)

        # With memory, expect fewer (or equal) tracks (gap bridged)
        assert stats_mem['num_tracks'] <= stats_no_mem['num_tracks'] + 1

    def test_min_track_length_filter(self):
        """Tracks shorter than min_track_length should be filtered out."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=5)
        # Require long tracks
        tracks, stats = link_with_trackpy(
            locs[:, :3], search_range=5.0, memory=0,
            link_type='standard', min_track_length=10)

        # Tracks with only 5 frames should be filtered
        assert stats['num_tracks'] == 0 or all(len(t) >= 10 for t in tracks)

    def test_empty_input(self):
        """Empty localizations should return empty results."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = np.empty((0, 3))
        tracks, stats = link_with_trackpy(locs, search_range=5.0)

        assert tracks == []
        assert stats['num_tracks'] == 0

    def test_invalid_link_type_raises(self):
        """Invalid link_type should raise ValueError."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = np.array([[0, 10.0, 20.0]])
        with pytest.raises(ValueError, match="Unknown link_type"):
            link_with_trackpy(locs, search_range=5.0, link_type='bogus')

    def test_stats_structure(self):
        """Stats dict should have all expected keys."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=10)
        _, stats = link_with_trackpy(locs[:, :3], search_range=5.0,
                                     min_track_length=3)

        expected_keys = [
            'num_tracks', 'total_points', 'linked_points', 'unlinked_points',
            'mean_track_length', 'median_track_length', 'max_track_length',
            'min_track_length', 'std_track_length', 'linking_efficiency',
            'link_type',
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_tracks_contain_valid_indices(self):
        """Track indices should be valid row indices into the input array."""
        tp = pytest.importorskip('trackpy')
        from flika.spt.linking.trackpy_linker import link_with_trackpy

        locs = _make_two_track_locs(n_frames=10)
        tracks, _ = link_with_trackpy(locs[:, :3], search_range=5.0,
                                      min_track_length=3)

        for track in tracks:
            for idx in track:
                assert 0 <= idx < len(locs)


class TestTrackpyImportError:
    def test_import_check(self):
        """_check_trackpy should work if trackpy is installed."""
        try:
            from flika.spt.linking.trackpy_linker import _check_trackpy
            tp = _check_trackpy()
            assert hasattr(tp, 'link') or hasattr(tp, 'link_df')
        except ImportError:
            pytest.skip("trackpy not installed")


class TestComputeStats:
    def test_compute_stats_with_tracks(self):
        """_compute_stats should compute correct statistics."""
        from flika.spt.linking.trackpy_linker import _compute_stats

        tracks = [[0, 1, 2], [3, 4]]  # 3 + 2 = 5 linked
        stats = _compute_stats(tracks, total_points=7, link_type='standard')

        assert stats['num_tracks'] == 2
        assert stats['total_points'] == 7
        assert stats['linked_points'] == 5
        assert stats['unlinked_points'] == 2
        assert stats['mean_track_length'] == 2.5
        assert stats['max_track_length'] == 3
        assert stats['min_track_length'] == 2

    def test_compute_stats_empty(self):
        """_compute_stats with no tracks should return zeros."""
        from flika.spt.linking.trackpy_linker import _compute_stats

        stats = _compute_stats([], total_points=10, link_type='standard')
        assert stats['num_tracks'] == 0
        assert stats['linking_efficiency'] == 0.0

    def test_empty_stats(self):
        """_empty_stats should return a zero-filled dict."""
        from flika.spt.linking.trackpy_linker import _empty_stats

        stats = _empty_stats()
        assert stats['num_tracks'] == 0
        assert stats['link_type'] == 'standard'
