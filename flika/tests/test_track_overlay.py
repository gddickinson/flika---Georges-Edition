import numpy as np
import pytest
from flika import global_vars as g
from flika.window import Window


class TestTrackOverlay:
    def setup_method(self):
        self.im = np.random.random((20, 64, 64)).astype(np.float32)
        self.win = Window(self.im)
        # Create some synthetic track data
        # Columns: frame, x, y, track_id
        self.tracks = np.array([
            [0, 10, 20, 1],
            [1, 11, 21, 1],
            [2, 12, 22, 1],
            [3, 13, 23, 1],
            [4, 14, 24, 1],
            [0, 50, 50, 2],
            [1, 51, 51, 2],
            [2, 52, 52, 2],
            [3, 53, 53, 2],
            [4, 54, 54, 2],
            [5, 55, 55, 2],
        ], dtype=np.float64)

    def teardown_method(self):
        if not self.win.closed:
            self.win.close()

    def test_create_overlay(self):
        from ..viewers.track_overlay import TrackOverlay
        overlay = TrackOverlay(self.win)
        assert overlay.window is self.win
        assert overlay.visible is True
        assert overlay.tail_length == 10
        overlay.cleanup()

    def test_load_from_array(self):
        from ..viewers.track_overlay import TrackOverlay
        overlay = TrackOverlay(self.win)
        overlay.load_tracks_from_array(self.tracks)
        assert len(overlay.tracks) == 2  # two track IDs
        assert 1 in overlay.tracks
        assert 2 in overlay.tracks
        overlay.cleanup()

    def test_color_modes(self):
        from ..viewers.track_overlay import TrackOverlay
        overlay = TrackOverlay(self.win)
        overlay.load_tracks_from_array(self.tracks)
        for mode in ['track_id', 'velocity', 'displacement', 'time']:
            overlay.set_color_mode(mode)
            assert overlay.color_mode == mode
            assert len(overlay._colors) == 2
        overlay.cleanup()

    def test_tail_length(self):
        from ..viewers.track_overlay import TrackOverlay
        overlay = TrackOverlay(self.win)
        overlay.load_tracks_from_array(self.tracks)
        overlay.set_tail_length(5)
        assert overlay.tail_length == 5
        overlay.cleanup()

    def test_min_track_length_filter(self):
        from ..viewers.track_overlay import TrackOverlay
        overlay = TrackOverlay(self.win)
        # Add a short track (2 points) and a long track (6 points)
        data = np.array([
            [0, 10, 10, 1],
            [1, 11, 11, 1],
            [0, 30, 30, 2],
            [1, 31, 31, 2],
            [2, 32, 32, 2],
            [3, 33, 33, 2],
            [4, 34, 34, 2],
            [5, 35, 35, 2],
        ], dtype=np.float64)
        overlay.load_tracks_from_array(data)
        overlay.set_min_track_length(3)
        assert overlay.min_track_length == 3
        # Track 1 has only 2 points, should be filtered
        overlay.cleanup()

    def test_cleanup(self):
        from ..viewers.track_overlay import TrackOverlay
        overlay = TrackOverlay(self.win)
        overlay.load_tracks_from_array(self.tracks)
        overlay.cleanup()
        assert len(overlay._plot_items) == 0
        assert len(overlay._point_items) == 0
