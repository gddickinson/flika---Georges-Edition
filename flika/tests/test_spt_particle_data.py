"""Comprehensive tests for ParticleData model.

Tests the core data model: constructors, properties, accessors, mutation,
I/O methods, and edge cases.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os


class TestParticleDataInit:
    def test_empty_init(self):
        """Empty ParticleData should have zero localizations."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData()
        assert pd_obj.n_localizations == 0
        assert pd_obj.n_tracks == 0
        assert pd_obj.n_frames == 0
        assert len(pd_obj) == 0

    def test_init_from_dataframe(self):
        """ParticleData from a DataFrame should preserve data."""
        from flika.spt.particle_data import ParticleData

        df = pd.DataFrame({
            'id': [0, 1, 2],
            'frame': [0, 0, 1],
            'x': [10.0, 20.0, 11.0],
            'y': [15.0, 25.0, 16.0],
            'intensity': [100.0, 200.0, 110.0],
        })
        pd_obj = ParticleData(df)
        assert pd_obj.n_localizations == 3
        assert pd_obj.n_frames == 2

    def test_init_copies_dataframe(self):
        """ParticleData should copy the input DataFrame (not reference it)."""
        from flika.spt.particle_data import ParticleData

        df = pd.DataFrame({
            'id': [0], 'frame': [0], 'x': [1.0], 'y': [2.0], 'intensity': [3.0]
        })
        pd_obj = ParticleData(df)
        df.loc[0, 'x'] = 999.0
        assert pd_obj.df['x'].iloc[0] == 1.0


class TestParticleDataFromNumpy:
    def test_from_4col_array(self):
        """4-column array should be interpreted as [frame, x, y, intensity]."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0],
                        [1, 11.0, 21.0, 110.0]])
        pd_obj = ParticleData.from_numpy(arr)
        assert pd_obj.n_localizations == 2
        assert 'id' in pd_obj.df.columns
        assert 'track_id' in pd_obj.df.columns
        assert pd_obj.df['track_id'].iloc[0] == -1  # unlinked

    def test_from_3col_array(self):
        """3-column array should be [frame, x, y] with intensity=0."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0], [1, 11.0, 21.0]])
        pd_obj = ParticleData.from_numpy(arr)
        assert pd_obj.n_localizations == 2
        assert (pd_obj.df['intensity'] == 0.0).all()

    def test_from_5col_array(self):
        """5-column array should include track_id."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0, 0],
                        [1, 11.0, 21.0, 110.0, 0]])
        pd_obj = ParticleData.from_numpy(arr)
        assert pd_obj.df['track_id'].iloc[0] == 0

    def test_from_8col_array(self):
        """8-column array should be ThunderSTORM format."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0, 1.5, 1.5, 50.0, 5.0]])
        pd_obj = ParticleData.from_numpy(arr)
        assert 'sigma_x' in pd_obj.df.columns
        assert 'sigma_y' in pd_obj.df.columns
        assert 'background' in pd_obj.df.columns
        assert 'uncertainty' in pd_obj.df.columns

    def test_from_1d_array(self):
        """1-D array should be reshaped to (1, N)."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([0, 10.0, 20.0, 100.0])
        pd_obj = ParticleData.from_numpy(arr)
        assert pd_obj.n_localizations == 1

    def test_from_empty_array(self):
        """Empty array should return empty ParticleData."""
        from flika.spt.particle_data import ParticleData

        arr = np.empty((0, 4))
        pd_obj = ParticleData.from_numpy(arr)
        assert pd_obj.n_localizations == 0

    def test_from_custom_columns(self):
        """Custom column names should be applied."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 5.0, 10.0, 50.0]])
        pd_obj = ParticleData.from_numpy(arr, columns=['frame', 'x', 'y', 'intensity'])
        assert pd_obj.df['x'].iloc[0] == 5.0

    def test_from_6col_array(self):
        """6-column array should use auto-generated names for extras."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0, 1.0, 2.0]])
        pd_obj = ParticleData.from_numpy(arr)
        assert pd_obj.n_localizations == 1
        assert 'col_4' in pd_obj.df.columns or 'track_id' in pd_obj.df.columns


class TestParticleDataFromSptDict:
    def test_from_spt_dict_basic(self):
        """from_spt_dict with localizations and tracks."""
        from flika.spt.particle_data import ParticleData

        locs = np.array([[0, 10.0, 20.0, 100.0],
                         [1, 11.0, 21.0, 110.0],
                         [0, 50.0, 50.0, 200.0],
                         [1, 51.0, 51.0, 210.0]])
        tracks = [[0, 1], [2, 3]]
        spt_dict = {'localizations': locs, 'tracks': tracks}
        pd_obj = ParticleData.from_spt_dict(spt_dict)
        assert pd_obj.n_localizations == 4
        assert pd_obj.n_tracks == 2

    def test_from_spt_dict_no_tracks(self):
        """from_spt_dict without tracks should set all track_id=-1."""
        from flika.spt.particle_data import ParticleData

        locs = np.array([[0, 10.0, 20.0, 100.0]])
        spt_dict = {'localizations': locs}
        pd_obj = ParticleData.from_spt_dict(spt_dict)
        assert pd_obj.n_tracks == 0

    def test_from_spt_dict_empty(self):
        """from_spt_dict with empty localizations."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData.from_spt_dict({'localizations': []})
        assert pd_obj.n_localizations == 0

    def test_from_spt_dict_none_locs(self):
        """from_spt_dict with None localizations."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData.from_spt_dict({})
        assert pd_obj.n_localizations == 0


class TestParticleDataAccessors:
    @pytest.fixture
    def pd_with_tracks(self):
        from flika.spt.particle_data import ParticleData

        locs = np.array([
            [0, 10.0, 20.0, 100.0],
            [1, 11.0, 21.0, 110.0],
            [2, 12.0, 22.0, 120.0],
            [0, 50.0, 50.0, 200.0],
            [1, 51.0, 51.0, 210.0],
        ])
        pd_obj = ParticleData.from_numpy(locs)
        pd_obj.set_tracks([[0, 1, 2], [3, 4]])
        return pd_obj

    def test_frame_locs(self, pd_with_tracks):
        """frame_locs should return only rows for that frame."""
        fl = pd_with_tracks.frame_locs(0)
        assert len(fl) == 2
        assert set(fl['x'].values) == {10.0, 50.0}

    def test_frame_locs_nonexistent(self, pd_with_tracks):
        """frame_locs for nonexistent frame should return empty."""
        fl = pd_with_tracks.frame_locs(99)
        assert len(fl) == 0

    def test_track_locs(self, pd_with_tracks):
        """track_locs should return rows for a specific track."""
        tl = pd_with_tracks.track_locs(0)
        assert len(tl) == 3

    def test_track_locs_nonexistent(self, pd_with_tracks):
        """track_locs for nonexistent track should return empty."""
        tl = pd_with_tracks.track_locs(99)
        assert len(tl) == 0

    def test_track_ids(self, pd_with_tracks):
        """track_ids should return sorted unique IDs excluding -1."""
        ids = pd_with_tracks.track_ids()
        np.testing.assert_array_equal(ids, [0, 1])

    def test_track_ids_no_tracks(self):
        """track_ids on unlinked data should return empty array."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData.from_numpy(np.array([[0, 10.0, 20.0, 100.0]]))
        ids = pd_obj.track_ids()
        assert len(ids) == 0

    def test_track_summary(self, pd_with_tracks):
        """track_summary should have one row per track."""
        summary = pd_with_tracks.track_summary()
        assert len(summary) == 2
        assert 'n_points' in summary.columns
        assert 'first_frame' in summary.columns
        assert 'last_frame' in summary.columns
        # Track 0 has 3 points
        row0 = summary[summary['track_id'] == 0].iloc[0]
        assert row0['n_points'] == 3
        assert row0['first_frame'] == 0
        assert row0['last_frame'] == 2

    def test_track_summary_empty(self):
        """track_summary on empty data should return empty DataFrame."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData()
        summary = pd_obj.track_summary()
        assert len(summary) == 0

    def test_df_property(self, pd_with_tracks):
        """df property should return the underlying DataFrame."""
        df = pd_with_tracks.df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_repr(self, pd_with_tracks):
        """repr should be informative."""
        r = repr(pd_with_tracks)
        assert 'ParticleData' in r
        assert '5 localizations' in r
        assert '2 tracks' in r


class TestParticleDataMutation:
    def test_set_localizations(self):
        """set_localizations should replace all data."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData()
        arr = np.array([[0, 10.0, 20.0, 100.0], [1, 11.0, 21.0, 110.0]])
        pd_obj.set_localizations(arr, detection_params={'method': 'utrack'})
        assert pd_obj.n_localizations == 2
        assert pd_obj._detection_params['method'] == 'utrack'

    def test_set_tracks(self):
        """set_tracks should assign track IDs."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0],
                        [1, 11.0, 21.0, 110.0],
                        [0, 50.0, 50.0, 200.0]])
        pd_obj = ParticleData.from_numpy(arr)
        pd_obj.set_tracks([[0, 1], [2]], linking_params={'method': 'greedy'})
        assert pd_obj.n_tracks == 2
        assert pd_obj.df['track_id'].iloc[0] == 0
        assert pd_obj.df['track_id'].iloc[2] == 1
        assert pd_obj._linking_params['method'] == 'greedy'

    def test_set_tracks_with_invalid_indices(self):
        """set_tracks should skip out-of-bounds indices."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0]])
        pd_obj = ParticleData.from_numpy(arr)
        pd_obj.set_tracks([[0, 999]])  # 999 is out of bounds
        assert pd_obj.n_tracks == 1

    def test_set_features(self):
        """set_features should broadcast per-track values to rows."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0],
                        [1, 11.0, 21.0, 110.0]])
        pd_obj = ParticleData.from_numpy(arr)
        pd_obj.set_tracks([[0, 1]])
        pd_obj.set_features({0: {'radius_gyration': 1.5, 'straightness': 0.9}})
        assert 'radius_gyration' in pd_obj.df.columns
        assert pd_obj.df['radius_gyration'].iloc[0] == 1.5

    def test_set_features_no_track_id(self):
        """set_features on data without track_id should be a no-op."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData()
        pd_obj.set_features({0: {'rg': 1.0}})  # should not crash

    def test_set_classification(self):
        """set_classification should set labels per track."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0],
                        [1, 11.0, 21.0, 110.0]])
        pd_obj = ParticleData.from_numpy(arr)
        pd_obj.set_tracks([[0, 1]])
        pd_obj.set_classification({0: 'Mobile'})
        assert 'classification' in pd_obj.df.columns
        assert pd_obj.df['classification'].iloc[0] == 'Mobile'


class TestParticleDataIO:
    def test_to_flika_csv_and_read(self):
        """to_flika_csv should produce a file readable by pandas."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0], [1, 11.0, 21.0, 110.0]])
        pd_obj = ParticleData.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            pd_obj.to_flika_csv(path)
            loaded = pd.read_csv(path)
            assert len(loaded) == 2
            assert 'frame' in loaded.columns
        finally:
            os.unlink(path)

    def test_to_thunderstorm_csv(self):
        """to_thunderstorm_csv should convert to nm coords and 1-based frames."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0]])
        pd_obj = ParticleData.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            pd_obj.to_thunderstorm_csv(path, pixel_size=108.0)
            loaded = pd.read_csv(path)
            assert 'x [nm]' in loaded.columns
            assert loaded['frame'].iloc[0] == 1  # 1-based
            assert abs(loaded['x [nm]'].iloc[0] - 1080.0) < 0.01
        finally:
            os.unlink(path)

    def test_to_thunderstorm_csv_empty(self):
        """to_thunderstorm_csv on empty data should write header-only."""
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData()
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            pd_obj.to_thunderstorm_csv(path)
            loaded = pd.read_csv(path)
            assert len(loaded) == 0
        finally:
            os.unlink(path)

    def test_to_thunderstorm_csv_with_ts_columns(self):
        """ThunderSTORM export should include optional columns when present."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0, 1.5, 1.5, 50.0, 5.0]])
        pd_obj = ParticleData.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            pd_obj.to_thunderstorm_csv(path, pixel_size=108.0)
            loaded = pd.read_csv(path)
            assert 'sigma [nm]' in loaded.columns
            assert 'uncertainty [nm]' in loaded.columns
        finally:
            os.unlink(path)

    def test_to_spt_dict_roundtrip(self):
        """to_spt_dict should produce a dict that from_spt_dict can consume."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([
            [0, 10.0, 20.0, 100.0],
            [1, 11.0, 21.0, 110.0],
            [0, 50.0, 50.0, 200.0],
        ])
        pd_obj = ParticleData.from_numpy(arr)
        pd_obj.set_tracks([[0, 1], [2]])

        spt_dict = pd_obj.to_spt_dict()
        assert 'localizations' in spt_dict
        assert 'tracks' in spt_dict
        assert 'tracks_dict' in spt_dict
        assert len(spt_dict['tracks']) == 2
        assert 0 in spt_dict['tracks_dict']

        # Round-trip
        pd_obj2 = ParticleData.from_spt_dict(spt_dict)
        assert pd_obj2.n_localizations == 3
        assert pd_obj2.n_tracks == 2

    def test_to_dataframe_returns_copy(self):
        """to_dataframe should return a copy, not a reference."""
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0]])
        pd_obj = ParticleData.from_numpy(arr)
        df = pd_obj.to_dataframe()
        df.loc[0, 'x'] = 999.0
        assert pd_obj.df['x'].iloc[0] == 10.0
