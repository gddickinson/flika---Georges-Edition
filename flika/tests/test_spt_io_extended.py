"""Extended I/O tests covering gaps in original test_spt_io.py.

Tests: unknown format detection, JSON tracks, ThunderSTORM extras,
missing columns errors, pixel size validation, ParticleData convenience
wrappers, empty DataFrames, and error paths.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json


class TestDetectFormatExtended:
    def test_unknown_format_no_header(self):
        """File with unrecognized header should return 'unknown'."""
        from flika.spt.io.spt_formats import detect_format

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('foo,bar,baz\n1,2,3\n')
            path = f.name
        try:
            assert detect_format(path) == 'unknown'
        finally:
            os.unlink(path)

    def test_unknown_format_empty_file(self):
        """Empty file should return 'unknown'."""
        from flika.spt.io.spt_formats import detect_format

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            path = f.name
        try:
            assert detect_format(path) == 'unknown'
        finally:
            os.unlink(path)

    def test_detect_format_file_not_found(self):
        """Nonexistent file should raise FileNotFoundError."""
        from flika.spt.io.spt_formats import detect_format

        with pytest.raises(FileNotFoundError):
            detect_format('/nonexistent/file.csv')


class TestReadWriteThunderstormExtras:
    def test_thunderstorm_with_extra_columns(self):
        """ThunderSTORM CSV with sigma, uncertainty should be read."""
        from flika.spt.io.spt_formats import read_localizations

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('"id","frame","x [nm]","y [nm]","sigma [nm]",'
                    '"intensity [photon]","uncertainty [nm]"\n')
            f.write('1,1,1080.0,2160.0,162.0,500.0,10.8\n')
            path = f.name
        try:
            df = read_localizations(path, format='thunderstorm', pixel_size=108.0)
            assert 'sigma_nm' in df.columns
            assert 'uncertainty_nm' in df.columns
        finally:
            os.unlink(path)

    def test_thunderstorm_round_trip_via_write(self):
        """ThunderSTORM write/read round trip should preserve data."""
        from flika.spt.io.spt_formats import write_localizations, read_localizations

        df = pd.DataFrame({
            'frame': [0, 1],
            'x': [10.0, 20.0],
            'y': [15.0, 25.0],
            'intensity': [100.0, 200.0],
        })
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_localizations(df, path, format='thunderstorm', pixel_size=108.0)
            loaded = read_localizations(path, format='thunderstorm', pixel_size=108.0)
            assert len(loaded) == 2
            assert abs(loaded['x'].iloc[0] - 10.0) < 0.01
        finally:
            os.unlink(path)


class TestJSONTracksFormat:
    def test_json_with_tracks(self):
        """JSON with tracks structure should be read correctly."""
        from flika.spt.io.spt_formats import read_localizations

        data = {
            'tracks': [
                {
                    'track_id': 0,
                    'points': [
                        {'frame': 0, 'x': 10.0, 'y': 20.0, 'intensity': 100.0},
                        {'frame': 1, 'x': 11.0, 'y': 21.0, 'intensity': 110.0},
                    ]
                },
                {
                    'track_id': 1,
                    'points': [
                        {'frame': 0, 'x': 50.0, 'y': 50.0, 'intensity': 200.0},
                    ]
                }
            ]
        }
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(data, f)
            path = f.name
        try:
            df = read_localizations(path, format='json')
            assert len(df) == 3
            assert 'track_id' in df.columns
            assert set(df['track_id'].unique()) == {0, 1}
        finally:
            os.unlink(path)

    def test_json_write_with_tracks(self):
        """Writing DataFrame with track_id should produce tracks JSON."""
        from flika.spt.io.spt_formats import write_localizations

        df = pd.DataFrame({
            'frame': [0, 1, 0],
            'x': [10.0, 11.0, 50.0],
            'y': [20.0, 21.0, 50.0],
            'intensity': [100.0, 110.0, 200.0],
            'track_id': [0, 0, 1],
        })
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            write_localizations(df, path, format='json')
            with open(path) as fh:
                data = json.load(fh)
            assert 'tracks' in data
            assert len(data['tracks']) == 2
        finally:
            os.unlink(path)

    def test_json_empty_tracks(self):
        """Empty tracks JSON should produce empty DataFrame."""
        from flika.spt.io.spt_formats import read_localizations

        data = {'tracks': []}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(data, f)
            path = f.name
        try:
            df = read_localizations(path, format='json')
            assert len(df) == 0
        finally:
            os.unlink(path)


class TestWriteValidation:
    def test_write_missing_columns_raises(self):
        """Writing DataFrame without required columns should raise."""
        from flika.spt.io.spt_formats import write_localizations

        df = pd.DataFrame({'a': [1], 'b': [2]})
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                write_localizations(df, path, format='flika')
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_write_negative_pixel_size_raises(self):
        """Negative pixel_size should raise ValueError."""
        from flika.spt.io.spt_formats import write_localizations

        df = pd.DataFrame({'frame': [0], 'x': [1.0], 'y': [2.0]})
        with pytest.raises(ValueError, match="positive"):
            write_localizations(df, '/tmp/test.csv',
                                format='thunderstorm', pixel_size=-1.0)

    def test_read_negative_pixel_size_raises(self):
        """Negative pixel_size in read should raise ValueError."""
        from flika.spt.io.spt_formats import read_localizations

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('frame,x,y\n0,1,2\n')
            path = f.name
        try:
            with pytest.raises(ValueError, match="positive"):
                read_localizations(path, format='flika', pixel_size=-1.0)
        finally:
            os.unlink(path)

    def test_write_unsupported_format_raises(self):
        """Unsupported format should raise ValueError."""
        from flika.spt.io.spt_formats import write_localizations

        df = pd.DataFrame({'frame': [0], 'x': [1.0], 'y': [2.0]})
        with pytest.raises(ValueError, match="Unsupported format"):
            write_localizations(df, '/tmp/test.xyz', format='hdf5')

    def test_write_features_none_raises(self):
        """Writing None features should raise ValueError."""
        from flika.spt.io.spt_formats import write_features

        with pytest.raises(ValueError, match="None"):
            write_features(None, '/tmp/features.csv')

    def test_write_metadata_non_dict_raises(self):
        """Non-dict params/stats should raise TypeError."""
        from flika.spt.io.spt_formats import write_analysis_metadata

        with pytest.raises(TypeError):
            write_analysis_metadata('/tmp/meta.json', 'not_a_dict', {})
        with pytest.raises(TypeError):
            write_analysis_metadata('/tmp/meta.json', {}, [1, 2, 3])

    def test_write_empty_dataframe(self):
        """Writing empty DataFrame should work (not crash)."""
        from flika.spt.io.spt_formats import write_localizations

        df = pd.DataFrame(columns=['frame', 'x', 'y', 'intensity'])
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_localizations(df, path, format='flika')
            loaded = pd.read_csv(path)
            assert len(loaded) == 0
        finally:
            os.unlink(path)

    def test_write_creates_parent_dir(self):
        """Writing to a nonexistent parent dir should create it."""
        from flika.spt.io.spt_formats import write_localizations

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'subdir', 'output.csv')
            df = pd.DataFrame({
                'frame': [0], 'x': [1.0], 'y': [2.0], 'intensity': [100.0]
            })
            write_localizations(df, path, format='flika')
            assert os.path.exists(path)


class TestParticleDataWrappers:
    def test_write_particle_data_flika(self):
        """write_particle_data with 'flika' format should work."""
        from flika.spt.io.spt_formats import write_particle_data, read_to_particle_data
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0], [1, 11.0, 21.0, 110.0]])
        pd_obj = ParticleData.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_particle_data(pd_obj, path, format='flika')
            loaded = read_to_particle_data(path, format='flika')
            assert loaded.n_localizations == 2
            assert isinstance(loaded, ParticleData)
        finally:
            os.unlink(path)

    def test_write_particle_data_thunderstorm(self):
        """write_particle_data with 'thunderstorm' format should work."""
        from flika.spt.io.spt_formats import write_particle_data
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0]])
        pd_obj = ParticleData.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_particle_data(pd_obj, path, format='thunderstorm', pixel_size=108.0)
            loaded = pd.read_csv(path)
            assert 'x [nm]' in loaded.columns
        finally:
            os.unlink(path)

    def test_write_particle_data_json(self):
        """write_particle_data with 'json' format should work."""
        from flika.spt.io.spt_formats import write_particle_data
        from flika.spt.particle_data import ParticleData

        arr = np.array([[0, 10.0, 20.0, 100.0]])
        pd_obj = ParticleData.from_numpy(arr)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            write_particle_data(pd_obj, path, format='json')
            with open(path) as fh:
                data = json.load(fh)
            # ParticleData always has track_id column, so JSON uses tracks format
            assert 'tracks' in data or 'localizations' in data
        finally:
            os.unlink(path)

    def test_write_particle_data_unsupported_raises(self):
        """Unsupported format in write_particle_data should raise."""
        from flika.spt.io.spt_formats import write_particle_data
        from flika.spt.particle_data import ParticleData

        pd_obj = ParticleData()
        with pytest.raises(ValueError):
            write_particle_data(pd_obj, '/tmp/test.xyz', format='hdf5')

    def test_read_to_particle_data_adds_missing_columns(self):
        """read_to_particle_data should ensure id and track_id columns."""
        from flika.spt.io.spt_formats import read_to_particle_data

        df = pd.DataFrame({
            'frame': [0, 1],
            'x': [10.0, 11.0],
            'y': [20.0, 21.0],
            'intensity': [100.0, 110.0],
        })
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            df.to_csv(path, index=False)
            pd_obj = read_to_particle_data(path, format='flika')
            assert 'id' in pd_obj.df.columns
            assert 'track_id' in pd_obj.df.columns
        finally:
            os.unlink(path)


class TestFlikaCsvColumnNormalization:
    def test_track_number_renamed(self):
        """Column 'track_number' should be renamed to 'track_id'."""
        from flika.spt.io.spt_formats import read_localizations

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('frame,x,y,intensity,track_number\n')
            f.write('0,10.0,20.0,100.0,0\n')
            path = f.name
        try:
            df = read_localizations(path, format='flika')
            assert 'track_id' in df.columns
        finally:
            os.unlink(path)

    def test_missing_intensity_filled(self):
        """Missing intensity column should be filled with 0."""
        from flika.spt.io.spt_formats import read_localizations

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('frame,x,y\n')
            f.write('0,10.0,20.0\n')
            path = f.name
        try:
            df = read_localizations(path, format='flika')
            assert 'intensity' in df.columns
            assert df['intensity'].iloc[0] == 0.0
        finally:
            os.unlink(path)

    def test_auto_detect_unknown_raises(self):
        """Auto-detect on unknown format should raise ValueError."""
        from flika.spt.io.spt_formats import read_localizations

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('col_a,col_b\n1,2\n')
            path = f.name
        try:
            with pytest.raises(ValueError, match="Cannot auto-detect"):
                read_localizations(path, format='auto')
        finally:
            os.unlink(path)
