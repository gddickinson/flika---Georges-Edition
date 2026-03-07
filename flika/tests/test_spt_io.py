"""Tests for SPT I/O."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os


class TestSPTFormats:
    def test_round_trip_localizations(self):
        """Localizations should survive a write-read round trip."""
        from flika.spt.io.spt_formats import write_localizations, read_localizations

        df = pd.DataFrame({
            'frame': [0, 0, 1],
            'x': [10.5, 30.1, 11.0],
            'y': [20.3, 40.7, 21.0],
            'intensity': [500.0, 600.0, 510.0],
        })
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_localizations(df, path, format='flika')
            loaded = read_localizations(path, format='flika')
            assert len(loaded) == 3
            assert 'frame' in loaded.columns
            assert 'x' in loaded.columns
            assert 'y' in loaded.columns
            assert 'intensity' in loaded.columns
        finally:
            os.unlink(path)

    def test_round_trip_tracks(self):
        """Tracked data should survive a write-read round trip."""
        from flika.spt.io.spt_formats import write_tracks, read_tracks

        df = pd.DataFrame({
            'frame': [0, 1, 0, 1],
            'x': [10.0, 11.0, 50.0, 51.0],
            'y': [20.0, 20.0, 50.0, 50.0],
            'intensity': [100.0, 110.0, 200.0, 210.0],
            'track_id': [0, 0, 1, 1],
        })
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_tracks(df, path, format='flika')
            loaded = read_tracks(path, format='flika')
            assert len(loaded) == 4
            assert 'track_id' in loaded.columns
        finally:
            os.unlink(path)

    def test_detect_format_thunderstorm(self):
        """ThunderSTORM CSV format should be detected correctly."""
        from flika.spt.io.spt_formats import detect_format

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('"id","frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]"\n')
            f.write('1,1,1000.0,2000.0,150.0,500.0\n')
            path = f.name
        try:
            fmt = detect_format(path)
            assert fmt == 'thunderstorm'
        finally:
            os.unlink(path)

    def test_detect_format_flika(self):
        """Flika CSV format should be detected correctly."""
        from flika.spt.io.spt_formats import detect_format

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('frame,x,y,intensity\n')
            f.write('0,10.5,20.3,500\n')
            path = f.name
        try:
            fmt = detect_format(path)
            assert fmt == 'flika'
        finally:
            os.unlink(path)

    def test_detect_format_json(self):
        """JSON format should be detected by extension."""
        from flika.spt.io.spt_formats import detect_format

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            f.write('{"localizations": []}')
            path = f.name
        try:
            fmt = detect_format(path)
            assert fmt == 'json'
        finally:
            os.unlink(path)

    def test_read_thunderstorm_csv(self):
        """ThunderSTORM CSV should be read and converted to pixel coords."""
        from flika.spt.io.spt_formats import read_localizations

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('"id","frame","x [nm]","y [nm]","intensity [photon]"\n')
            f.write('1,1,1080.0,2160.0,500.0\n')
            f.write('2,2,2160.0,3240.0,600.0\n')
            path = f.name
        try:
            df = read_localizations(path, format='thunderstorm', pixel_size=108.0)
            assert len(df) == 2
            # ThunderSTORM is 1-based, so frame 1 -> 0
            assert df['frame'].iloc[0] == 0
            # 1080 nm / 108 nm/px = 10 px
            assert abs(df['x'].iloc[0] - 10.0) < 0.01
        finally:
            os.unlink(path)

    def test_round_trip_json(self):
        """JSON round trip should preserve localization data."""
        from flika.spt.io.spt_formats import write_localizations, read_localizations

        df = pd.DataFrame({
            'frame': [0, 1, 2],
            'x': [10.0, 11.0, 12.0],
            'y': [20.0, 21.0, 22.0],
            'intensity': [500.0, 510.0, 520.0],
        })
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            write_localizations(df, path, format='json')
            loaded = read_localizations(path, format='json')
            assert len(loaded) == 3
            assert abs(loaded['x'].iloc[0] - 10.0) < 0.01
        finally:
            os.unlink(path)

    def test_write_features_and_read(self):
        """Feature CSV round trip should preserve data."""
        from flika.spt.io.spt_formats import write_features, read_features

        df = pd.DataFrame({
            'track_id': [0, 1, 2],
            'radius_gyration': [1.5, 2.3, 0.8],
            'diffusion_coefficient': [0.1, 0.5, 0.02],
        })
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            write_features(df, path)
            loaded = read_features(path)
            assert len(loaded) == 3
            assert 'track_id' in loaded.columns
        finally:
            os.unlink(path)

    def test_write_analysis_metadata(self):
        """Analysis metadata should be written and readable as JSON."""
        from flika.spt.io.spt_formats import (write_analysis_metadata,
                                               read_analysis_metadata)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            params = {'detector': 'utrack', 'max_distance': 5.0}
            stats = {'num_tracks': 10, 'mean_track_length': 15.0}
            write_analysis_metadata(path, params, stats)
            loaded = read_analysis_metadata(path)
            assert loaded['parameters']['detector'] == 'utrack'
            assert loaded['statistics']['num_tracks'] == 10
            assert 'analysis_timestamp' in loaded
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        """Reading a nonexistent file should raise FileNotFoundError."""
        from flika.spt.io.spt_formats import read_localizations

        with pytest.raises(FileNotFoundError):
            read_localizations('/nonexistent/path/file.csv')

    def test_write_none_raises(self):
        """Writing None should raise ValueError."""
        from flika.spt.io.spt_formats import write_localizations

        with pytest.raises(ValueError):
            write_localizations(None, '/tmp/test.csv')
