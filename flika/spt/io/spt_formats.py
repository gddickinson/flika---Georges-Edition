"""SPT data format I/O for flika.

Supports reading and writing localizations and tracks in ThunderSTORM CSV,
flika CSV, and JSON formats with automatic format detection.
"""
import numpy as np
import pandas as pd
import json
import os
import datetime
from ...logger import logger


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path):
    """Detect file format from extension and column headers.

    Inspects the file extension first (for JSON), then reads the header line
    of CSV files to distinguish ThunderSTORM from flika format.

    Args:
        path: Path to the data file.

    Returns:
        One of ``'thunderstorm'``, ``'flika'``, ``'json'``, or ``'unknown'``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        return 'json'

    # For CSV / TSV files, peek at the header row.
    try:
        with open(path, 'r') as f:
            header = f.readline().strip()
    except UnicodeDecodeError:
        return 'unknown'

    if not header:
        return 'unknown'

    if 'x [nm]' in header or 'y [nm]' in header:
        return 'thunderstorm'

    header_lower = header.lower()
    if 'x' in header_lower and ('frame' in header_lower or 'id' in header_lower):
        return 'flika'

    return 'unknown'


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def _read_thunderstorm(path, pixel_size):
    """Read ThunderSTORM CSV and convert nanometre coordinates to pixels.

    ThunderSTORM uses 1-based frame numbering; the returned DataFrame uses
    0-based frames.  Coordinates are divided by *pixel_size* to convert from
    nanometres to pixels.
    """
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=['frame', 'x', 'y', 'intensity'])

    result = pd.DataFrame()
    result['frame'] = df['frame'].astype(int) - 1  # ThunderSTORM is 1-based
    result['x'] = df['x [nm]'] / pixel_size
    result['y'] = df['y [nm]'] / pixel_size

    if 'intensity [photon]' in df.columns:
        result['intensity'] = df['intensity [photon]']
    elif 'intensity' in df.columns:
        result['intensity'] = df['intensity']
    else:
        result['intensity'] = 0.0

    # Carry over optional ThunderSTORM columns under cleaned names.
    for col in ['sigma [nm]', 'uncertainty [nm]', 'chi2']:
        if col in df.columns:
            clean = col.replace(' [nm]', '_nm')
            result[clean] = df[col]

    if 'id' in df.columns:
        result['detection_id'] = df['id']

    return result


def _read_flika_csv(path):
    """Read flika-format CSV.

    Column names are normalised: ``track_number`` / ``track_id`` /
    ``trackid`` all become ``track_id``.  If an ``intensity`` column is
    missing it is filled with zeros.
    """
    df = pd.read_csv(path)
    if df.empty:
        # Return with canonical columns so downstream code is safe.
        cols = ['frame', 'x', 'y', 'intensity']
        return pd.DataFrame(columns=cols)

    # Build a renaming map by inspecting each column.
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('track_number', 'track_id', 'trackid'):
            col_map[c] = 'track_id'
        elif cl == 'x':
            col_map[c] = 'x'
        elif cl == 'y':
            col_map[c] = 'y'
        elif cl == 'frame':
            col_map[c] = 'frame'
        elif cl == 'intensity':
            col_map[c] = 'intensity'

    df = df.rename(columns=col_map)

    if 'frame' in df.columns:
        df['frame'] = df['frame'].astype(int)

    if 'intensity' not in df.columns:
        df['intensity'] = 0.0

    return df


def _read_json_locs(path):
    """Read JSON track / localisation format.

    Accepts two top-level keys:

    * ``tracks`` -- list of ``{track_id, points: [{frame, x, y, ...}]}``
    * ``localizations`` -- flat list of ``{frame, x, y, ...}``

    Returns a DataFrame with at least ``frame``, ``x``, ``y``, ``intensity``
    columns (and ``track_id`` when tracks are present).
    """
    with open(path, 'r') as f:
        data = json.load(f)

    rows = []
    if 'tracks' in data:
        for track in data['tracks']:
            tid = track.get('track_id', track.get('id', 0))
            for pt in track.get('points', []):
                rows.append({
                    'frame': int(pt['frame']),
                    'x': float(pt['x']),
                    'y': float(pt['y']),
                    'intensity': float(pt.get('intensity', 0)),
                    'track_id': int(tid),
                })
    elif 'localizations' in data:
        for pt in data['localizations']:
            rows.append({
                'frame': int(pt['frame']),
                'x': float(pt['x']),
                'y': float(pt['y']),
                'intensity': float(pt.get('intensity', 0)),
            })

    if not rows:
        cols = ['frame', 'x', 'y', 'intensity']
        if 'tracks' in data:
            cols.append('track_id')
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public reading API
# ---------------------------------------------------------------------------

def read_localizations(path, format='auto', pixel_size=108.0):
    """Read localization data from file.

    Args:
        path: File path to a CSV or JSON file.
        format: ``'auto'``, ``'thunderstorm'``, ``'flika'``, or ``'json'``.
            When ``'auto'``, the format is detected via :func:`detect_format`.
        pixel_size: Nanometres per pixel.  Used only for ThunderSTORM files
            to convert coordinates from nanometres to pixels.

    Returns:
        :class:`~pandas.DataFrame` with at least the columns ``frame``,
        ``x``, ``y``, ``intensity``.  Coordinates are always in pixels.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *format* is ``'auto'`` and the format cannot be
            determined, or if *pixel_size* is not positive.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")

    if format == 'auto':
        format = detect_format(path)
        if format == 'unknown':
            raise ValueError(
                f"Cannot auto-detect format for {path}. "
                "Specify format='thunderstorm', 'flika', or 'json'."
            )

    if format == 'json':
        return _read_json_locs(path)
    elif format == 'thunderstorm':
        return _read_thunderstorm(path, pixel_size)
    elif format == 'flika':
        return _read_flika_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format!r}")


def read_tracks(path, format='auto', pixel_size=108.0):
    """Read tracked data (localizations with track IDs).

    This is a convenience wrapper around :func:`read_localizations` that
    logs a warning when the resulting DataFrame does not contain a
    ``track_id`` column.

    Args:
        path: File path.
        format: See :func:`read_localizations`.
        pixel_size: See :func:`read_localizations`.

    Returns:
        :class:`~pandas.DataFrame` with columns ``frame``, ``x``, ``y``,
        ``intensity``, and (ideally) ``track_id`` plus any additional
        feature columns present in the file.
    """
    df = read_localizations(path, format=format, pixel_size=pixel_size)
    if 'track_id' not in df.columns:
        logger.warning("No track_id column found in %s", path)
    return df


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------

def _write_thunderstorm(df, path, pixel_size):
    """Write ThunderSTORM format CSV (pixel coords -> nm, 1-based frames)."""
    if df.empty:
        # Write a valid header-only file.
        header_cols = ['frame', 'x [nm]', 'y [nm]', 'intensity [photon]']
        pd.DataFrame(columns=header_cols).to_csv(path, index=False)
        return

    out = pd.DataFrame()
    out['frame'] = df['frame'].astype(int) + 1  # 1-based
    out['x [nm]'] = df['x'] * pixel_size
    out['y [nm]'] = df['y'] * pixel_size

    if 'intensity' in df.columns:
        out['intensity [photon]'] = df['intensity']
    else:
        out['intensity [photon]'] = 0.0

    out.to_csv(path, index=False)


def _write_json(df, path):
    """Write JSON format (tracks or flat localizations)."""
    if df.empty:
        # Write a valid but empty structure.
        data = {'localizations': []}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return

    if 'track_id' in df.columns:
        tracks = []
        for tid, group in df.groupby('track_id'):
            points = []
            for _, row in group.iterrows():
                pt = {
                    'frame': int(row['frame']),
                    'x': float(row['x']),
                    'y': float(row['y']),
                }
                if 'intensity' in row.index:
                    pt['intensity'] = float(row['intensity'])
                points.append(pt)
            tracks.append({'track_id': int(tid), 'points': points})
        data = {'tracks': tracks}
    else:
        locs = []
        for _, row in df.iterrows():
            pt = {
                'frame': int(row['frame']),
                'x': float(row['x']),
                'y': float(row['y']),
            }
            if 'intensity' in row.index:
                pt['intensity'] = float(row['intensity'])
            locs.append(pt)
        data = {'localizations': locs}

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Public writing API
# ---------------------------------------------------------------------------

def write_localizations(df, path, format='flika', pixel_size=108.0):
    """Write localization data to file.

    Args:
        df: :class:`~pandas.DataFrame` with at least ``frame``, ``x``,
            ``y`` columns.  Coordinates must be in pixels.
        path: Output file path.
        format: ``'flika'`` (CSV), ``'thunderstorm'`` (CSV with nm coords),
            or ``'json'``.
        pixel_size: Nanometres per pixel for ThunderSTORM output.

    Raises:
        ValueError: If *df* is missing required columns or *pixel_size*
            is not positive.
    """
    if df is None:
        raise ValueError("DataFrame must not be None")

    required = {'frame', 'x', 'y'}
    if not df.empty:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size}")

    # Ensure parent directory exists.
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    if format == 'thunderstorm':
        _write_thunderstorm(df, path, pixel_size)
    elif format == 'json':
        _write_json(df, path)
    elif format == 'flika':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format!r}")


def write_tracks(df, path, format='flika', pixel_size=108.0):
    """Write tracked data (localizations with track IDs) to file.

    This is a convenience alias for :func:`write_localizations`.  All
    arguments are forwarded directly.
    """
    write_localizations(df, path, format=format, pixel_size=pixel_size)


# ---------------------------------------------------------------------------
# Per-track features I/O
# ---------------------------------------------------------------------------

def write_features(features_df, path):
    """Write per-track feature DataFrame to CSV.

    Args:
        features_df: :class:`~pandas.DataFrame` of per-track features.
        path: Output CSV path.

    Raises:
        ValueError: If *features_df* is ``None``.
    """
    if features_df is None:
        raise ValueError("features_df must not be None")

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    features_df.to_csv(path, index=False)


def read_features(path):
    """Read per-track features from CSV.

    Args:
        path: Path to CSV file.

    Returns:
        :class:`~pandas.DataFrame` of per-track features.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Analysis metadata
# ---------------------------------------------------------------------------

def write_analysis_metadata(path, params, stats):
    """Write analysis metadata to JSON.

    Creates a JSON file containing the analysis parameters, summary
    statistics, and a timestamp.

    Args:
        path: Output JSON path.
        params: ``dict`` of analysis parameters (must be JSON-serialisable).
        stats: ``dict`` of analysis statistics (must be JSON-serialisable).

    Raises:
        TypeError: If *params* or *stats* are not dicts.
    """
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")
    if not isinstance(stats, dict):
        raise TypeError(f"stats must be a dict, got {type(stats).__name__}")

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    metadata = {
        'analysis_timestamp': datetime.datetime.now().isoformat(),
        'parameters': params,
        'statistics': stats,
    }
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def write_particle_data(particle_data, path, format='flika', pixel_size=108.0):
    """Write a :class:`~flika.spt.particle_data.ParticleData` to file.

    Convenience wrapper that extracts the DataFrame and delegates to
    :func:`write_localizations`.

    Args:
        particle_data: A :class:`ParticleData` instance.
        path: Output file path.
        format: ``'flika'``, ``'thunderstorm'``, or ``'json'``.
        pixel_size: Nanometres per pixel (ThunderSTORM only).
    """
    if format == 'thunderstorm':
        particle_data.to_thunderstorm_csv(path, pixel_size)
    elif format == 'flika':
        particle_data.to_flika_csv(path)
    elif format == 'json':
        write_localizations(particle_data.df, path, format='json')
    else:
        raise ValueError(f"Unsupported format: {format!r}")


def read_to_particle_data(path, format='auto', pixel_size=108.0):
    """Read a file and return a :class:`~flika.spt.particle_data.ParticleData`.

    Convenience wrapper around :func:`read_localizations` that returns a
    ParticleData instance instead of a raw DataFrame.

    Args:
        path: File path.
        format: See :func:`read_localizations`.
        pixel_size: See :func:`read_localizations`.

    Returns:
        :class:`ParticleData` instance.
    """
    df = read_localizations(path, format=format, pixel_size=pixel_size)

    from ..particle_data import ParticleData

    # Ensure required columns
    if 'id' not in df.columns:
        import numpy as _np
        df.insert(0, 'id', _np.arange(len(df), dtype=_np.int64))
    if 'track_id' not in df.columns:
        import numpy as _np
        df['track_id'] = _np.int64(-1)
    elif df['track_id'].dtype != 'int64':
        df['track_id'] = df['track_id'].astype('int64')

    return ParticleData(df)


def read_analysis_metadata(path):
    """Read analysis metadata from JSON.

    Args:
        path: Path to the metadata JSON file.

    Returns:
        ``dict`` with keys ``analysis_timestamp``, ``parameters``, and
        ``statistics``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)
