# -*- coding: utf-8 -*-
"""
ParticleData — single source of truth for SPT localization/track data.

Wraps a pandas DataFrame with typed columns and fast accessors.
After every mutation the legacy ``window.metadata['spt']`` keys are
kept in sync so existing consumers work unchanged.
"""
import numpy as np
import pandas as pd
from ..logger import logger


# Core columns always present after detection
_CORE_COLUMNS = ['id', 'frame', 'x', 'y', 'intensity']

# Optional columns populated by ThunderSTORM detection
_TS_EXTRA_COLUMNS = ['sigma_x', 'sigma_y', 'background', 'uncertainty']

# Linking column
_LINK_COLUMN = 'track_id'


class ParticleData:
    """Pandas DataFrame-backed model for SPT localization and track data.

    Core columns (always present after detection):
        id (int64), frame (int64), x (float64), y (float64), intensity (float64)

    Optional detection columns (ThunderSTORM):
        sigma_x, sigma_y, background, uncertainty

    Linking column:
        track_id (int64, -1 = unlinked)

    Feature/classification columns are broadcast from per-track computation.
    """

    def __init__(self, df=None):
        if df is not None:
            self._df = df.copy()
        else:
            self._df = pd.DataFrame(columns=_CORE_COLUMNS)
            self._df['id'] = self._df['id'].astype('int64')
            self._df['frame'] = self._df['frame'].astype('int64')
        self._frame_index = {}  # dict[int] -> np.ndarray of row indices
        self._detection_params = {}
        self._linking_params = {}
        if len(self._df) > 0:
            self._rebuild_frame_index()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(cls, locs_array, columns=None):
        """Create from a numpy array of localizations.

        Parameters
        ----------
        locs_array : ndarray
            (N, C) array. If *columns* is None the first columns are
            interpreted as [frame, x, y, intensity, ...].
        columns : list of str, optional
            Column names.  If not given, defaults to
            ``['frame', 'x', 'y', 'intensity']`` (+ extras mapped to
            ThunderSTORM columns if C >= 8).
        """
        arr = np.asarray(locs_array, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return cls()

        n_cols = arr.shape[1]

        if columns is not None:
            df = pd.DataFrame(arr, columns=columns[:n_cols])
        elif n_cols == 4:
            # [frame, x, y, intensity]
            df = pd.DataFrame(arr, columns=['frame', 'x', 'y', 'intensity'])
        elif n_cols == 8:
            # ThunderSTORM detect_stack output:
            # [frame, x, y, intensity, sigma_x, sigma_y, background, uncertainty]
            df = pd.DataFrame(arr, columns=[
                'frame', 'x', 'y', 'intensity',
                'sigma_x', 'sigma_y', 'background', 'uncertainty'])
        elif n_cols == 5:
            # [frame, x, y, intensity, track_id]
            df = pd.DataFrame(arr, columns=[
                'frame', 'x', 'y', 'intensity', 'track_id'])
        elif n_cols == 3:
            df = pd.DataFrame(arr, columns=['frame', 'x', 'y'])
            df['intensity'] = 0.0
        else:
            cols = ['frame', 'x', 'y', 'intensity'] + \
                   [f'col_{i}' for i in range(4, n_cols)]
            df = pd.DataFrame(arr, columns=cols)

        # Ensure id column
        if 'id' not in df.columns:
            df.insert(0, 'id', np.arange(len(df), dtype=np.int64))
        # Ensure frame is int
        df['frame'] = df['frame'].astype('int64')
        df['id'] = df['id'].astype('int64')

        # Ensure track_id exists (unlinked = -1)
        if 'track_id' not in df.columns:
            df['track_id'] = np.int64(-1)
        else:
            df['track_id'] = df['track_id'].astype('int64')

        obj = cls.__new__(cls)
        obj._df = df
        obj._frame_index = {}
        obj._detection_params = {}
        obj._linking_params = {}
        obj._rebuild_frame_index()
        return obj

    @classmethod
    def from_spt_dict(cls, spt_dict):
        """Create from a legacy ``window.metadata['spt']`` dict.

        Reads 'localizations' (numpy array) and optionally 'tracks'
        (list of index lists) to reconstruct a ParticleData.
        """
        locs = spt_dict.get('localizations')
        if locs is None or (hasattr(locs, '__len__') and len(locs) == 0):
            return cls()

        locs = np.asarray(locs, dtype=np.float64)
        obj = cls.from_numpy(locs)

        # Apply track info if available
        tracks = spt_dict.get('tracks')
        if tracks is not None:
            obj.set_tracks(tracks, linking_params=spt_dict.get('linking_params'))

        obj._detection_params = spt_dict.get('detection_params', {})
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self):
        """The underlying DataFrame (mutable reference)."""
        return self._df

    @property
    def n_localizations(self):
        return len(self._df)

    @property
    def n_tracks(self):
        if 'track_id' not in self._df.columns:
            return 0
        valid = self._df['track_id']
        return int((valid >= 0).sum() and valid[valid >= 0].nunique())

    @property
    def n_frames(self):
        if len(self._df) == 0:
            return 0
        return int(self._df['frame'].nunique())

    # ------------------------------------------------------------------
    # Fast accessors (views, not copies)
    # ------------------------------------------------------------------

    def frame_locs(self, frame):
        """Return DataFrame rows for a single frame (view)."""
        if frame in self._frame_index:
            idx = self._frame_index[frame]
            return self._df.iloc[idx]
        return self._df.iloc[0:0]  # empty

    def track_locs(self, track_id):
        """Return DataFrame rows for a single track."""
        if 'track_id' not in self._df.columns:
            return self._df.iloc[0:0]
        mask = self._df['track_id'] == track_id
        return self._df.loc[mask]

    def track_ids(self):
        """Return sorted array of unique track IDs (excluding -1)."""
        if 'track_id' not in self._df.columns:
            return np.array([], dtype=np.int64)
        ids = self._df['track_id'].unique()
        ids = ids[ids >= 0]
        ids.sort()
        return ids

    def track_summary(self):
        """Return a DataFrame with one row per track: track_id, n_points,
        first_frame, last_frame, mean_intensity."""
        if 'track_id' not in self._df.columns or self.n_tracks == 0:
            return pd.DataFrame(columns=[
                'track_id', 'n_points', 'first_frame', 'last_frame',
                'mean_intensity'])
        linked = self._df[self._df['track_id'] >= 0]
        summary = linked.groupby('track_id').agg(
            n_points=('frame', 'count'),
            first_frame=('frame', 'min'),
            last_frame=('frame', 'max'),
            mean_intensity=('intensity', 'mean'),
        ).reset_index()
        return summary

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def set_localizations(self, locs_array, detection_params=None):
        """Replace all localizations from a numpy array.

        Parameters
        ----------
        locs_array : ndarray
            (N, C) array with columns [frame, x, y, intensity, ...].
        detection_params : dict, optional
            Parameters used for detection (stored for provenance).
        """
        new = ParticleData.from_numpy(locs_array)
        self._df = new._df
        self._frame_index = new._frame_index
        if detection_params:
            self._detection_params = dict(detection_params)

    def set_tracks(self, tracks_list, linking_params=None):
        """Assign track IDs from a list-of-lists of row indices.

        Parameters
        ----------
        tracks_list : list of list of int
            Each sub-list contains row indices belonging to one track.
        linking_params : dict, optional
            Parameters used for linking (stored for provenance).
        """
        self._df['track_id'] = np.int64(-1)
        for tid, indices in enumerate(tracks_list):
            valid = [i for i in indices if 0 <= i < len(self._df)]
            self._df.iloc[valid, self._df.columns.get_loc('track_id')] = np.int64(tid)
        if linking_params:
            self._linking_params = dict(linking_params)

    def set_features(self, features_by_track):
        """Broadcast per-track features to the localization DataFrame.

        Parameters
        ----------
        features_by_track : dict
            ``{track_id: {feature_name: value, ...}, ...}``
        """
        if 'track_id' not in self._df.columns:
            return
        # Collect all feature names
        all_names = set()
        for feat in features_by_track.values():
            all_names.update(feat.keys())
        all_names.discard('n_points')

        for name in sorted(all_names):
            self._df[name] = np.nan

        for tid, feat in features_by_track.items():
            mask = self._df['track_id'] == tid
            for name, value in feat.items():
                if name == 'n_points':
                    continue
                if name in self._df.columns:
                    self._df.loc[mask, name] = value

    def set_classification(self, classification_dict):
        """Set the 'classification' column from a {track_id: label} dict."""
        if 'track_id' not in self._df.columns:
            return
        self._df['classification'] = ''
        for tid, label in classification_dict.items():
            mask = self._df['track_id'] == int(tid)
            self._df.loc[mask, 'classification'] = str(label)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_thunderstorm_csv(self, path, pixel_size=108.0):
        """Export in ThunderSTORM-compatible CSV format.

        Coordinates are converted from pixels to nanometres.
        Frame numbering is converted to 1-based.
        """
        if self._df.empty:
            cols = ['id', 'frame', 'x [nm]', 'y [nm]', 'intensity [photon]']
            pd.DataFrame(columns=cols).to_csv(path, index=False)
            return

        out = pd.DataFrame()
        out['id'] = self._df['id'] + 1  # 1-based
        out['frame'] = self._df['frame'] + 1  # 1-based
        out['x [nm]'] = self._df['x'] * pixel_size
        out['y [nm]'] = self._df['y'] * pixel_size
        out['intensity [photon]'] = self._df['intensity']

        # Optional ThunderSTORM columns
        if 'sigma_x' in self._df.columns:
            out['sigma [nm]'] = self._df['sigma_x'] * pixel_size
        if 'uncertainty' in self._df.columns:
            out['uncertainty [nm]'] = self._df['uncertainty'] * pixel_size
        if 'background' in self._df.columns:
            out['offset [photon]'] = self._df['background']

        out.to_csv(path, index=False)
        logger.info("Exported %d localizations to ThunderSTORM CSV: %s",
                     len(out), path)

    def to_flika_csv(self, path):
        """Export in flika CSV format (all columns, pixel coords, 0-based)."""
        self._df.to_csv(path, index=False)
        logger.info("Exported %d localizations to flika CSV: %s",
                     len(self._df), path)

    def to_spt_dict(self):
        """Convert to legacy ``window.metadata['spt']`` format.

        Returns a dict with keys compatible with existing SPT consumers:
        'localizations', 'tracks', 'tracks_dict', 'detection_params',
        'linking_params'.
        """
        result = {}

        # Build localizations array: [frame, x, y, intensity]
        if len(self._df) > 0:
            base_cols = ['frame', 'x', 'y', 'intensity']
            if 'track_id' in self._df.columns:
                has_linked = (self._df['track_id'] >= 0).any()
                if has_linked:
                    base_cols.append('track_id')
            available = [c for c in base_cols if c in self._df.columns]
            result['localizations'] = self._df[available].values.astype(np.float64)
        else:
            result['localizations'] = np.empty((0, 4))

        # Build tracks list-of-lists
        if 'track_id' in self._df.columns:
            tracks = []
            tracks_dict = {}
            for tid in self.track_ids():
                mask = self._df['track_id'] == tid
                indices = self._df.index[mask].tolist()
                tracks.append(indices)
                # tracks_dict: {track_id: (N, 3) array [frame, x, y]}
                sub = self._df.loc[mask, ['frame', 'x', 'y']].values
                tracks_dict[int(tid)] = sub
            result['tracks'] = tracks
            result['tracks_dict'] = tracks_dict
        else:
            result['tracks'] = []
            result['tracks_dict'] = {}

        result['detection_params'] = dict(self._detection_params)
        result['linking_params'] = dict(self._linking_params)

        return result

    def to_dataframe(self):
        """Return a copy of the underlying DataFrame."""
        return self._df.copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_frame_index(self):
        """Build dict[frame] -> array of row positions for O(1) lookup."""
        self._frame_index = {}
        if len(self._df) == 0:
            return
        frames = self._df['frame'].values
        for frame_val in np.unique(frames):
            self._frame_index[int(frame_val)] = np.where(frames == frame_val)[0]

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        n_t = self.n_tracks
        return (f"ParticleData({self.n_localizations} localizations, "
                f"{n_t} tracks, {self.n_frames} frames)")
