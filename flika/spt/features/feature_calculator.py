"""Unified SPT feature computation pipeline.

Exact replica of the column layout and computation order from:
- locsAndTracksPlotter/joinTracks.py::calcFeaturesforFiles
- spt_batch_analysis::calculate_features + add_lag_features
"""
import numpy as np
import pandas as pd
import math
from .geometric import (radius_of_gyration, fractal_dimension,
                        net_displacement_efficiency, straightness,
                        get_scaled_rg, get_radius_of_gyration_simple,
                        radius_gyration_scaled, radius_gyration_scaled_nSegments,
                        radius_gyration_scaled_trackLength,
                        scaled_radius_of_gyration)
from .kinematic import (lag_displacements, velocity_analysis, msd_analysis,
                        direction_analysis, distance_from_origin,
                        add_diffusion_columns, add_velocity_columns,
                        add_track_intensity_stats)
from .spatial import nearest_neighbor_distances, neighbor_counts, analyze_frame_neighbors
from flika.logger import logger


class FeatureCalculator:
    """Computes all enabled features for a set of tracks.

    Replicates the exact output of joinTracks.py::calcFeaturesforFiles
    which produces these columns in this order:

        track_number, frame, id, x, y, intensity,
        n_segments, track_length, radius_gyration, asymmetry, skewness,
        kurtosis, radius_gyration_scaled, radius_gyration_scaled_nSegments,
        radius_gyration_scaled_trackLength, track_intensity_mean,
        track_intensity_std, lag, meanLag, fracDimension, netDispl, Straight,
        nnDist, nnIndex_inFrame, nnDist_inFrame

    Args:
        pixel_size: nm per pixel (for converting to physical units)
        frame_interval: seconds between frames
        min_segments: minimum number of segments (track points) to keep
        enable_geometric: compute Rg, asymmetry, fractal dim, etc.
        enable_kinematic: compute velocity, MSD, diffusion, direction
        enable_spatial: compute nearest neighbors, density
    """

    def __init__(self, pixel_size=108.0, frame_interval=1.0,
                 min_segments=1,
                 enable_geometric=True, enable_kinematic=True,
                 enable_spatial=True):
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        self.min_segments = min_segments
        self.enable_geometric = enable_geometric
        self.enable_kinematic = enable_kinematic
        self.enable_spatial = enable_spatial

    # ------------------------------------------------------------------
    # Per-track feature computation (tensor method)
    # ------------------------------------------------------------------

    def compute_track_features(self, positions):
        """Compute all enabled features for a single track.

        Args:
            positions: (N, 2) array of [x, y] pixel positions
        Returns:
            dict of feature_name -> value
        """
        positions = np.asarray(positions, dtype=np.float64)
        features = {'n_points': len(positions)}

        if self.enable_geometric and len(positions) >= 3:
            rg = radius_of_gyration(positions)
            features.update({
                'radius_gyration': rg['rg'],
                'asymmetry': rg['asymmetry'],
                'skewness': rg['skewness'],
                'kurtosis': rg['kurtosis'],
            })
            features['fracDimension'] = fractal_dimension(positions)

            ne = net_displacement_efficiency(positions)
            features['netDispl'] = ne['net_displacement']
            features['efficiency'] = ne['efficiency']

            st = straightness(positions)
            features['Straight'] = st['straightness']

            # Track intensity is computed separately per-track (see below)

        if self.enable_kinematic and len(positions) >= 3:
            vel = velocity_analysis(positions, dt=self.frame_interval)
            features['mean_velocity'] = vel['mean_velocity']
            features['max_velocity'] = vel['max_velocity']

            msd = msd_analysis(positions)
            features['diffusion_coefficient'] = msd['diffusion_coefficient']
            features['anomalous_exponent'] = msd['anomalous_exponent']

            dirn = direction_analysis(positions)
            features['directional_persistence'] = dirn['directional_persistence']
            features['mean_direction'] = dirn['mean_direction']

        return features

    # ------------------------------------------------------------------
    # Full pipeline: exact replica of joinTracks.py::calcFeaturesforFiles
    # ------------------------------------------------------------------

    def compute_all_from_df(self, tracksDF, min_segments=None):
        """Compute all features on a DataFrame with track_number, frame, x, y, intensity columns.

        This is the primary entry point that replicates the exact pipeline from
        joinTracks.py::calcFeaturesforFiles:
            1. Add n_segments, filter by minNumberSegments
            2. getRadiusGyrationForAllTracksinDF (Rg, asymmetry, skewness, kurtosis,
               track_intensity_mean, track_intensity_std)
            3. getFeaturesForAllTracksinDF (fracDimension, netDispl, Straight)
            4. addLagDisplacementToDF (lag, meanLag, track_length,
               radius_gyration_scaled, radius_gyration_scaled_nSegments,
               radius_gyration_scaled_trackLength)
            5. Select and round columns

        Args:
            tracksDF: DataFrame with at least track_number, frame, x, y, intensity
            min_segments: override self.min_segments if provided

        Returns:
            DataFrame with all feature columns, rounded.
        """
        if min_segments is None:
            min_segments = self.min_segments

        tracksDF = tracksDF.copy()

        # --- Step 1: n_segments and filtering ---
        tracksDF['n_segments'] = tracksDF.groupby('track_number')['track_number'].transform('count')

        if min_segments != 0:
            tracksDF = tracksDF[tracksDF['n_segments'] > min_segments].copy()

        if len(tracksDF) == 0:
            return tracksDF

        # --- Step 2: Radius of gyration, asymmetry, skewness, kurtosis,
        #             track_intensity_mean, track_intensity_std ---
        tracksToTest = tracksDF['track_number'].tolist()
        idTested = []
        radius_gyration_list = []
        asymmetry_list = []
        skewness_list = []
        kurtosis_list = []
        trackIntensity_mean = []
        trackIntensity_std = []

        for i in range(len(tracksToTest)):
            idToTest = tracksToTest[i]
            if idToTest not in idTested:
                trackDF = tracksDF[tracksDF['track_number'] == idToTest]
                rg_result = radius_of_gyration(np.array(trackDF[['x', 'y']].dropna()))
                rg_val = rg_result['rg']
                asym_val = rg_result['asymmetry']
                skew_val = rg_result['skewness']
                kurt_val = rg_result['kurtosis']
                idTested.append(idToTest)

            radius_gyration_list.append(rg_val)
            asymmetry_list.append(asym_val)
            skewness_list.append(skew_val)
            kurtosis_list.append(kurt_val)

            trackIntensity_mean.append(
                float(np.mean(tracksDF[tracksDF['track_number'] == idToTest]['intensity']))
            )
            trackIntensity_std.append(
                float(np.std(tracksDF[tracksDF['track_number'] == idToTest]['intensity']))
            )

        tracksDF['radius_gyration'] = radius_gyration_list
        tracksDF['asymmetry'] = asymmetry_list
        tracksDF['skewness'] = skewness_list
        tracksDF['kurtosis'] = kurtosis_list
        tracksDF['track_intensity_mean'] = trackIntensity_mean
        tracksDF['track_intensity_std'] = trackIntensity_std

        # --- Step 3: Fractal dimension, net displacement, straightness ---
        idTested = []
        fracDim_list = []
        netDispl_list = []
        straight_list = []

        for i in range(len(tracksToTest)):
            idToTest = tracksToTest[i]
            if idToTest not in idTested:
                trackDF = tracksDF[tracksDF['track_number'] == idToTest]
                points_array = np.array(trackDF[['x', 'y']].dropna())

                fractal_dimension_value = fractal_dimension(points_array)
                ne = net_displacement_efficiency(points_array)
                net_displacement_value = ne['net_displacement']
                st = straightness(points_array)
                cos_mean_val = st['straightness']

                idTested.append(idToTest)

            fracDim_list.append(fractal_dimension_value)
            netDispl_list.append(net_displacement_value)
            straight_list.append(cos_mean_val)

        tracksDF['fracDimension'] = fracDim_list
        tracksDF['netDispl'] = netDispl_list
        tracksDF['Straight'] = straight_list

        # --- Step 4: Lag displacement features ---
        # Exact replica of joinTracks.py::addLagDisplacementToDF
        tracksDF = tracksDF.assign(x2=tracksDF.x.shift(-1))
        tracksDF = tracksDF.assign(y2=tracksDF.y.shift(-1))

        tracksDF['x2-x1_sqr'] = np.square(tracksDF['x2'] - tracksDF['x'])
        tracksDF['y2-y1_sqr'] = np.square(tracksDF['y2'] - tracksDF['y'])
        tracksDF['distance'] = np.sqrt(tracksDF['x2-x1_sqr'] + tracksDF['y2-y1_sqr'])

        # Mask final track position lags
        tracksDF['mask'] = True
        tracksDF.loc[tracksDF.groupby('track_number').tail(1).index, 'mask'] = False

        # Get lags for all track locations (not next track)
        tracksDF['lag'] = tracksDF['distance'].where(tracksDF['mask'])

        # Add track mean lag distance to all rows
        tracksDF['meanLag'] = tracksDF.groupby('track_number')['lag'].transform('mean')

        # Add track length for each track row
        tracksDF['track_length'] = tracksDF.groupby('track_number')['lag'].transform('sum')

        # Add scaled Rg variants
        tracksDF['radius_gyration_scaled'] = tracksDF['radius_gyration'] / tracksDF['meanLag']
        tracksDF['radius_gyration_scaled_nSegments'] = tracksDF['radius_gyration'] / tracksDF['n_segments']
        tracksDF['radius_gyration_scaled_trackLength'] = tracksDF['radius_gyration'] / tracksDF['track_length']

        # Clean up temporary columns
        tracksDF = tracksDF.drop(
            columns=['x2', 'y2', 'x2-x1_sqr', 'y2-y1_sqr', 'distance', 'mask'],
            errors='ignore'
        )

        # --- Step 5: Select and round columns ---
        # Build column list (only include columns that exist)
        desired_columns = [
            'track_number', 'frame', 'id', 'x', 'y', 'intensity',
            'n_segments', 'track_length', 'radius_gyration', 'asymmetry',
            'skewness', 'kurtosis',
            'radius_gyration_scaled', 'radius_gyration_scaled_nSegments',
            'radius_gyration_scaled_trackLength',
            'track_intensity_mean', 'track_intensity_std',
            'lag', 'meanLag', 'fracDimension', 'netDispl', 'Straight',
            'nnDist', 'nnIndex_inFrame', 'nnDist_inFrame',
        ]
        available_columns = [c for c in desired_columns if c in tracksDF.columns]
        tracksDF = tracksDF[available_columns]

        # Round values -- exact plugin rounding
        round_map = {
            'track_length': 3,
            'radius_gyration': 3,
            'asymmetry': 3,
            'skewness': 3,
            'kurtosis': 3,
            'radius_gyration_scaled': 3,
            'radius_gyration_scaled_nSegments': 3,
            'radius_gyration_scaled_trackLength': 3,
            'track_intensity_mean': 2,
            'track_intensity_std': 2,
            'lag': 3,
            'meanLag': 3,
            'fracDimension': 3,
            'netDispl': 3,
            'Straight': 3,
            'nnDist': 3,
        }
        # Only round columns that exist
        round_existing = {k: v for k, v in round_map.items() if k in tracksDF.columns}
        tracksDF = tracksDF.round(round_existing)

        return tracksDF

    # ------------------------------------------------------------------
    # Older API -- compute from raw arrays (kept for backward compat)
    # ------------------------------------------------------------------

    def compute_all(self, locs_array, tracks):
        """Compute features for all tracks from raw arrays.

        Args:
            locs_array: (N, 3+) array [frame, x, y, ...]
            tracks: list of lists of row indices
        Returns:
            DataFrame with one row per track, columns are feature names
        """
        locs = np.asarray(locs_array, dtype=np.float64)
        rows = []

        for tid, track_indices in enumerate(tracks):
            positions = locs[track_indices][:, 1:3]  # x, y
            features = self.compute_track_features(positions)
            features['track_id'] = tid
            rows.append(features)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        cols = ['track_id'] + [c for c in df.columns if c != 'track_id']
        return df[cols]

    def compute_per_point(self, locs_array, tracks):
        """Compute per-point features with track-level features replicated.

        Returns DataFrame with one row per detection point.
        """
        locs = np.asarray(locs_array, dtype=np.float64)

        # First compute track-level features
        track_features = self.compute_all(locs, tracks)

        # Build per-point dataframe
        rows = []
        for tid, track_indices in enumerate(tracks):
            track_locs = locs[track_indices]
            positions = track_locs[:, 1:3]

            # Per-point features
            lags = lag_displacements(positions)
            dists = distance_from_origin(positions)

            for i, idx in enumerate(track_indices):
                row = {
                    'track_id': tid,
                    'frame': int(locs[idx, 0]),
                    'x': float(locs[idx, 1]),
                    'y': float(locs[idx, 2]),
                }
                if locs.shape[1] > 3:
                    row['intensity'] = float(locs[idx, 3])
                row['distance_from_origin'] = float(dists[i])
                row['lag'] = float(lags[i]) if i < len(lags) else np.nan

                # Add track-level features
                if tid < len(track_features):
                    tf = track_features.iloc[tid]
                    for col in track_features.columns:
                        if col not in row:
                            row[col] = tf[col]

                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
