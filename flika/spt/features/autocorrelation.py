"""Directional persistence analysis via velocity autocorrelation.

Exact replica of spt_batch_analysis::AutocorrelationAnalyzer which
implements the method described in Gorelik & Gautreau (2014) for
quantifying directional persistence in particle trajectories.

Algorithm (faithful to the original plugin):
    1. Detect trajectory segment boundaries by frame monotonicity:
       wherever frame[i+1] <= frame[i], a new segment starts.
    2. Compute normalized displacement vectors (dx, dy) / magnitude
       for each consecutive pair of positions within a segment.
    3. For each step size (time lag), compute dot products between
       normalized vectors separated by that many steps.
    4. Aggregate: mean + SEM for each time interval.

Reference: Gorelik, R. & Gautreau, A. (2014). Quantitative and
unbiased analysis of directional persistence in cell migration.
Nature Protocols, 9(8), 1931-1943.
"""
import numpy as np
import pandas as pd
from flika.logger import logger


class AutocorrelationAnalyzer:
    """Velocity autocorrelation analysis for directional persistence.

    Exact replica of spt_batch_analysis::AutocorrelationAnalyzer, using
    the same three-step algorithm:
        calculate_normed_vectors -> calculate_scalar_products -> calculate_averages

    Args:
        n_intervals: Maximum number of time-lag intervals to compute.
        min_track_length: Minimum number of points a track must have
            to be included in the analysis.
        time_interval: Time interval multiplier (default 1).
    """

    def __init__(self, n_intervals=20, min_track_length=10, time_interval=1):
        self.n_intervals = n_intervals
        self.min_track_length = min_track_length
        self.time_interval = time_interval

    # ------------------------------------------------------------------
    # Step 1: Normalized displacement vectors (exact plugin code)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_normed_vectors(df):
        """Calculate normalized vectors for each step in the trajectory.

        Exact replica of spt_batch_analysis::AutocorrelationAnalyzer.calculate_normed_vectors.

        Detects trajectory segment boundaries via frame monotonicity:
        wherever frame[i] <= frame[i-1], a new segment starts.

        Args:
            df: DataFrame with columns 'frame', 'x', 'y'

        Returns:
            (result_df, traj_starts) where result_df has added columns
            'x_vector' and 'y_vector', and traj_starts is a list of
            segment start indices (plus a sentinel at the end).
        """
        result_df = df.copy()
        result_df['x_vector'] = np.nan
        result_df['y_vector'] = np.nan

        # Use pandas diff to find frame discontinuities
        frame_diff = df['frame'].diff()
        new_traj_mask = (frame_diff <= 0).copy()
        new_traj_mask.iloc[0] = True

        # Find trajectory start indices
        traj_starts = list(new_traj_mask[new_traj_mask].index)
        traj_starts.append(len(df))

        # Process each trajectory segment
        for i in range(len(traj_starts) - 1):
            start_idx = traj_starts[i]
            end_idx = traj_starts[i + 1]

            if end_idx - start_idx < 2:
                continue

            # Get the trajectory segment
            traj = result_df.iloc[start_idx:end_idx]

            # Calculate differences between consecutive points
            # diff(-1) computes x[i] - x[i+1]
            dx = traj['x'].diff(-1).iloc[:-1]
            dy = traj['y'].diff(-1).iloc[:-1]

            # Calculate magnitudes
            magnitudes = np.sqrt(dx ** 2 + dy ** 2)

            # Find valid movements (non-zero magnitude)
            valid_moves = magnitudes > 0

            # Normalize vectors where magnitude > 0
            if len(dx) > 0:
                result_df.iloc[start_idx + 1:end_idx,
                               result_df.columns.get_loc('x_vector')] = \
                    np.where(valid_moves, dx / magnitudes, np.nan)
                result_df.iloc[start_idx + 1:end_idx,
                               result_df.columns.get_loc('y_vector')] = \
                    np.where(valid_moves, dy / magnitudes, np.nan)

        return result_df, traj_starts

    # ------------------------------------------------------------------
    # Step 2: Scalar products (exact plugin code)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_scalar_products(df, traj_starts, time_interval, num_intervals):
        """Calculate scalar products of vectors for different time intervals.

        Exact replica of spt_batch_analysis::AutocorrelationAnalyzer.calculate_scalar_products.

        Args:
            df: DataFrame with 'x_vector', 'y_vector' columns
            traj_starts: list of trajectory segment start indices
            time_interval: base time interval multiplier
            num_intervals: number of lag intervals to compute

        Returns:
            (combined_df, tracks_df) where combined_df has one column per
            time point with all individual dot products, and tracks_df has
            per-track average correlations.
        """
        combined_scalar_results = {
            time_interval * step: [] for step in range(1, num_intervals + 1)
        }
        individual_track_results = {}

        for i in range(len(traj_starts) - 1):
            start_idx = traj_starts[i]
            end_idx = traj_starts[i + 1]
            traj_length = end_idx - start_idx

            if traj_length < 2:
                continue

            track_id = f"track_{i + 1}"
            individual_track_results[track_id] = {}

            max_intervals = min(num_intervals, traj_length)
            traj_vectors = df.iloc[start_idx:end_idx]

            # For each step size
            for step in range(1, max_intervals):
                time_point = time_interval * step

                x_vecs1 = traj_vectors['x_vector'].values[:-step]
                y_vecs1 = traj_vectors['y_vector'].values[:-step]
                x_vecs2 = traj_vectors['x_vector'].values[step:]
                y_vecs2 = traj_vectors['y_vector'].values[step:]

                # Calculate dot products
                dot_products = x_vecs1 * x_vecs2 + y_vecs1 * y_vecs2

                # Filter valid values
                valid_mask = ~np.isnan(dot_products)
                valid_dots = dot_products[valid_mask]

                # Add to combined results
                combined_scalar_results[time_point].extend(valid_dots.tolist())

                # Store for this individual track
                if len(valid_dots) > 0:
                    track_avg_corr = float(np.mean(valid_dots))
                    individual_track_results[track_id][time_point] = track_avg_corr

        # Convert results to DataFrame
        combined_df = pd.DataFrame(
            {k: pd.Series(v) for k, v in combined_scalar_results.items()}
        )

        # Convert individual track results
        track_data = []
        for track_id, time_points in individual_track_results.items():
            for time_point, corr in time_points.items():
                track_data.append({
                    'track_id': track_id,
                    'time_interval': time_point,
                    'correlation': corr,
                })

        tracks_df = pd.DataFrame(track_data)

        return combined_df, tracks_df

    # ------------------------------------------------------------------
    # Step 3: Averages and SEM (exact plugin code)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_averages(scalar_products):
        """Calculate averages and standard errors for each time interval.

        Exact replica of spt_batch_analysis::AutocorrelationAnalyzer.calculate_averages.

        Returns:
            DataFrame with index ['AVG', 'SEM'] and one column per
            time point. Column 0 is always [1, 0] (perfect correlation
            at dt=0).
        """
        results = pd.DataFrame(index=['AVG', 'SEM'])
        results[0] = [1, 0]  # Perfect correlation at time=0

        for col in scalar_products.columns:
            values = scalar_products[col].dropna()

            avg = values.mean()
            n = len(values)
            sem = values.std() / np.sqrt(n) if n > 0 else 0

            results[col] = [avg, sem]

        return results

    # ------------------------------------------------------------------
    # High-level API: process track data (exact plugin pipeline)
    # ------------------------------------------------------------------

    def compute(self, tracks_dict):
        """Compute velocity autocorrelation for all tracks.

        This is the new-style API that accepts a dict of tracks.

        Args:
            tracks_dict: ``{track_id: (N, 3) array}`` mapping track IDs
                to arrays with columns ``[frame, x, y]``.  Tracks shorter
                than :attr:`min_track_length` are silently skipped.

        Returns:
            dict with keys:
                - time_lags, mean_correlation, sem_correlation,
                  std_correlation, n_tracks_used, n_tracks_skipped,
                  individual_tracks
        """
        if not tracks_dict:
            return self._empty_result()

        # Filter tracks by minimum length
        qualifying = {}
        n_skipped = 0
        for tid, pts in tracks_dict.items():
            pts = np.asarray(pts, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if len(pts) >= self.min_track_length:
                qualifying[tid] = pts
            else:
                n_skipped += 1

        if not qualifying:
            logger.info("AutocorrelationAnalyzer: no tracks meet min_length=%d "
                        "(skipped %d)", self.min_track_length, n_skipped)
            return self._empty_result()

        # Determine the effective number of lag intervals
        max_possible = max(len(pts) - 2 for pts in qualifying.values())
        n_lags = min(self.n_intervals, max_possible)
        if n_lags < 1:
            return self._empty_result()

        # Compute per-track autocorrelation
        individual = []
        for tid, pts in qualifying.items():
            if pts.shape[1] >= 3:
                positions = pts[:, 1:3]
            else:
                positions = pts[:, :2]

            ac = self._track_autocorrelation(positions, n_lags)
            individual.append(ac)

        ac_matrix = np.array(individual)  # shape (n_tracks, n_lags)

        # Compute statistics, handling NaN values
        time_lags = np.arange(1, n_lags + 1)
        mean_corr = np.full(n_lags, np.nan)
        sem_corr = np.full(n_lags, np.nan)
        std_corr = np.full(n_lags, np.nan)

        for i in range(n_lags):
            col = ac_matrix[:, i]
            valid = ~np.isnan(col)
            n_valid = np.sum(valid)
            if n_valid > 0:
                vals = col[valid]
                mean_corr[i] = np.mean(vals)
                if n_valid > 1:
                    std_val = np.std(vals, ddof=1)
                    std_corr[i] = std_val
                    sem_corr[i] = std_val / np.sqrt(n_valid)
                else:
                    std_corr[i] = 0.0
                    sem_corr[i] = 0.0

        logger.info("AutocorrelationAnalyzer: %d tracks analysed, "
                     "%d skipped, %d lag intervals",
                     len(qualifying), n_skipped, n_lags)

        return {
            'time_lags': time_lags,
            'mean_correlation': mean_corr,
            'sem_correlation': sem_corr,
            'std_correlation': std_corr,
            'n_tracks_used': len(qualifying),
            'n_tracks_skipped': n_skipped,
            'individual_tracks': individual,
        }

    def process_track_data(self, tracks_df):
        """Process track data for autocorrelation analysis using the exact plugin pipeline.

        Exact replica of spt_batch_analysis::AutocorrelationAnalyzer.process_track_data.

        Args:
            tracks_df: DataFrame with columns track_number, frame, x, y

        Returns:
            (scalar_products, averages, individual_tracks) or (None, None, None)
        """
        # Group by track number and sort by frame
        grouped_tracks = []

        for track_num in tracks_df['track_number'].unique():
            track_data = tracks_df[tracks_df['track_number'] == track_num].sort_values('frame')

            if len(track_data) >= 3:  # Need minimum track length
                track_subset = track_data[['frame', 'x', 'y']].copy()
                grouped_tracks.append(track_subset)

        if not grouped_tracks:
            return None, None, None

        # Combine all tracks into single dataframe for vector analysis
        combined_df = pd.concat(grouped_tracks, ignore_index=True)

        # Calculate normalized vectors
        vectors_df, traj_starts = self.calculate_normed_vectors(combined_df)

        # Calculate scalar products
        scalar_products, individual_tracks = self.calculate_scalar_products(
            vectors_df, traj_starts, self.time_interval, self.n_intervals
        )

        # Calculate averages
        averages = self.calculate_averages(scalar_products)

        return scalar_products, averages, individual_tracks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _track_autocorrelation(self, positions, n_intervals):
        """Compute velocity autocorrelation for a single track.

        For each time lag dt from 1 to *n_intervals*, computes the
        average cosine of the angle between the displacement vector at
        time t and the displacement vector at time t+dt.

        This uses normalized displacement vectors (unit vectors) and
        dot products, exactly matching the plugin's scalar product method.

        Args:
            positions: (N, 2) array of [x, y] positions
            n_intervals: Number of lag intervals to compute

        Returns:
            (n_intervals,) array of autocorrelation values
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)

        displacements = np.diff(positions, axis=0)  # (n-1, 2)
        n_disp = len(displacements)

        autocorr = np.full(n_intervals, np.nan)

        for dt in range(1, n_intervals + 1):
            if dt >= n_disp:
                break

            v1 = displacements[:n_disp - dt]
            v2 = displacements[dt:]

            if len(v1) == 0:
                break

            mag1 = np.sqrt(np.sum(v1 ** 2, axis=1))
            mag2 = np.sqrt(np.sum(v2 ** 2, axis=1))

            valid = (mag1 > 1e-15) & (mag2 > 1e-15)
            if not np.any(valid):
                autocorr[dt - 1] = 0.0
                continue

            dot = np.sum(v1[valid] * v2[valid], axis=1)
            cos_theta = dot / (mag1[valid] * mag2[valid])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            autocorr[dt - 1] = float(np.mean(cos_theta))

        return autocorr

    def compute_single_track(self, positions):
        """Convenience method to compute autocorrelation for one track.

        Args:
            positions: (N, 2) array of [x, y] positions.

        Returns:
            dict with *time_lags* and *correlation* arrays
        """
        positions = np.asarray(positions, dtype=np.float64)
        if len(positions) < self.min_track_length:
            return {'time_lags': np.array([]), 'correlation': np.array([])}

        max_possible = len(positions) - 2
        n_lags = min(self.n_intervals, max_possible)
        if n_lags < 1:
            return {'time_lags': np.array([]), 'correlation': np.array([])}

        ac = self._track_autocorrelation(positions, n_lags)
        time_lags = np.arange(1, n_lags + 1)

        # Trim trailing NaN values
        valid_mask = ~np.isnan(ac)
        if np.any(valid_mask):
            last_valid = np.where(valid_mask)[0][-1] + 1
            ac = ac[:last_valid]
            time_lags = time_lags[:last_valid]

        return {
            'time_lags': time_lags,
            'correlation': ac,
        }

    def persistence_index(self, tracks_dict, lag=1):
        """Compute a single persistence index for a set of tracks.

        The persistence index is the mean autocorrelation at the
        specified lag.

        Args:
            tracks_dict: Track dict as for :meth:`compute`.
            lag: Time lag at which to evaluate (default 1).

        Returns:
            float persistence index, or NaN if no tracks qualify.
        """
        result = self.compute(tracks_dict)
        if result['n_tracks_used'] == 0:
            return float('nan')

        idx = lag - 1
        if idx < 0 or idx >= len(result['mean_correlation']):
            return float('nan')

        return float(result['mean_correlation'][idx])

    def _empty_result(self):
        """Return an empty results dict."""
        return {
            'time_lags': np.array([]),
            'mean_correlation': np.array([]),
            'sem_correlation': np.array([]),
            'std_correlation': np.array([]),
            'n_tracks_used': 0,
            'n_tracks_skipped': 0,
            'individual_tracks': [],
        }
