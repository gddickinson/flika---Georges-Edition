"""Kinematic track features: velocity, MSD, diffusion, direction.

Exact replica of the algorithms from:
- locsAndTracksPlotter/joinTracks.py (addVelocitytoDF, addDiffusiontoDF,
  addLagDisplacementToDF)
- spt_batch_analysis (add_velocity_analysis, add_diffusion_analysis,
  add_lag_features)
"""
import numpy as np
import math
from ...logger import logger


# ---------------------------------------------------------------------------
# Lag displacements -- core building block
# ---------------------------------------------------------------------------

def lag_displacements(positions):
    """Frame-to-frame step distances.

    Args:
        positions: (N, 2) array
    Returns:
        (N-1,) array of step distances
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return np.array([])
    return np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))


# ---------------------------------------------------------------------------
# Velocity -- exact joinTracks.py::addVelocitytoDF
# ---------------------------------------------------------------------------

def velocity_analysis(positions, dt=1.0):
    """Velocity analysis for a track.

    Args:
        positions: (N, 2) array
        dt: time interval between frames
    Returns:
        dict with: instantaneous_velocities (array), mean_velocity, max_velocity
    """
    lags = lag_displacements(positions)
    if len(lags) == 0:
        return {'instantaneous_velocities': np.array([]), 'mean_velocity': 0.0, 'max_velocity': 0.0}

    velocities = lags / dt
    return {
        'instantaneous_velocities': velocities,
        'mean_velocity': float(np.mean(velocities)),
        'max_velocity': float(np.max(velocities)),
    }


def add_velocity_columns(df, frame_length=1.0):
    """Add gap-aware velocity columns to a tracks DataFrame.

    Exact replica of joinTracks.py::addVelocitytoDF and
    spt_batch_analysis::add_velocity_analysis.

    Expects the DataFrame to already contain columns:
        track_number, frame, x, y, lag, zeroed_X, zeroed_Y,
        distanceFromOrigin, lagNumber

    Adds columns:
        dt, velocity, direction_Relative_To_Origin, meanVelocity

    Args:
        df: tracks DataFrame (modified in place and returned)
        frame_length: seconds per frame (used for dt computation)

    Returns:
        DataFrame with velocity columns added.
    """
    import pandas as pd

    df = df.sort_values(['track_number', 'frame']).copy()

    # dt accounts for frame gaps: dt = diff(frame) * frame_length
    # Exact plugin: newDF['dt'] = shift(frame) - frame, then mask non-positive
    df['dt'] = df.groupby('track_number')['frame'].diff() * frame_length
    # Mask invalid dt (NaN at first row of each track is fine)

    # instantaneous velocity = lag / dt (NOT just lag / fixed_dt)
    df['velocity'] = df['lag'] / df['dt']

    # direction relative to 0,0 origin : 360 degrees
    # Exact plugin: np.arctan2(zeroed_Y, zeroed_X)/np.pi*180, negative -> +360
    if 'zeroed_X' in df.columns and 'zeroed_Y' in df.columns:
        degrees = np.arctan2(df['zeroed_Y'].to_numpy(), df['zeroed_X'].to_numpy()) / np.pi * 180
        degrees[degrees < 0] = 360 + degrees[degrees < 0]
        df['direction_Relative_To_Origin'] = degrees

    # add mean track velocity
    df['meanVelocity'] = df.groupby('track_number')['velocity'].transform('mean')

    return df


# ---------------------------------------------------------------------------
# Diffusion columns -- exact joinTracks.py::addDiffusiontoDF
# ---------------------------------------------------------------------------

def add_diffusion_columns(df):
    """Add diffusion-related columns to a tracks DataFrame.

    Exact replica of joinTracks.py::addDiffusiontoDF and
    spt_batch_analysis::add_velocity_analysis + add_diffusion_analysis.

    Adds columns:
        zeroed_X, zeroed_Y, lagNumber, distanceFromOrigin,
        d_squared, lag_squared

    Args:
        df: tracks DataFrame with columns track_number, frame, x, y, lag.
            Modified in place and returned.

    Returns:
        DataFrame with diffusion columns added.
    """
    import pandas as pd

    df = df.sort_values(['track_number', 'frame']).copy()

    # Initialise columns
    df['zeroed_X'] = np.nan
    df['zeroed_Y'] = np.nan
    df['lagNumber'] = np.nan
    df['distanceFromOrigin'] = np.nan

    for track_num in df['track_number'].unique():
        mask = df['track_number'] == track_num
        track_data = df.loc[mask]

        if len(track_data) == 0:
            continue

        # Origin at first position
        minFrame = track_data['frame'].min()
        origin_X = float(track_data.loc[track_data['frame'] == minFrame, 'x'].iloc[0])
        origin_Y = float(track_data.loc[track_data['frame'] == minFrame, 'y'].iloc[0])

        df.loc[mask, 'zeroed_X'] = track_data['x'] - origin_X
        df.loc[mask, 'zeroed_Y'] = track_data['y'] - origin_Y

        # Generate lag numbers (frame offset from first frame)
        df.loc[mask, 'lagNumber'] = track_data['frame'] - minFrame

        # Distance from origin
        df.loc[mask, 'distanceFromOrigin'] = np.sqrt(
            np.square(df.loc[mask, 'zeroed_X']) + np.square(df.loc[mask, 'zeroed_Y'])
        )

    # Squared values -- exact plugin
    df['d_squared'] = np.square(df['distanceFromOrigin'])
    df['lag_squared'] = np.square(df['lag'])

    return df


# ---------------------------------------------------------------------------
# Track intensity statistics -- exact joinTracks.py::getRadiusGyrationForAllTracksinDF
# ---------------------------------------------------------------------------

def add_track_intensity_stats(df):
    """Add track intensity mean and std columns.

    Exact replica of joinTracks.py which computes:
        trackIntensity_mean = mean(intensity) per track
        trackIntensity_std  = std(intensity)  per track
    and broadcasts to all rows of that track.

    Args:
        df: tracks DataFrame with columns track_number, intensity.

    Returns:
        DataFrame with track_intensity_mean and track_intensity_std added.
    """
    df = df.copy()
    df['track_intensity_mean'] = df.groupby('track_number')['intensity'].transform('mean')
    df['track_intensity_std'] = df.groupby('track_number')['intensity'].transform('std')
    return df


# ---------------------------------------------------------------------------
# MSD analysis
# ---------------------------------------------------------------------------

def msd_analysis(positions, max_lag=None, min_points=5):
    """Mean Squared Displacement analysis.

    Args:
        positions: (N, 2) array of positions
        max_lag: maximum lag in frames (default: N//4)
        min_points: minimum points for fitting

    Returns:
        dict with: msd_curve (array of (lag, msd) pairs),
                   diffusion_coefficient (D from linear fit, 2D: MSD = 4*D*t),
                   anomalous_exponent (alpha from log-log fit: MSD ~ t^alpha)
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)

    if n < min_points:
        return {'msd_curve': np.array([]), 'diffusion_coefficient': 0.0, 'anomalous_exponent': 1.0}

    if max_lag is None:
        max_lag = max(n // 4, 2)
    max_lag = min(max_lag, n - 1)

    lags = np.arange(1, max_lag + 1)
    msd_values = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        displacements = positions[lag:] - positions[:-lag]
        sq_displacements = np.sum(displacements ** 2, axis=1)
        msd_values[i] = np.mean(sq_displacements)

    msd_curve = np.column_stack([lags, msd_values])

    # Linear fit for diffusion coefficient: MSD = 4*D*t (2D)
    D = 0.0
    if len(lags) >= 2:
        fit_n = min(len(lags), 10)
        coeffs = np.polyfit(lags[:fit_n].astype(float), msd_values[:fit_n], 1)
        D = coeffs[0] / 4.0
        D = max(D, 0.0)

    # Log-log fit for anomalous exponent: MSD ~ t^alpha
    alpha = 1.0
    if len(lags) >= 3:
        valid = msd_values > 0
        if np.sum(valid) >= 2:
            log_lags = np.log(lags[valid].astype(float))
            log_msd = np.log(msd_values[valid])
            coeffs = np.polyfit(log_lags, log_msd, 1)
            alpha = float(coeffs[0])

    return {
        'msd_curve': msd_curve,
        'diffusion_coefficient': float(D),
        'anomalous_exponent': float(alpha),
    }


# ---------------------------------------------------------------------------
# Direction analysis
# ---------------------------------------------------------------------------

def direction_analysis(positions):
    """Direction of travel analysis.

    Args:
        positions: (N, 2) array
    Returns:
        dict with: directions_deg (N-1 array, 0-360), directions_rad,
                   direction_changes, mean_direction, directional_persistence
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return {'directions_deg': np.array([]), 'directions_rad': np.array([]),
                'direction_changes': np.array([]), 'mean_direction': 0.0,
                'directional_persistence': 0.0}

    steps = np.diff(positions, axis=0)
    dx = steps[:, 0]
    dy = -steps[:, 1]  # Invert Y for microscopy convention

    # Angles in radians and degrees
    angles_rad = np.arctan2(dy, dx)
    angles_deg = np.degrees(angles_rad)
    angles_deg[angles_deg < 0] += 360.0

    # Direction changes
    if len(angles_deg) >= 2:
        diffs = np.abs(np.diff(angles_deg))
        diffs[diffs > 180] = 360 - diffs[diffs > 180]
        direction_changes = diffs
    else:
        direction_changes = np.array([])

    # Directional persistence (0=random, 1=straight)
    if len(direction_changes) > 0:
        mean_change = np.nanmean(direction_changes)
        persistence = 1.0 - (mean_change / 180.0)
    else:
        persistence = 0.0

    return {
        'directions_deg': angles_deg,
        'directions_rad': angles_rad,
        'direction_changes': direction_changes,
        'mean_direction': float(np.mean(angles_deg)) if len(angles_deg) > 0 else 0.0,
        'directional_persistence': float(persistence),
    }


# ---------------------------------------------------------------------------
# Distance from origin
# ---------------------------------------------------------------------------

def distance_from_origin(positions):
    """Compute distance from track start at each point.

    Returns: (N,) array of distances from first position.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) == 0:
        return np.array([])
    origin = positions[0]
    return np.sqrt(np.sum((positions - origin) ** 2, axis=1))
