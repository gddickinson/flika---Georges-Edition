"""Fiducial-based drift correction for single-molecule localization.

Identifies bright, persistent fiducial markers (e.g. gold nanoparticles,
TetraSpeck beads) and uses their trajectories to estimate and correct
sample drift with sub-pixel accuracy.

The algorithm:
1. Identify fiducial candidates by brightness and presence across frames.
2. Track fiducials using nearest-neighbor linking.
3. Compute drift as the mean displacement of all fiducials per frame.
4. Smooth drift trajectory with cubic spline.
5. Subtract drift from all localizations.

References:
    Mlodzianoski et al., Opt Express 19:15024 (2011).
    Schnitzbauer et al., Nat Protoc 12:1198 (2017) — Picasso.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline


def identify_fiducials(localizations, n_frames, min_presence=0.8,
                       brightness_percentile=95, max_fiducials=10):
    """Identify fiducial markers from localization data.

    Fiducials are selected as the brightest localizations that appear in
    at least ``min_presence`` fraction of all frames.

    Parameters
    ----------
    localizations : ndarray
        (N, 4+) array: [frame, x, y, intensity, ...].
    n_frames : int
        Total number of frames in the acquisition.
    min_presence : float
        Minimum fraction of frames a fiducial must appear in (default 0.8).
    brightness_percentile : float
        Intensity percentile threshold for candidate selection (default 95).
    max_fiducials : int
        Maximum number of fiducials to return (default 10).

    Returns
    -------
    list of ndarray
        Each element is an (M, 3) array [frame, x, y] for one fiducial
        trajectory, sorted by frame.
    """
    frames = localizations[:, 0].astype(int)
    x = localizations[:, 1]
    y = localizations[:, 2]
    intensity = localizations[:, 3]

    # Brightness threshold
    bright_thresh = np.percentile(intensity, brightness_percentile)
    bright_mask = intensity >= bright_thresh
    bright_idx = np.where(bright_mask)[0]

    if len(bright_idx) == 0:
        return []

    # Cluster bright localizations across frames using simple
    # nearest-neighbor linking with a generous search radius
    bright_locs = localizations[bright_idx]
    search_radius = 3.0  # pixels

    # Group by frame
    frame_groups = {}
    for i, idx in enumerate(bright_idx):
        f = int(frames[idx])
        if f not in frame_groups:
            frame_groups[f] = []
        frame_groups[f].append((x[idx], y[idx], idx))

    # Simple greedy tracking: start tracks from first available frame
    tracks = []
    used = set()
    sorted_frames = sorted(frame_groups.keys())

    for start_frame in sorted_frames:
        for loc_x, loc_y, loc_idx in frame_groups[start_frame]:
            if loc_idx in used:
                continue

            track = [(start_frame, loc_x, loc_y)]
            used.add(loc_idx)
            last_x, last_y = loc_x, loc_y

            for f in sorted_frames:
                if f <= start_frame:
                    continue
                best_dist = search_radius
                best_match = None
                for cx, cy, cidx in frame_groups.get(f, []):
                    if cidx in used:
                        continue
                    d = np.sqrt((cx - last_x) ** 2 + (cy - last_y) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_match = (f, cx, cy, cidx)

                if best_match is not None:
                    track.append((best_match[0], best_match[1],
                                  best_match[2]))
                    used.add(best_match[3])
                    last_x, last_y = best_match[1], best_match[2]

            tracks.append(np.array(track, dtype=np.float64))

    # Filter by presence
    min_frames = int(min_presence * n_frames)
    tracks = [t for t in tracks if len(t) >= min_frames]

    # Sort by mean intensity (brightest first), limit count
    def _mean_brightness(track):
        total = 0.0
        for row in track:
            f = int(row[0])
            if f in frame_groups:
                for cx, cy, cidx in frame_groups[f]:
                    if abs(cx - row[1]) < 0.5 and abs(cy - row[2]) < 0.5:
                        total += intensity[cidx]
                        break
        return total / len(track)

    tracks.sort(key=_mean_brightness, reverse=True)
    return tracks[:max_fiducials]


def compute_drift_from_fiducials(fiducial_tracks, n_frames,
                                 smoothing_factor=None):
    """Compute drift trajectory from fiducial tracks.

    Parameters
    ----------
    fiducial_tracks : list of ndarray
        Each (M, 3) array: [frame, x, y].  From ``identify_fiducials``.
    n_frames : int
        Total number of frames.
    smoothing_factor : float, optional
        Spline smoothing factor.  If None, uses len(frames) * 0.1.

    Returns
    -------
    drift_x, drift_y : ndarray
        Drift in x and y for each frame, shape (n_frames,).
        Frame 0 has drift = 0 by convention.
    """
    if not fiducial_tracks:
        return np.zeros(n_frames), np.zeros(n_frames)

    # For each fiducial, compute displacement relative to its first position
    raw_dx = np.full((n_frames, len(fiducial_tracks)), np.nan)
    raw_dy = np.full((n_frames, len(fiducial_tracks)), np.nan)

    for k, track in enumerate(fiducial_tracks):
        frames = track[:, 0].astype(int)
        x_ref = track[0, 1]
        y_ref = track[0, 2]
        for i in range(len(track)):
            f = frames[i]
            if 0 <= f < n_frames:
                raw_dx[f, k] = track[i, 1] - x_ref
                raw_dy[f, k] = track[i, 2] - y_ref

    # Average across fiducials (ignoring NaN)
    with np.errstate(all='ignore'):
        mean_dx = np.nanmean(raw_dx, axis=1)
        mean_dy = np.nanmean(raw_dy, axis=1)

    # Replace remaining NaN with interpolation
    valid = ~np.isnan(mean_dx)
    if not np.any(valid):
        return np.zeros(n_frames), np.zeros(n_frames)

    frame_idx = np.arange(n_frames)
    mean_dx[~valid] = np.interp(frame_idx[~valid], frame_idx[valid],
                                 mean_dx[valid])
    mean_dy[~valid] = np.interp(frame_idx[~valid], frame_idx[valid],
                                 mean_dy[valid])

    # Smooth with cubic spline
    if smoothing_factor is None:
        smoothing_factor = max(n_frames * 0.1, 1.0)

    valid_frames = frame_idx[valid]
    try:
        spline_x = UnivariateSpline(valid_frames, mean_dx[valid],
                                     s=smoothing_factor, k=3)
        spline_y = UnivariateSpline(valid_frames, mean_dy[valid],
                                     s=smoothing_factor, k=3)
        drift_x = spline_x(frame_idx)
        drift_y = spline_y(frame_idx)
    except Exception:
        drift_x = mean_dx
        drift_y = mean_dy

    # Normalize so frame 0 has zero drift
    drift_x -= drift_x[0]
    drift_y -= drift_y[0]

    return drift_x, drift_y


def correct_drift_fiducial(localizations, n_frames, min_presence=0.8,
                           brightness_percentile=95, max_fiducials=10,
                           smoothing_factor=None):
    """Full fiducial-based drift correction pipeline.

    Parameters
    ----------
    localizations : ndarray
        (N, 4+) array: [frame, x, y, intensity, ...].
    n_frames : int
        Total number of frames.
    min_presence, brightness_percentile, max_fiducials : see identify_fiducials
    smoothing_factor : see compute_drift_from_fiducials

    Returns
    -------
    corrected : ndarray
        Copy of localizations with x, y columns drift-corrected.
    drift_x, drift_y : ndarray
        Applied drift vectors (n_frames,).
    fiducial_tracks : list
        Identified fiducial trajectories.
    """
    fiducials = identify_fiducials(
        localizations, n_frames,
        min_presence=min_presence,
        brightness_percentile=brightness_percentile,
        max_fiducials=max_fiducials)

    drift_x, drift_y = compute_drift_from_fiducials(
        fiducials, n_frames, smoothing_factor=smoothing_factor)

    corrected = localizations.copy()
    frames = corrected[:, 0].astype(int)
    valid = (frames >= 0) & (frames < n_frames)
    corrected[valid, 1] -= drift_x[frames[valid]]
    corrected[valid, 2] -= drift_y[frames[valid]]

    return corrected, drift_x, drift_y, fiducials
