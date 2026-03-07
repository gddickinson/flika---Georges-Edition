"""Spatial features: nearest neighbors, local density.

Exact replica of the algorithms from:
- spt_batch_analysis::NearestNeighborAnalyzer (get_nearest_neighbors,
  count_neighbors_in_radius, analyze_frame_neighbors)
- locsAndTracksPlotter output columns: nnDist, nnIndex_inFrame, nnDist_inFrame
"""
import numpy as np
from scipy.spatial import KDTree
from ...logger import logger


# ---------------------------------------------------------------------------
# Core nearest-neighbor helpers
# ---------------------------------------------------------------------------

def nearest_neighbor_distances(xy_positions):
    """Compute nearest neighbor distance AND index for each point.

    Uses KDTree with k=2 (self + nearest), exactly as in
    spt_batch_analysis::NearestNeighborAnalyzer.get_nearest_neighbors.

    Args:
        xy_positions: (N, 2) array of [x, y] positions
    Returns:
        tuple of:
            distances: (N,) array of nearest neighbor distances (NaN if N < 2)
            indices:   (N,) array of nearest neighbor indices   (NaN if N < 2)
    """
    xy = np.asarray(xy_positions, dtype=np.float64)
    n = len(xy)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan)

    tree = KDTree(xy)
    dists, idxs = tree.query(xy, k=2)  # k=2 because first is self
    return dists[:, 1], idxs[:, 1].astype(float)


def neighbor_counts(xy_positions, radii=(3, 5, 10, 20, 30)):
    """Count neighbors within each radius for every point.

    Exact replica of spt_batch_analysis::NearestNeighborAnalyzer.count_neighbors_in_radius.

    Args:
        xy_positions: (N, 2) array
        radii: iterable of radius values
    Returns:
        dict mapping radius -> (N,) array of counts (excluding self)
    """
    xy = np.asarray(xy_positions, dtype=np.float64)
    n = len(xy)
    result = {}

    if n < 2:
        for r in radii:
            result[r] = np.zeros(n, dtype=int)
        return result

    tree = KDTree(xy)
    for r in radii:
        counts = tree.query_ball_point(xy, r=r, return_length=True)
        result[r] = np.asarray(counts) - 1  # exclude self

    return result


# ---------------------------------------------------------------------------
# Per-frame neighbor analysis -- exact spt_batch_analysis pipeline
# ---------------------------------------------------------------------------

def analyze_frame_neighbors(locs_df, radii=(3, 5, 10, 20, 30)):
    """Add nearest neighbor columns to a localizations DataFrame.

    Exact replica of spt_batch_analysis::NearestNeighborAnalyzer.analyze_frame_neighbors.

    Produces the following columns (matching the original plugin output):
        - nnDist:           nearest neighbor distance (across all points in frame)
        - nnIndex_inFrame:  index of nearest neighbor within the same frame
        - nnDist_inFrame:   distance to nearest neighbor within the same frame
        - nnCountInFrame_within_{r}_pixels: count of neighbors within radius r

    Args:
        locs_df: DataFrame with columns 'frame', 'x', 'y'
        radii: iterable of radius values
    Returns:
        DataFrame with added columns
    """
    import pandas as pd

    df = locs_df.copy()
    df = df.sort_values(by=['frame'])

    # Initialise columns
    df['nnDist'] = np.nan
    df['nnIndex_inFrame'] = np.nan
    df['nnDist_inFrame'] = np.nan
    for r in radii:
        df[f'nnCountInFrame_within_{r}_pixels'] = 0

    frames = df['frame'].unique()

    for frame in frames:
        mask = df['frame'] == frame
        frame_indices = df.index[mask]
        xy = df.loc[mask, ['x', 'y']].to_numpy()

        if len(xy) < 2:
            # Single point in frame -- leave as NaN / 0
            df.loc[mask, 'nnDist'] = np.nan
            continue

        # Nearest neighbor distances and indices
        nn_dists, nn_idxs = nearest_neighbor_distances(xy)

        # nnDist -- distance to nearest neighbor in this frame
        df.loc[frame_indices, 'nnDist'] = nn_dists

        # nnIndex_inFrame -- the DataFrame index of the nearest neighbor
        # The nn_idxs are positional within the frame subset, so we map
        # them back to the DataFrame index.
        nn_df_indices = frame_indices[nn_idxs.astype(int)]
        df.loc[frame_indices, 'nnIndex_inFrame'] = nn_df_indices.values

        # nnDist_inFrame -- same as nnDist (nearest neighbor distance
        # computed within the same frame)
        df.loc[frame_indices, 'nnDist_inFrame'] = nn_dists

        # Neighbor counts within radii
        counts = neighbor_counts(xy, radii)
        for r in radii:
            df.loc[frame_indices, f'nnCountInFrame_within_{r}_pixels'] = counts[r]

    # Restore original sort order
    df = df.sort_values(['track_number', 'frame']) if 'track_number' in df.columns else df

    return df
