"""Greedy nearest-neighbor particle linker.

Supports both iterative (queue-based) and recursive linking methods.
The iterative method avoids stack overflow on large datasets.
The recursive method faithfully replicates the original plugin's recursive
linking algorithm with configurable depth limit and failure tracking.

Produces the same output format as other linkers: (tracks, stats).
"""
import numpy as np
from collections import deque
from ...logger import logger


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_DISTANCE = 5.0
_DEFAULT_MAX_GAP = 1
_DEFAULT_MIN_TRACK_LENGTH = 1
_DEFAULT_RECURSIVE_DEPTH_LIMIT = 5000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_particles(locs, max_distance=_DEFAULT_MAX_DISTANCE, max_gap=_DEFAULT_MAX_GAP,
                   min_track_length=_DEFAULT_MIN_TRACK_LENGTH,
                   method='iterative', recursive_depth_limit=_DEFAULT_RECURSIVE_DEPTH_LIMIT,
                   image_data=None):
    """Link localizations into tracks using greedy nearest-neighbor.

    Args:
        locs: (N, 3+) array with columns [frame, x, y, ...].
              Extra columns are preserved but not used for linking.
        max_distance: Maximum linking distance in pixels
        max_gap: Maximum frames to skip (gap closing)
        min_track_length: Minimum track length to keep
        method: 'iterative' (default, queue-based) or 'recursive'
            (original plugin algorithm with stack recursion)
        recursive_depth_limit: Maximum recursion depth for recursive method.
            Default 5000. If exceeded, the recursiveFailure flag is set
            in stats and the track is terminated at that point.
        image_data: Optional 3D array (T, H, W) for intensity extraction
            via getIntensities. If provided, intensities are included
            in stats.

    Returns:
        (tracks, stats) where:
            tracks: list of lists of row indices into locs
            stats: dict with comprehensive linking metrics matching
                   the original plugin format
    """
    locs = np.asarray(locs, dtype=np.float64)
    if len(locs) == 0:
        return [], _empty_stats()

    frames = np.unique(locs[:, 0]).astype(int)
    max_frame = int(np.max(frames))

    # Build per-frame index structures
    pts_by_frame = []    # positions per frame
    idx_by_frame = []    # global indices per frame
    remaining = []       # availability flags

    for f in range(max_frame + 1):
        mask = locs[:, 0].astype(int) == f
        indices = np.where(mask)[0]
        pos = locs[indices, 1:3]  # x, y only
        pts_by_frame.append(pos)
        idx_by_frame.append(indices)
        remaining.append(np.ones(len(indices), dtype=bool))

    # Track recursive failure
    recursive_failure = False

    # Link tracks
    tracks = []
    if method == 'recursive':
        import sys
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(max(old_limit, recursive_depth_limit + 100))
            for f in frames:
                f = int(f)
                available = np.where(remaining[f])[0]
                for local_idx in available:
                    if not remaining[f][local_idx]:
                        continue
                    remaining[f][local_idx] = False
                    abs_idx = idx_by_frame[f][local_idx]

                    track = [abs_idx]
                    failed = _extend_track_recursive(
                        abs_idx, locs, pts_by_frame, idx_by_frame, remaining,
                        max_distance, max_gap, max_frame, track, 0,
                        recursive_depth_limit)
                    if failed:
                        recursive_failure = True
                    tracks.append(track)
        finally:
            sys.setrecursionlimit(old_limit)
    else:
        # Iterative (default)
        for f in frames:
            f = int(f)
            available = np.where(remaining[f])[0]
            for local_idx in available:
                if not remaining[f][local_idx]:
                    continue
                remaining[f][local_idx] = False
                abs_idx = idx_by_frame[f][local_idx]

                track = _extend_track_iterative(
                    abs_idx, locs, pts_by_frame, idx_by_frame, remaining,
                    max_distance, max_gap, max_frame)
                tracks.append(track)

    # Filter by minimum length
    if min_track_length > 1:
        tracks = [t for t in tracks if len(t) >= min_track_length]

    stats = _compute_stats(tracks, len(locs), method, recursive_failure)

    # Extract intensities from image data if provided
    if image_data is not None:
        intensities = getIntensities(locs, tracks, image_data)
        stats['intensities'] = intensities

    return tracks, stats


def getIntensities(locs, tracks, image_data):
    """Extract intensities from image data for each localization in tracks.

    For each localization, extracts the pixel value at the (frame, y, x)
    coordinates from the image_data array.

    Args:
        locs: (N, 3+) localization array [frame, x, y, ...]
        tracks: list of lists of row indices into locs
        image_data: 3D array (T, H, W) or 4D array (T, Z, H, W)

    Returns:
        dict: {track_id: list of intensity values}
    """
    intensities = {}
    if image_data is None:
        return intensities

    for tid, track in enumerate(tracks):
        track_intensities = []
        for idx in track:
            frame = int(locs[idx, 0])
            x = locs[idx, 1]
            y = locs[idx, 2]

            # Round to nearest pixel
            ix = int(np.round(x))
            iy = int(np.round(y))

            if image_data.ndim == 3:
                # (T, H, W)
                T, H, W = image_data.shape
                if 0 <= frame < T and 0 <= iy < H and 0 <= ix < W:
                    track_intensities.append(float(image_data[frame, iy, ix]))
                else:
                    track_intensities.append(0.0)
            elif image_data.ndim == 4:
                # (T, Z, H, W) - use max projection over Z
                T, Z, H, W = image_data.shape
                if 0 <= frame < T and 0 <= iy < H and 0 <= ix < W:
                    val = float(np.max(image_data[frame, :, iy, ix]))
                    track_intensities.append(val)
                else:
                    track_intensities.append(0.0)
            else:
                track_intensities.append(0.0)

        intensities[tid] = track_intensities

    return intensities


# ---------------------------------------------------------------------------
# Iterative linking (queue-based)
# ---------------------------------------------------------------------------

def _extend_track_iterative(start_idx, locs, pts_by_frame, idx_by_frame,
                            remaining, max_distance, max_gap, max_frame):
    """Extend a track iteratively using a queue.

    This is the safe default method that avoids stack overflow.

    Args:
        start_idx: global index of starting localization
        locs: full localization array
        pts_by_frame: list of per-frame position arrays
        idx_by_frame: list of per-frame global index arrays
        remaining: list of per-frame availability boolean arrays
        max_distance: maximum linking distance
        max_gap: maximum frame gap
        max_frame: maximum frame number in dataset

    Returns:
        list of global indices forming the track
    """
    track = [start_idx]
    queue = deque([start_idx])

    while queue:
        current_idx = queue.popleft()
        pt = locs[current_idx]
        current_frame = int(pt[0])

        for dt in range(1, max_gap + 2):
            next_frame = current_frame + dt
            if next_frame > max_frame:
                break

            avail = remaining[next_frame]
            if not np.any(avail):
                continue

            candidate_local = np.where(avail)[0]
            candidate_pos = pts_by_frame[next_frame][candidate_local]

            dists = np.sqrt(np.sum((candidate_pos - pt[1:3]) ** 2, axis=1))
            valid = dists < max_distance

            if np.any(valid):
                best = candidate_local[np.argmin(dists)]
                abs_next = idx_by_frame[next_frame][best]
                remaining[next_frame][best] = False
                track.append(abs_next)
                queue.append(abs_next)
                break  # found connection, continue from new point

    return track


# ---------------------------------------------------------------------------
# Recursive linking (original plugin algorithm)
# ---------------------------------------------------------------------------

def _extend_track_recursive(current_idx, locs, pts_by_frame, idx_by_frame,
                            remaining, max_distance, max_gap, max_frame,
                            track, depth, depth_limit):
    """Extend a track recursively (original plugin algorithm).

    This replicates the original plugin's recursive linking. It has a
    configurable depth limit to prevent true stack overflow.

    Args:
        current_idx: global index of current localization
        locs: full localization array
        pts_by_frame: list of per-frame position arrays
        idx_by_frame: list of per-frame global index arrays
        remaining: list of per-frame availability boolean arrays
        max_distance: maximum linking distance
        max_gap: maximum frame gap
        max_frame: maximum frame number in dataset
        track: list being built (mutated in place)
        depth: current recursion depth
        depth_limit: maximum recursion depth

    Returns:
        bool: True if recursion depth limit was hit (recursiveFailure)
    """
    if depth >= depth_limit:
        logger.warning("Recursive linking hit depth limit %d at index %d",
                        depth_limit, current_idx)
        return True  # recursiveFailure

    pt = locs[current_idx]
    current_frame = int(pt[0])

    for dt in range(1, max_gap + 2):
        next_frame = current_frame + dt
        if next_frame > max_frame:
            break

        avail = remaining[next_frame]
        if not np.any(avail):
            continue

        candidate_local = np.where(avail)[0]
        candidate_pos = pts_by_frame[next_frame][candidate_local]

        dists = np.sqrt(np.sum((candidate_pos - pt[1:3]) ** 2, axis=1))
        valid = dists < max_distance

        if np.any(valid):
            best = candidate_local[np.argmin(dists)]
            abs_next = idx_by_frame[next_frame][best]
            remaining[next_frame][best] = False
            track.append(abs_next)

            # Recurse to extend from new point
            failed = _extend_track_recursive(
                abs_next, locs, pts_by_frame, idx_by_frame,
                remaining, max_distance, max_gap, max_frame,
                track, depth + 1, depth_limit)
            return failed

    return False  # no failure, track just ended naturally


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def tracks_to_array(locs, tracks):
    """Add track_id column to localizations array.

    Args:
        locs: (N, M) original localizations array
        tracks: list of lists of indices (from link_particles)
    Returns:
        (N, M+1) array with track_id as last column (-1 = unlinked)
    """
    track_ids = np.full(len(locs), -1, dtype=int)
    for tid, track in enumerate(tracks):
        for idx in track:
            track_ids[idx] = tid
    return np.column_stack([locs, track_ids])


def tracks_to_dict(locs, tracks):
    """Convert tracks to dict format {track_id: (N, 3) array of [frame, x, y]}.

    Suitable for TrackOverlay.load_tracks_from_dict().
    """
    result = {}
    for tid, track in enumerate(tracks):
        pts = locs[track][:, :3]  # frame, x, y
        result[tid] = pts[pts[:, 0].argsort()]  # sort by frame
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(tracks, total_points, method='iterative',
                   recursive_failure=False):
    """Compute comprehensive linking statistics matching plugin format.

    Args:
        tracks: list of track index lists
        total_points: total number of localizations
        method: linking method used ('iterative' or 'recursive')
        recursive_failure: whether recursive depth limit was hit

    Returns:
        dict with comprehensive linking statistics
    """
    if not tracks:
        stats = _empty_stats()
        stats['method'] = method
        stats['recursiveFailure'] = recursive_failure
        return stats

    lengths = np.array([len(t) for t in tracks])
    linked = int(np.sum(lengths))

    # Track length histogram
    max_len = int(np.max(lengths))
    length_hist, length_edges = np.histogram(
        lengths, bins=min(max_len, 50), range=(1, max_len + 1))

    stats = {
        'num_tracks': len(tracks),
        'total_points': total_points,
        'linked_points': linked,
        'unlinked_points': total_points - linked,
        'mean_track_length': float(np.mean(lengths)),
        'median_track_length': float(np.median(lengths)),
        'max_track_length': int(np.max(lengths)),
        'min_track_length': int(np.min(lengths)),
        'std_track_length': float(np.std(lengths)),
        'linking_efficiency': linked / total_points if total_points > 0 else 0.0,
        'method': method,
        'recursiveFailure': recursive_failure,
        'length_histogram': {
            'counts': length_hist.tolist(),
            'edges': length_edges.tolist(),
        },
        'single_point_tracks': int(np.sum(lengths == 1)),
        'long_tracks': int(np.sum(lengths >= 10)),
    }
    return stats


def _empty_stats():
    """Return an empty statistics dict in the comprehensive format."""
    return {
        'num_tracks': 0,
        'total_points': 0,
        'linked_points': 0,
        'unlinked_points': 0,
        'mean_track_length': 0,
        'median_track_length': 0,
        'max_track_length': 0,
        'min_track_length': 0,
        'std_track_length': 0,
        'linking_efficiency': 0.0,
        'method': 'iterative',
        'recursiveFailure': False,
        'length_histogram': {'counts': [], 'edges': []},
        'single_point_tracks': 0,
        'long_tracks': 0,
    }
