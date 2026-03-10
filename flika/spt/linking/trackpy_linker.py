"""Trackpy adapter for particle linking.

Wraps the trackpy package (Crocker-Grier algorithm with predictive
tracking and adaptive search) to provide linking compatible with the
flika SPT framework.  Falls back gracefully when trackpy is not
installed by raising an ``ImportError`` with an installation hint.

Supports all four linking types from the original plugin:
  - 'standard':                     basic trackpy.link
  - 'adaptive':                     adaptive search range reduction
  - 'velocityPredict':              tp.predict.NearestVelocityPredict()
  - 'adaptive + velocityPredict':   both adaptive search and velocity prediction

Reference: Allan et al. (2021). soft-matter/trackpy, Zenodo.
"""
import numpy as np
from flika.logger import logger


def _check_trackpy():
    """Import and return the trackpy module, or raise ImportError."""
    try:
        import trackpy
        return trackpy
    except ImportError:
        raise ImportError(
            "trackpy is required for Crocker-Grier linking but is not "
            "installed. Install it with:\n"
            "    pip install trackpy\n"
            "or\n"
            "    conda install -c conda-forge trackpy\n\n"
            "Alternatively, use the built-in greedy or u-track linkers "
            "which have no additional dependencies."
        )


def link_with_trackpy(locs, search_range=5.0, memory=1,
                      link_type='standard',
                      adaptive_stop=None, adaptive_step=0.95,
                      min_track_length=3):
    """Link particles into tracks using trackpy (Crocker-Grier).

    Converts the flika-format localisation array into a pandas DataFrame,
    runs trackpy linking with the specified link_type, and converts the
    result back into the ``(tracks, stats)`` tuple format.

    All four linking types from the original plugin are supported:
      - 'standard': basic ``trackpy.link`` with no prediction
      - 'adaptive': adaptive search range that narrows in dense regions
      - 'velocityPredict': ``tp.predict.NearestVelocityPredict()``
          wraps the linker so particles are predicted to continue with
          their most recent velocity
      - 'adaptive + velocityPredict': both adaptive search and velocity
          prediction combined

    After linking, tracks shorter than ``min_track_length`` are removed
    and the remaining tracks are renumbered starting from 0.

    Args:
        locs: (N, 3+) array with columns ``[frame, x, y, ...]``.
            Extra columns beyond the first three are preserved in the
            output but are not used for linking.
        search_range: Maximum linking distance (pixels) between
            consecutive frames.  Corresponds to trackpy's ``search_range``
            parameter.
        memory: Maximum number of frames a particle can disappear and
            still be linked (gap closing).
        link_type: One of 'standard', 'adaptive', 'velocityPredict',
            'adaptive + velocityPredict'. Controls which trackpy linking
            mode is used.
        adaptive_stop: If link_type includes 'adaptive', this is the
            minimum search range.  If None and adaptive is requested,
            defaults to search_range * 0.3.
        adaptive_step: Multiplicative factor for adaptive search
            reduction (default 0.95).
        min_track_length: Minimum number of localizations a track must
            contain to be retained in the output.

    Returns:
        ``(tracks, stats)`` tuple:

        - **tracks**: list of lists of row indices into *locs*.
        - **stats**: dict with linking summary metrics.

    Raises:
        ImportError: If trackpy is not installed.
        ValueError: If link_type is not recognized.
    """
    tp = _check_trackpy()
    import pandas as pd

    # Suppress trackpy's verbose output
    tp.quiet()

    locs = np.asarray(locs, dtype=np.float64)
    if len(locs) == 0:
        return [], _empty_stats()

    # Validate link_type
    valid_types = ('standard', 'adaptive', 'velocityPredict',
                   'adaptive + velocityPredict')
    if link_type not in valid_types:
        raise ValueError(
            f"Unknown link_type {link_type!r}. Must be one of {valid_types}")

    use_adaptive = 'adaptive' in link_type
    use_velocity = 'velocityPredict' in link_type

    # Build DataFrame in the format trackpy expects
    df = pd.DataFrame({
        'frame': locs[:, 0].astype(int),
        'x': locs[:, 1],
        'y': locs[:, 2],
        '_orig_idx': np.arange(len(locs)),
    })

    logger.info("trackpy linking: %d localisations, search_range=%.1f, "
                "memory=%d, link_type=%s",
                len(df), search_range, memory, link_type)

    # Build link kwargs
    link_kwargs = dict(
        search_range=search_range,
        memory=memory,
    )

    # Adaptive search parameters
    if use_adaptive:
        if adaptive_stop is None:
            adaptive_stop = search_range * 0.3
        link_kwargs['adaptive_stop'] = adaptive_stop
        link_kwargs['adaptive_step'] = adaptive_step

    # Determine which link function to use
    # trackpy >= 0.5 uses tp.link; older versions use tp.link_df
    link_fn = getattr(tp, 'link', None) or getattr(tp, 'link_df', None)
    if link_fn is None:
        raise RuntimeError(
            "Could not find trackpy.link or trackpy.link_df. "
            "Please update trackpy to a recent version.")

    try:
        if use_velocity:
            # Velocity prediction mode: NearestVelocityPredict
            pred = tp.predict.NearestVelocityPredict()

            if use_adaptive:
                # adaptive + velocityPredict: use pred.link_df with adaptive params
                # NearestVelocityPredict wraps the link function
                linked = pred.link_df(df, **link_kwargs)
            else:
                # velocityPredict only (no adaptive)
                linked = pred.link_df(df, **link_kwargs)
        else:
            # Standard or adaptive-only linking
            linked = link_fn(df, **link_kwargs)

    except Exception as exc:
        logger.error("trackpy linking failed: %s", exc)
        raise

    # trackpy adds a 'particle' column with track IDs
    if 'particle' not in linked.columns:
        logger.warning("trackpy did not produce a 'particle' column; "
                       "returning empty results")
        return [], _empty_stats()

    # Convert back to tracks (list of lists of original row indices)
    tracks = []
    for pid, group in linked.groupby('particle'):
        indices = group['_orig_idx'].values.tolist()
        tracks.append(indices)

    # Filter by minimum track length
    if min_track_length > 1:
        before_count = len(tracks)
        tracks = [t for t in tracks if len(t) >= min_track_length]
        removed = before_count - len(tracks)
        if removed > 0:
            logger.debug("Filtered %d short tracks (< %d points)",
                         removed, min_track_length)

    # Renumber tracks after filtering (sequential from 0)
    # This ensures track IDs are contiguous 0..N-1
    # (The tracks list itself is already 0-indexed by position,
    #  but this comment documents the intent for downstream consumers)

    stats = _compute_stats(tracks, len(locs), link_type)
    logger.info("trackpy linking complete: %d tracks, efficiency=%.1f%%, "
                "link_type=%s",
                stats['num_tracks'],
                stats['linking_efficiency'] * 100,
                link_type)
    return tracks, stats


def _compute_stats(tracks, total_points, link_type='standard'):
    """Compute linking statistics (same format as greedy_linker)."""
    if not tracks:
        stats = _empty_stats()
        stats['link_type'] = link_type
        return stats
    lengths = np.array([len(t) for t in tracks])
    linked = int(np.sum(lengths))
    return {
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
        'link_type': link_type,
    }


def _empty_stats():
    """Return an empty statistics dict."""
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
        'link_type': 'standard',
    }
