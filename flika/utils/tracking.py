"""Particle tracking / linking utilities.

Provides nearest-neighbour linking of localisations across frames with
optional gap-closing (frame skipping).

Example::

    from flika.utils.tracking import link_particles

    # locs is an (N, 3+) array: columns [frame, x, y, ...]
    tracks = link_particles(locs, max_distance=5.0, max_gap=2)
    # tracks is a list of lists, each sub-list containing row indices
    # into locs that form a connected track.
"""
from __future__ import annotations

import numpy as np


def link_particles(locs, max_distance=5.0, max_gap=1):
    """Link particle localisations across frames using greedy nearest-neighbour.

    Parameters
    ----------
    locs : ndarray (N, 3+)
        Each row is ``[frame, x, y, ...]``.  Only the first three columns
        are used for linking.
    max_distance : float
        Maximum Euclidean distance (in pixels) a particle can move between
        consecutive frames.
    max_gap : int
        Maximum number of consecutive frames a particle can skip and
        still be linked (0 = no gap closing).

    Returns
    -------
    tracks : list of list of int
        Each track is a list of row indices into *locs*.
    """
    locs = np.asarray(locs, dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] < 3:
        raise ValueError("locs must have shape (N, 3+) with columns [frame, x, y, ...]")

    frames = locs[:, 0].astype(int)
    max_frame = int(frames.max()) if len(frames) > 0 else 0

    # Organise points by frame
    pts_by_frame = []
    idx_by_frame = []
    remaining = []
    for f in range(max_frame + 1):
        mask = frames == f
        indices = np.where(mask)[0]
        positions = locs[indices, 1:3]  # (x, y)
        pts_by_frame.append(positions)
        idx_by_frame.append(indices)
        remaining.append(np.ones(len(indices), dtype=bool))

    tracks = []
    unique_frames = np.unique(frames).astype(int)

    for frame in unique_frames:
        for pt_i in range(len(pts_by_frame[frame])):
            if not remaining[frame][pt_i]:
                continue
            remaining[frame][pt_i] = False
            abs_idx = idx_by_frame[frame][pt_i]
            track = [abs_idx]
            track = _extend_track(track, locs, pts_by_frame, idx_by_frame,
                                  remaining, max_distance, max_gap, max_frame)
            tracks.append(track)

    return tracks


def _extend_track(track, locs, pts_by_frame, idx_by_frame, remaining,
                  max_distance, max_gap, max_frame):
    """Greedily extend *track* forward in time (iterative to avoid stack overflow)."""
    while True:
        last_pt = locs[track[-1]]  # [frame, x, y, ...]
        current_frame = int(last_pt[0])
        pos = last_pt[1:3]

        found = False
        for dt in range(1, max_gap + 2):
            f = current_frame + dt
            if f > max_frame:
                return track

            cand_mask = remaining[f]
            if not np.any(cand_mask):
                continue

            cand_pos = pts_by_frame[f][cand_mask]
            distances = np.sqrt(np.sum((cand_pos - pos) ** 2, axis=1))

            if np.any(distances < max_distance):
                best = np.argmin(distances)
                cand_indices = np.where(cand_mask)[0]
                local_idx = cand_indices[best]
                abs_idx = idx_by_frame[f][local_idx]
                remaining[f][local_idx] = False
                track.append(abs_idx)
                found = True
                break

        if not found:
            return track


def tracks_to_array(locs, tracks):
    """Convert track lists to a labelled array with a track_id column.

    Parameters
    ----------
    locs : ndarray (N, M)
    tracks : list of list of int

    Returns
    -------
    labelled : ndarray (N, M+1)
        Same as *locs* with an extra last column ``track_id``.
        Points not in any track get ``track_id = -1``.
    """
    locs = np.asarray(locs)
    labels = np.full(len(locs), -1, dtype=int)
    for track_id, track in enumerate(tracks):
        for idx in track:
            labels[idx] = track_id
    return np.column_stack([locs, labels])
