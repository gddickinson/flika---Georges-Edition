# -*- coding: utf-8 -*-
"""Structured ground truth container and conversion helpers."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GroundTruth:
    """Structured ground truth container for simulation evaluation.

    All fields are optional since different simulation presets populate
    different subsets. Use ``to_dict()``/``from_dict()`` for serialization
    into ``window.metadata['simulation']``.
    """
    # --- Detection GT ---
    # Per-frame emitter positions: (N, 4+) [frame, y, x, particle_id, ...]
    positions_per_frame: np.ndarray | None = None

    # --- Tracking GT ---
    # {particle_id: (N, 3) array [frame, x, y]}
    trajectories: dict | None = None

    # --- Segmentation GT ---
    # Integer label mask: 0 = background, 1..N = object labels
    segmentation_mask: np.ndarray | None = None

    # Binary foreground mask
    binary_mask: np.ndarray | None = None

    # --- Calcium GT ---
    calcium_trace: np.ndarray | None = None
    calcium_events: list | None = None
    calcium_spike_frames: np.ndarray | None = None

    # --- Diffusion GT ---
    diffusion_coefficients: np.ndarray | None = None
    motion_states: np.ndarray | None = None

    # --- SMLM GT ---
    emitter_states: np.ndarray | None = None
    true_positions_nm: np.ndarray | None = None

    # --- Raw ---
    n_photons: np.ndarray | None = None

    def to_dict(self):
        """Convert to dict for storage in window.metadata."""
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, np.ndarray):
                    d[k] = v
                elif isinstance(v, dict):
                    d[k] = v
                elif isinstance(v, list):
                    d[k] = v
                else:
                    d[k] = v
        return d

    @classmethod
    def from_dict(cls, d):
        """Reconstruct from dict stored in window.metadata."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**kwargs)


def positions_to_detection_table(positions, trajectories, config):
    """Convert engine positions + trajectories into a per-frame detection table.

    Parameters
    ----------
    positions : ndarray
        (N, ndim) initial emitter positions in pixels.
    trajectories : ndarray or None
        (N, T, ndim) displacement array from dynamics (already in pixels).
    config : SimulationConfig
        Simulation configuration for frame count.

    Returns
    -------
    ndarray
        (M, 4) array with columns [frame, y, x, particle_id].
        For 3D positions, uses the last two dims as (y, x).
    """
    n_particles = len(positions)
    ndim = positions.shape[1]
    rows = []

    for t in range(config.nt):
        for p_id in range(n_particles):
            if trajectories is not None:
                pos = positions[p_id] + trajectories[p_id, t]
            else:
                pos = positions[p_id]
            # Extract y, x (last two dimensions)
            if ndim >= 2:
                y, x = pos[-2], pos[-1]
            else:
                y, x = pos[0], 0
            rows.append([t, y, x, p_id])

    return np.array(rows) if rows else np.empty((0, 4))


def trajectories_to_track_dict(positions, trajectories, config):
    """Convert engine positions + trajectories into SPT track format.

    Parameters
    ----------
    positions : ndarray
        (N, ndim) initial positions in pixels.
    trajectories : ndarray or None
        (N, T, ndim) displacements in pixels.
    config : SimulationConfig
        Configuration with nt, pixel_size, etc.

    Returns
    -------
    dict
        {particle_id: (T, 3) array [frame, x, y]} matching flika SPT format.
    """
    n_particles = len(positions)
    ndim = positions.shape[1]
    tracks = {}

    for p_id in range(n_particles):
        frames = []
        for t in range(config.nt):
            if trajectories is not None:
                pos = positions[p_id] + trajectories[p_id, t]
            else:
                pos = positions[p_id]
            x = pos[-1]
            y = pos[-2] if ndim >= 2 else 0
            frames.append([t, x, y])
        tracks[p_id] = np.array(frames)

    return tracks


def detection_table_for_frame(gt_table, frame):
    """Extract ground truth detections for a single frame.

    Parameters
    ----------
    gt_table : ndarray
        (M, 4+) table [frame, y, x, ...].
    frame : int
        Frame index.

    Returns
    -------
    ndarray
        (K, 2) positions [y, x] for that frame.
    """
    mask = gt_table[:, 0] == frame
    return gt_table[mask, 1:3]
