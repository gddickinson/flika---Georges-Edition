"""Density-peak clustering utility.

Implements the Rodriguez & Laio (2014) density-peak clustering algorithm
adapted from the detect_puffs plugin.  For each data point, the algorithm
finds the nearest point of higher local density, then identifies cluster
centres as points with both high density and large distance to any
higher-density neighbour.

Example::

    from flika.utils.clustering import density_peak_cluster

    # points is (N, D) — e.g. (N, 3) for (t, x, y)
    labels, centres = density_peak_cluster(points, density_radius=5.0,
                                           min_density=0.1)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import distance_matrix as _distance_matrix

# Try to import numba for JIT acceleration of O(N²) loop
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _higher_density_distance_numba(points, rho, order):
    """Numba-accelerated O(N²) distance computation."""
    N = len(points)
    delta = np.full(N, np.inf)
    nearest = np.full(N, -1, dtype=np.int64)

    for idx_in_order in range(1, N):
        i = order[idx_in_order]
        best_dist = np.inf
        best_j = -1
        for k in range(idx_in_order):
            j = order[k]
            d = 0.0
            for dim in range(points.shape[1]):
                diff = points[i, dim] - points[j, dim]
                d += diff * diff
            d = np.sqrt(d)
            if d < best_dist:
                best_dist = d
                best_j = j
        delta[i] = best_dist
        nearest[i] = best_j

    return delta, nearest


if _HAS_NUMBA:
    _higher_density_distance_jit = _njit(_higher_density_distance_numba, cache=True)
else:
    _higher_density_distance_jit = None


def local_density(points, radius):
    """Compute local density for each point using a Gaussian kernel.

    Parameters
    ----------
    points : ndarray (N, D)
    radius : float
        Characteristic scale for the Gaussian kernel.

    Returns
    -------
    rho : ndarray (N,)  — density estimate for each point
    """
    N = len(points)
    if N == 0:
        return np.array([])
    # For very large N, use chunked computation
    if N > 10000:
        rho = np.zeros(N)
        chunk = 2000
        for i in range(0, N, chunk):
            end_i = min(i + chunk, N)
            D = _distance_matrix(points[i:end_i], points)
            rho[i:end_i] = np.sum(np.exp(-(D / radius) ** 2), axis=1) - 1.0
        return rho
    D = _distance_matrix(points, points)
    rho = np.sum(np.exp(-(D / radius) ** 2), axis=1) - 1.0  # subtract self
    return rho


def higher_density_distance(points, rho):
    """For each point, find the distance to the nearest point of higher density.

    Parameters
    ----------
    points : ndarray (N, D)
    rho : ndarray (N,)

    Returns
    -------
    delta : ndarray (N,) — distance to nearest higher-density point
    nearest : ndarray (N,) int — index of that higher-density point
    """
    N = len(points)
    order = np.argsort(-rho)

    # Use numba-accelerated version if available
    if _higher_density_distance_jit is not None:
        try:
            delta, nearest = _higher_density_distance_jit(
                np.ascontiguousarray(points), rho,
                np.ascontiguousarray(order))
        except Exception:
            # Fallback to pure Python on any numba error
            delta, nearest = _higher_density_distance_pure(points, rho, order, N)
    else:
        delta, nearest = _higher_density_distance_pure(points, rho, order, N)

    # Highest-density point: set delta to max
    top = order[0]
    delta[top] = np.max(delta[delta < np.inf]) if N > 1 else 0.0
    nearest[top] = top

    return delta, nearest


def _higher_density_distance_pure(points, rho, order, N):
    """Pure Python/NumPy fallback for higher_density_distance."""
    delta = np.full(N, np.inf)
    nearest = np.full(N, -1, dtype=int)

    for idx_in_order in range(1, N):
        i = order[idx_in_order]
        higher = order[:idx_in_order]
        dists = np.sqrt(np.sum((points[higher] - points[i]) ** 2, axis=1))
        best = np.argmin(dists)
        delta[i] = dists[best]
        nearest[i] = higher[best]

    return delta, nearest


def density_peak_cluster(points, density_radius=5.0, min_density=None,
                         min_delta=None, n_clusters=None):
    """Cluster *points* using the density-peak algorithm.

    Parameters
    ----------
    points : ndarray (N, D)
    density_radius : float
        Scale for the density estimation kernel.
    min_density : float, optional
        Minimum density to be considered a cluster centre.
        Defaults to the median density.
    min_delta : float, optional
        Minimum δ (distance to higher-density neighbour) for a centre.
        Defaults to the 90th percentile of δ.
    n_clusters : int, optional
        If given, select the top *n_clusters* centres by ``rho * delta``
        and ignore *min_density* / *min_delta*.

    Returns
    -------
    labels : ndarray (N,) int  — cluster label for each point (−1 = noise)
    centres : ndarray (K,) int — indices of cluster centres
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if N == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rho = local_density(points, density_radius)
    delta, nearest = higher_density_distance(points, rho)

    # Identify cluster centres
    if n_clusters is not None:
        gamma = rho * delta
        centres = np.argsort(-gamma)[:n_clusters]
    else:
        if min_density is None:
            min_density = np.median(rho)
        if min_delta is None:
            min_delta = np.percentile(delta[delta < np.inf], 90) if N > 1 else 0
        centres = np.where((rho >= min_density) & (delta >= min_delta))[0]

    if len(centres) == 0:
        return np.full(N, -1, dtype=int), np.array([], dtype=int)

    # Assign labels by propagation from high to low density
    labels = np.full(N, -1, dtype=int)
    for ci, c in enumerate(centres):
        labels[c] = ci

    order = np.argsort(-rho)
    for i in order:
        if labels[i] == -1 and nearest[i] >= 0 and labels[nearest[i]] >= 0:
            labels[i] = labels[nearest[i]]

    # Second pass for any remaining unassigned
    for _ in range(N):
        changed = False
        for i in order:
            if labels[i] == -1 and nearest[i] >= 0 and labels[nearest[i]] >= 0:
                labels[i] = labels[nearest[i]]
                changed = True
        if not changed:
            break

    return labels, centres
