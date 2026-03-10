# -*- coding: utf-8 -*-
"""Biological structure generation for microscopy simulation."""
import numpy as np
from scipy import ndimage


def generate_ellipsoid(volume_shape, center, radii, value=1.0):
    """Generate a 3D ellipsoid mask.

    Parameters
    ----------
    volume_shape : tuple
        (D, H, W) or (H, W) output shape.
    center : tuple
        Center coordinates matching volume dimensions.
    radii : tuple
        Radii along each axis.
    value : float
        Fill value inside the ellipsoid.

    Returns
    -------
    ndarray
        Volume with ellipsoid filled with `value`.
    """
    ndim = len(volume_shape)
    coords = np.ogrid[tuple(slice(0, s) for s in volume_shape)]
    dist = sum(((c - ctr) / max(r, 1e-10))**2
               for c, ctr, r in zip(coords, center, radii))
    result = np.zeros(volume_shape)
    result[dist <= 1.0] = value
    return result


def generate_cell_mask(shape, cell_type='round', **kwargs):
    """Generate a single cell mask.

    Parameters
    ----------
    shape : tuple
        (H, W) or (D, H, W) output shape.
    cell_type : str
        'round', 'elongated', 'irregular', 'neuron'.
    **kwargs
        Additional parameters (e.g., radius, aspect_ratio).

    Returns
    -------
    ndarray
        Binary cell mask.
    """
    is_3d = len(shape) == 3
    if is_3d:
        d, h, w = shape
        center = (d // 2, h // 2, w // 2)
    else:
        h, w = shape
        center = (h // 2, w // 2)

    if cell_type == 'round':
        radius = kwargs.get('radius', min(h, w) // 4)
        if is_3d:
            radii = (max(1, radius // 2), radius, radius)
        else:
            radii = (radius, radius)
        mask = generate_ellipsoid(shape, center, radii)

    elif cell_type == 'elongated':
        aspect = kwargs.get('aspect_ratio', 2.5)
        radius = kwargs.get('radius', min(h, w) // 5)
        if is_3d:
            radii = (max(1, radius // 2), radius, int(radius * aspect))
        else:
            radii = (radius, int(radius * aspect))
        mask = generate_ellipsoid(shape, center, radii)

    elif cell_type == 'irregular':
        radius = kwargs.get('radius', min(h, w) // 4)
        if is_3d:
            radii = (max(1, radius // 2), radius, radius)
        else:
            radii = (radius, radius)
        mask = generate_ellipsoid(shape, center, radii)
        # Perturb boundary with smooth noise
        noise = np.random.randn(*shape) * 0.3
        noise = ndimage.gaussian_filter(noise, sigma=radius / 3)
        # Distance from center
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        dist = np.sqrt(sum((c - ctr)**2 for c, ctr in zip(coords, center)))
        boundary = np.abs(dist - radius) < radius * 0.3
        mask = (dist + noise * radius * 0.3) < radius
        mask = mask.astype(float)

    elif cell_type == 'neuron':
        # Cell body + random dendrite-like extensions
        radius = kwargs.get('soma_radius', min(h, w) // 8)
        if is_3d:
            radii = (max(1, radius // 2), radius, radius)
        else:
            radii = (radius, radius)
        mask = generate_ellipsoid(shape, center, radii)
        # Add dendrites via random walks
        n_dendrites = kwargs.get('n_dendrites', 5)
        for _ in range(n_dendrites):
            mask = _add_dendrite(mask, center, radius, shape)
    else:
        raise ValueError(f"Unknown cell_type: {cell_type}")

    return (mask > 0.5).astype(float)


def _add_dendrite(mask, center, soma_radius, shape):
    """Add a dendrite-like extension to a cell mask via random walk."""
    ndim = len(shape)
    pos = np.array(center, dtype=float)
    # Random initial direction
    direction = np.random.randn(ndim)
    direction /= np.linalg.norm(direction) + 1e-10

    length = soma_radius * np.random.uniform(3, 8)
    step_size = 1.0
    thickness = max(1.0, soma_radius * 0.15)
    n_steps = int(length / step_size)

    for i in range(n_steps):
        # Random walk with persistence
        direction += np.random.randn(ndim) * 0.3
        direction /= np.linalg.norm(direction) + 1e-10
        pos += direction * step_size

        # Taper thickness
        t = thickness * (1 - 0.7 * i / max(n_steps, 1))
        t = max(0.5, t)

        # Draw sphere at position
        idx = tuple(int(round(p)) for p in pos)
        if all(0 <= idx[j] < shape[j] for j in range(ndim)):
            slices = tuple(
                slice(max(0, int(idx[j] - t * 2)),
                      min(shape[j], int(idx[j] + t * 2) + 1))
                for j in range(ndim)
            )
            sub = mask[slices]
            coords = np.ogrid[tuple(slice(0, s) for s in sub.shape)]
            offsets = tuple(max(0, int(idx[j] - t * 2)) for j in range(ndim))
            dist = np.sqrt(sum((c + o - idx[j])**2
                               for c, o, j in zip(coords, offsets,
                                                   range(ndim))))
            sub[dist <= t] = 1.0
        else:
            break
    return mask


def generate_nucleus(cell_mask, relative_size=0.3):
    """Generate a nucleus mask centered within a cell.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask.
    relative_size : float
        Nucleus radius as fraction of cell radius.

    Returns
    -------
    ndarray
        Binary nucleus mask.
    """
    shape = cell_mask.shape
    # Find center of mass of cell
    com = ndimage.center_of_mass(cell_mask)
    # Estimate cell radius
    cell_pixels = np.sum(cell_mask > 0)
    ndim = len(shape)
    if ndim == 3:
        radius = (cell_pixels * 3 / (4 * np.pi))**(1/3)
        nuc_r = radius * relative_size
        radii = (max(1, nuc_r * 0.7), nuc_r, nuc_r)
    else:
        radius = np.sqrt(cell_pixels / np.pi)
        nuc_r = radius * relative_size
        radii = (nuc_r, nuc_r)

    nucleus = generate_ellipsoid(shape, com, radii)
    # Constrain to cell
    nucleus *= cell_mask
    return (nucleus > 0.5).astype(float)


def generate_mitochondria(cell_mask, n_mito=20, length_range=(3, 15)):
    """Generate mitochondria-like tubular structures within a cell.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask (2D or 3D).
    n_mito : int
        Number of mitochondria.
    length_range : tuple
        (min, max) length in pixels.

    Returns
    -------
    ndarray
        Mitochondria mask.
    """
    shape = cell_mask.shape
    ndim = len(shape)
    result = np.zeros(shape)

    # Get valid cell positions
    positions = np.argwhere(cell_mask > 0)
    if len(positions) == 0:
        return result

    for _ in range(n_mito):
        # Random start within cell
        idx = np.random.randint(len(positions))
        pos = positions[idx].astype(float)

        direction = np.random.randn(ndim)
        direction /= np.linalg.norm(direction) + 1e-10

        length = np.random.uniform(*length_range)
        n_steps = int(length)
        thickness = np.random.uniform(0.5, 1.5)

        for _ in range(n_steps):
            direction += np.random.randn(ndim) * 0.2
            direction /= np.linalg.norm(direction) + 1e-10
            pos += direction

            idx_int = tuple(int(round(p)) for p in pos)
            if all(0 <= idx_int[j] < shape[j] for j in range(ndim)):
                if cell_mask[idx_int] > 0:
                    # Draw small sphere
                    r = int(np.ceil(thickness))
                    slices = tuple(
                        slice(max(0, idx_int[j] - r),
                              min(shape[j], idx_int[j] + r + 1))
                        for j in range(ndim)
                    )
                    sub = result[slices]
                    coords = np.ogrid[tuple(slice(0, s) for s in sub.shape)]
                    offsets = tuple(max(0, idx_int[j] - r) for j in range(ndim))
                    dist = np.sqrt(sum((c + o - idx_int[j])**2
                                       for c, o, j in zip(coords, offsets,
                                                           range(ndim))))
                    sub[dist <= thickness] = 1.0
                else:
                    break
            else:
                break
    return result


def generate_er(cell_mask, density=0.15):
    """Generate endoplasmic reticulum-like tubular network.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask.
    density : float
        Network density (0-1).

    Returns
    -------
    ndarray
        ER mask.
    """
    shape = cell_mask.shape
    # Create random network using thresholded smooth noise
    noise = np.random.randn(*shape)
    sigma = max(2, min(shape) // 20)
    smooth = ndimage.gaussian_filter(noise, sigma=sigma)
    # Create thin network by finding zero-crossings
    threshold = np.percentile(smooth[cell_mask > 0], (1 - density) * 100)
    er = np.abs(smooth - threshold) < 0.1 * np.std(smooth)
    er = er.astype(float) * cell_mask
    # Thin it
    er = ndimage.binary_dilation(er, iterations=1).astype(float) * cell_mask
    return er


def generate_vesicles(cell_mask, n_vesicles=50, radius_range=(1, 3)):
    """Generate spherical vesicles within a cell.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask.
    n_vesicles : int
        Number of vesicles.
    radius_range : tuple
        (min, max) radius in pixels.

    Returns
    -------
    ndarray
        Vesicle mask.
    """
    shape = cell_mask.shape
    ndim = len(shape)
    result = np.zeros(shape)
    positions = np.argwhere(cell_mask > 0)
    if len(positions) == 0:
        return result

    for _ in range(n_vesicles):
        idx = np.random.randint(len(positions))
        center = positions[idx]
        r = np.random.uniform(*radius_range)
        ri = int(np.ceil(r))
        slices = tuple(
            slice(max(0, center[j] - ri),
                  min(shape[j], center[j] + ri + 1))
            for j in range(ndim)
        )
        sub = result[slices]
        coords = np.ogrid[tuple(slice(0, s) for s in sub.shape)]
        offsets = tuple(max(0, center[j] - ri) for j in range(ndim))
        dist = np.sqrt(sum((c + o - center[j])**2
                           for c, o, j in zip(coords, offsets, range(ndim))))
        sub[dist <= r] = 1.0
    return result * cell_mask


def generate_filaments(shape, n_filaments=10, persistence_length=20.0,
                       thickness=1.0):
    """Generate filament-like structures using worm-like chain model.

    Parameters
    ----------
    shape : tuple
        (H, W) or (D, H, W) output shape.
    n_filaments : int
        Number of filaments.
    persistence_length : float
        Persistence length in pixels (higher = straighter).
    thickness : float
        Filament thickness in pixels.

    Returns
    -------
    ndarray
        Binary filament mask.
    """
    ndim = len(shape)
    result = np.zeros(shape)

    for _ in range(n_filaments):
        # Random start position
        pos = np.array([np.random.uniform(0, s) for s in shape], dtype=float)
        direction = np.random.randn(ndim)
        direction /= np.linalg.norm(direction) + 1e-10

        length = np.random.uniform(persistence_length * 0.5,
                                   persistence_length * 3)
        n_steps = int(length)

        for step in range(n_steps):
            # Worm-like chain: angular diffusion inversely proportional
            # to persistence length
            perturbation = np.random.randn(ndim) / persistence_length
            direction += perturbation
            direction /= np.linalg.norm(direction) + 1e-10
            pos += direction

            idx = tuple(int(round(p)) for p in pos)
            if all(0 <= idx[j] < shape[j] for j in range(ndim)):
                ri = int(np.ceil(thickness))
                slices = tuple(
                    slice(max(0, idx[j] - ri),
                          min(shape[j], idx[j] + ri + 1))
                    for j in range(ndim)
                )
                sub = result[slices]
                coords = np.ogrid[tuple(slice(0, s) for s in sub.shape)]
                offsets = tuple(max(0, idx[j] - ri) for j in range(ndim))
                dist = np.sqrt(sum((c + o - idx[j])**2
                                   for c, o, j in zip(coords, offsets,
                                                       range(ndim))))
                sub[dist <= thickness] = 1.0
            else:
                break
    return result


def generate_microtubule_network(cell_mask, n_tubules=15, mtoc=None):
    """Generate microtubule network radiating from MTOC.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask.
    n_tubules : int
        Number of microtubules.
    mtoc : tuple or None
        MTOC position; defaults to center of mass.

    Returns
    -------
    ndarray
        Microtubule network mask.
    """
    shape = cell_mask.shape
    ndim = len(shape)
    if mtoc is None:
        mtoc = ndimage.center_of_mass(cell_mask)

    result = np.zeros(shape)
    for _ in range(n_tubules):
        pos = np.array(mtoc, dtype=float)
        direction = np.random.randn(ndim)
        direction /= np.linalg.norm(direction) + 1e-10

        for step in range(int(min(shape) * 1.5)):
            direction += np.random.randn(ndim) * 0.05
            direction /= np.linalg.norm(direction) + 1e-10
            pos += direction

            idx = tuple(int(round(p)) for p in pos)
            if all(0 <= idx[j] < shape[j] for j in range(ndim)):
                if cell_mask[idx] > 0:
                    result[idx] = 1.0
                else:
                    break
            else:
                break
    # Dilate slightly for visibility
    result = ndimage.binary_dilation(result, iterations=1).astype(float)
    return result * cell_mask


def generate_actin_cortex(cell_mask, thickness=2):
    """Generate actin cortex at cell periphery.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask.
    thickness : int
        Cortex thickness in pixels.

    Returns
    -------
    ndarray
        Cortex mask.
    """
    eroded = ndimage.binary_erosion(cell_mask > 0, iterations=thickness)
    cortex = (cell_mask > 0).astype(float) - eroded.astype(float)
    return np.clip(cortex, 0, 1)


def generate_cell_field(shape, n_cells=20, cell_type='round',
                        spacing='random', labels=True):
    """Generate a field of cells with non-overlapping placement.

    Parameters
    ----------
    shape : tuple
        (H, W) or (D, H, W) output shape.
    n_cells : int
        Number of cells.
    cell_type : str
        Cell type for each cell.
    spacing : str
        'random' for Poisson disc sampling, 'grid' for regular grid.
    labels : bool
        If True, return labeled (each cell has unique ID) volume.

    Returns
    -------
    ndarray
        Labeled or binary cell field.
    """
    ndim = len(shape)
    result = np.zeros(shape)
    is_3d = ndim == 3

    if is_3d:
        d, h, w = shape
        spatial = (h, w)
    else:
        h, w = shape
        spatial = (h, w)

    # Estimate cell radius for spacing
    cell_radius = min(spatial) // (int(np.sqrt(n_cells)) + 2)
    cell_radius = max(5, cell_radius)

    # Generate positions via Poisson disc sampling
    positions = _poisson_disc_2d(spatial, cell_radius * 1.8, n_cells)

    for i, (cy, cx) in enumerate(positions):
        label_val = (i + 1) if labels else 1.0
        r = int(cell_radius * np.random.uniform(0.7, 1.0))
        if is_3d:
            cell_shape = (d, min(r * 4, h), min(r * 4, w))
        else:
            cell_shape = (min(r * 4, h), min(r * 4, w))

        cell = generate_cell_mask(cell_shape, cell_type=cell_type,
                                  radius=r)
        # Place cell at position
        if is_3d:
            y_start = max(0, int(cy) - cell_shape[1] // 2)
            x_start = max(0, int(cx) - cell_shape[2] // 2)
            y_end = min(h, y_start + cell_shape[1])
            x_end = min(w, x_start + cell_shape[2])
            cy_sub = cell_shape[1] // 2 - (int(cy) - cell_shape[1] // 2 - y_start)
            cx_sub = cell_shape[2] // 2 - (int(cx) - cell_shape[2] // 2 - x_start)
            sub = cell[:, :y_end - y_start, :x_end - x_start]
            region = result[:, y_start:y_end, x_start:x_end]
            mask = (sub > 0) & (region == 0)
            region[mask] = label_val
        else:
            y_start = max(0, int(cy) - cell_shape[0] // 2)
            x_start = max(0, int(cx) - cell_shape[1] // 2)
            y_end = min(h, y_start + cell_shape[0])
            x_end = min(w, x_start + cell_shape[1])
            ch, cw = y_end - y_start, x_end - x_start
            sub = cell[:ch, :cw]
            region = result[y_start:y_end, x_start:x_end]
            mask = (sub > 0) & (region == 0)
            region[mask] = label_val

    return result


def _poisson_disc_2d(shape, min_dist, n_points):
    """Simple Poisson disc sampling in 2D."""
    h, w = shape
    points = []
    max_attempts = n_points * 30
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        y = np.random.uniform(min_dist, h - min_dist)
        x = np.random.uniform(min_dist, w - min_dist)
        ok = True
        for py, px in points:
            if np.sqrt((y - py)**2 + (x - px)**2) < min_dist:
                ok = False
                break
        if ok:
            points.append((y, x))
        attempts += 1
    return points


def generate_bead_field(shape, n_beads=100, bead_radius=0.1,
                        positions=None):
    """Generate a field of fluorescent beads.

    Parameters
    ----------
    shape : tuple
        Output shape (2D or 3D).
    n_beads : int
        Number of beads.
    bead_radius : float
        Bead radius in pixels.
    positions : ndarray or None
        Predefined positions (N, ndim). If None, random placement.

    Returns
    -------
    ndarray
        Bead field (point sources or small spheres).
    """
    ndim = len(shape)
    result = np.zeros(shape)

    if positions is None:
        positions = np.column_stack(
            [np.random.uniform(2, s - 2, n_beads) for s in shape])

    for pos in positions:
        idx = tuple(int(round(p)) for p in pos)
        if all(0 <= idx[j] < shape[j] for j in range(ndim)):
            if bead_radius < 0.5:
                # Sub-pixel: single point
                result[idx] = 1.0
            else:
                ri = int(np.ceil(bead_radius)) + 1
                slices = tuple(
                    slice(max(0, idx[j] - ri),
                          min(shape[j], idx[j] + ri + 1))
                    for j in range(ndim)
                )
                sub = result[slices]
                coords = np.ogrid[tuple(slice(0, s) for s in sub.shape)]
                offsets = tuple(max(0, idx[j] - ri) for j in range(ndim))
                dist = np.sqrt(sum((c + o - pos[j])**2
                                   for c, o, j in zip(coords, offsets,
                                                       range(ndim))))
                sub[dist <= bead_radius] = 1.0
    return result
