"""Compressed Sensing STORM (CSSTORM) for high-density localization.

Implements compressed-sensing / sparse recovery approaches to resolve
overlapping emitters in dense super-resolution data, where conventional
single/multi-emitter fitting fails.

The core idea: discretize the field of view into a fine grid and solve
for a sparse vector of emitter intensities using L1-minimized
deconvolution (ISTA/FISTA).

Algorithm:
1. Define a fine grid (sub-pixel) of candidate emitter positions.
2. Build the forward model matrix A where each column is the PSF
   at that grid position.
3. Solve: minimize ‖Ax - b‖² + λ‖x‖₁   (LASSO / basis pursuit)
   using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
4. Extract emitter positions from non-zero entries.
5. Optionally refine positions with sub-grid MLE fitting.

References:
    Zhu et al., Nat Methods 9:721 (2012) — CSSTORM.
    Mukamel et al., Biophys J 102:2391 (2012) — statistical deconvolution.
    Beck & Teboulle, SIAM J Imaging Sci 2:183 (2009) — FISTA.
"""

import numpy as np


def _build_psf_dictionary(grid_x, grid_y, roi_shape, sigma):
    """Build PSF dictionary matrix A.

    Each column of A is a flattened Gaussian PSF centered at one grid
    position.

    Parameters
    ----------
    grid_x, grid_y : ndarray
        1D arrays of candidate x, y positions (sub-pixel).
    roi_shape : tuple
        (height, width) of the ROI in pixels.
    sigma : float
        PSF sigma in pixels.

    Returns
    -------
    A : ndarray
        (n_pixels, n_grid) dictionary matrix.
    grid_coords : ndarray
        (n_grid, 2) array of [x, y] positions.
    """
    h, w = roi_shape
    n_pixels = h * w
    yy, xx = np.mgrid[0:h, 0:w]
    xx_flat = xx.ravel().astype(np.float64)
    yy_flat = yy.ravel().astype(np.float64)

    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_coords = np.column_stack([gx.ravel(), gy.ravel()])
    n_grid = len(grid_coords)

    A = np.empty((n_pixels, n_grid), dtype=np.float64)
    two_s2 = 2.0 * sigma ** 2

    for k in range(n_grid):
        r2 = (xx_flat - grid_coords[k, 0]) ** 2 + \
             (yy_flat - grid_coords[k, 1]) ** 2
        col = np.exp(-r2 / two_s2)
        norm = col.sum()
        if norm > 0:
            col /= norm
        A[:, k] = col

    return A, grid_coords


def _soft_threshold(x, threshold):
    """Soft-thresholding (proximal operator for L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def fista_solve(A, b, lam, max_iter=500, tol=1e-6, non_negative=True):
    """Solve min ‖Ax - b‖² + λ‖x‖₁ using FISTA.

    Parameters
    ----------
    A : ndarray
        (m, n) dictionary matrix.
    b : ndarray
        (m,) observation vector.
    lam : float
        Regularization parameter (sparsity).
    max_iter : int
        Maximum iterations (default 500).
    tol : float
        Convergence tolerance on relative change (default 1e-6).
    non_negative : bool
        Enforce non-negativity (emitter intensities ≥ 0).

    Returns
    -------
    x : ndarray
        (n,) sparse solution.
    info : dict
        'iterations': int, 'converged': bool, 'residual': float.
    """
    m, n = A.shape
    b = b.ravel().astype(np.float64)

    # Lipschitz constant: L = largest eigenvalue of A^T A
    # For efficiency, use power iteration estimate
    AtA = A.T @ A
    # Quick estimate: L ≈ ‖A^T A‖_F / sqrt(n)  or use trace
    L = np.linalg.norm(AtA, ord=2)
    if L < 1e-12:
        return np.zeros(n), {'iterations': 0, 'converged': True,
                              'residual': float(np.linalg.norm(b))}

    step = 1.0 / L
    Atb = A.T @ b

    x = np.zeros(n, dtype=np.float64)
    y = x.copy()
    t = 1.0

    for iteration in range(max_iter):
        x_old = x.copy()

        # Gradient step
        grad = AtA @ y - Atb
        z = y - step * grad

        # Proximal (soft-thresholding)
        x = _soft_threshold(z, lam * step)

        if non_negative:
            x = np.maximum(x, 0.0)

        # FISTA momentum
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        y = x + (t - 1.0) / t_new * (x - x_old)
        t = t_new

        # Convergence check
        change = np.linalg.norm(x - x_old)
        if change < tol * max(np.linalg.norm(x), 1e-12):
            residual = float(np.linalg.norm(A @ x - b))
            return x, {'iterations': iteration + 1, 'converged': True,
                        'residual': residual}

    residual = float(np.linalg.norm(A @ x - b))
    return x, {'iterations': max_iter, 'converged': False,
                'residual': residual}


def extract_emitters(x_sparse, grid_coords, threshold_fraction=0.1):
    """Extract emitter positions and intensities from sparse solution.

    Parameters
    ----------
    x_sparse : ndarray
        (n_grid,) sparse solution from FISTA.
    grid_coords : ndarray
        (n_grid, 2) grid positions.
    threshold_fraction : float
        Minimum intensity as fraction of max (default 0.1).

    Returns
    -------
    ndarray
        (K, 3) array: [x, y, intensity] for each detected emitter.
    """
    if np.max(x_sparse) <= 0:
        return np.empty((0, 3), dtype=np.float64)

    thresh = threshold_fraction * np.max(x_sparse)
    active = x_sparse > thresh

    if not np.any(active):
        return np.empty((0, 3), dtype=np.float64)

    positions = grid_coords[active]
    intensities = x_sparse[active]

    # Merge nearby detections (within 1 grid spacing)
    result = []
    used = np.zeros(len(positions), dtype=bool)

    for i in range(len(positions)):
        if used[i]:
            continue
        cluster_x = [positions[i, 0] * intensities[i]]
        cluster_y = [positions[i, 1] * intensities[i]]
        cluster_int = [intensities[i]]
        used[i] = True

        for j in range(i + 1, len(positions)):
            if used[j]:
                continue
            d = np.sqrt((positions[i, 0] - positions[j, 0]) ** 2 +
                        (positions[i, 1] - positions[j, 1]) ** 2)
            if d < 1.5:  # merge radius
                cluster_x.append(positions[j, 0] * intensities[j])
                cluster_y.append(positions[j, 1] * intensities[j])
                cluster_int.append(intensities[j])
                used[j] = True

        total_int = sum(cluster_int)
        result.append([sum(cluster_x) / total_int,
                       sum(cluster_y) / total_int,
                       total_int])

    return np.array(result, dtype=np.float64)


class CSSTORMDetector:
    """Compressed Sensing STORM detector for high-density localization.

    Parameters
    ----------
    sigma : float
        PSF sigma in pixels (default 1.3).
    grid_step : float
        Sub-pixel grid spacing (default 0.5 pixels → 2x oversampling).
    lam : float
        L1 regularization parameter (default 1.0).
        Higher = sparser result. Typical range 0.1-10.
    roi_size : int
        Half-size of analysis ROI around each detected region (default 8).
    max_iter : int
        FISTA maximum iterations (default 500).
    merge_distance : float
        Distance to merge nearby detections in final output (pixels, default 1.0).
    """

    def __init__(self, sigma=1.3, grid_step=0.5, lam=1.0, roi_size=8,
                 max_iter=500, merge_distance=1.0):
        self.sigma = sigma
        self.grid_step = grid_step
        self.lam = lam
        self.roi_size = roi_size
        self.max_iter = max_iter
        self.merge_distance = merge_distance

    def detect_frame(self, image, initial_detections=None):
        """Run CSSTORM on a single frame.

        Parameters
        ----------
        image : ndarray
            2D image (float).
        initial_detections : ndarray, optional
            (N, 2) initial detection positions from a conventional detector.
            If provided, CSSTORM is run on ROIs around these positions.
            If None, runs on the full image (slower).

        Returns
        -------
        ndarray
            (K, 3) array: [x, y, intensity] for detected emitters.
        """
        image = np.asarray(image, dtype=np.float64)
        h, w = image.shape

        if initial_detections is not None and len(initial_detections) > 0:
            return self._detect_rois(image, initial_detections)
        else:
            return self._detect_full(image)

    def _detect_full(self, image):
        """Run CSSTORM on the full image."""
        h, w = image.shape
        grid_x = np.arange(0, w, self.grid_step)
        grid_y = np.arange(0, h, self.grid_step)

        A, grid_coords = _build_psf_dictionary(grid_x, grid_y, (h, w),
                                                self.sigma)
        b = image.ravel()

        # Subtract background estimate
        bg = np.percentile(b, 10)
        b_sub = np.maximum(b - bg, 0.0)

        x_sparse, info = fista_solve(A, b_sub, self.lam,
                                      max_iter=self.max_iter)

        return extract_emitters(x_sparse, grid_coords)

    def _detect_rois(self, image, initial_detections):
        """Run CSSTORM on ROIs around initial detections."""
        h, w = image.shape
        r = self.roi_size
        all_emitters = []

        for det in initial_detections:
            cx, cy = int(round(det[0])), int(round(det[1]))
            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)

            if x1 - x0 < 3 or y1 - y0 < 3:
                continue

            roi = image[y0:y1, x0:x1]
            roi_h, roi_w = roi.shape

            grid_x = np.arange(0, roi_w, self.grid_step)
            grid_y = np.arange(0, roi_h, self.grid_step)

            A, grid_coords = _build_psf_dictionary(
                grid_x, grid_y, (roi_h, roi_w), self.sigma)

            bg = np.percentile(roi, 10)
            b = np.maximum(roi.ravel() - bg, 0.0)

            x_sparse, info = fista_solve(A, b, self.lam,
                                          max_iter=self.max_iter)

            emitters = extract_emitters(x_sparse, grid_coords)
            if len(emitters) > 0:
                # Shift to global coordinates
                emitters[:, 0] += x0
                emitters[:, 1] += y0
                all_emitters.append(emitters)

        if not all_emitters:
            return np.empty((0, 3), dtype=np.float64)

        result = np.vstack(all_emitters)

        # Final merge of nearby detections from overlapping ROIs
        return self._merge_nearby(result)

    def _merge_nearby(self, emitters):
        """Merge detections within merge_distance."""
        if len(emitters) <= 1:
            return emitters

        used = np.zeros(len(emitters), dtype=bool)
        merged = []

        for i in range(len(emitters)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True

            for j in range(i + 1, len(emitters)):
                if used[j]:
                    continue
                d = np.sqrt((emitters[i, 0] - emitters[j, 0]) ** 2 +
                            (emitters[i, 1] - emitters[j, 1]) ** 2)
                if d < self.merge_distance:
                    cluster.append(j)
                    used[j] = True

            # Intensity-weighted centroid
            ints = emitters[cluster, 2]
            total = ints.sum()
            mx = np.sum(emitters[cluster, 0] * ints) / total
            my = np.sum(emitters[cluster, 1] * ints) / total
            merged.append([mx, my, total])

        return np.array(merged, dtype=np.float64)

    def detect_stack(self, stack, initial_detections_per_frame=None,
                     progress_callback=None):
        """Run CSSTORM on a 3D stack (T, H, W).

        Parameters
        ----------
        stack : ndarray
            3D array (n_frames, height, width).
        initial_detections_per_frame : dict, optional
            {frame_idx: (N, 2) array of initial positions}.
        progress_callback : callable, optional
            Called with (frame_idx, n_frames).

        Returns
        -------
        ndarray
            (K, 4) array: [frame, x, y, intensity].
        """
        n_frames = stack.shape[0]
        all_results = []

        for f in range(n_frames):
            init_det = None
            if initial_detections_per_frame is not None:
                init_det = initial_detections_per_frame.get(f)

            emitters = self.detect_frame(stack[f], init_det)
            if len(emitters) > 0:
                frame_col = np.full((len(emitters), 1), f, dtype=np.float64)
                all_results.append(np.hstack([frame_col, emitters]))

            if progress_callback is not None:
                progress_callback(f, n_frames)

        if not all_results:
            return np.empty((0, 4), dtype=np.float64)

        return np.vstack(all_results)


def auto_lambda(image, sigma, percentile=99):
    """Estimate a reasonable λ from image statistics.

    Uses the principle that λ should be proportional to the noise level
    to suppress spurious detections while preserving real emitters.

    Parameters
    ----------
    image : ndarray
        2D image.
    sigma : float
        PSF sigma in pixels.
    percentile : float
        Percentile for signal level estimation.

    Returns
    -------
    float
        Suggested λ value.
    """
    from scipy.ndimage import median_filter
    noise = np.std(image - median_filter(image, size=int(2 * sigma + 1)))
    signal = np.percentile(image, percentile) - np.median(image)
    # λ ≈ noise_level, scaled so that SNR~3 emitters are just detectable
    return max(noise * 0.5, signal * 0.01)
