"""3D astigmatic Gaussian PSF model for ThunderSTORM-style localization.

Implements the defocused astigmatic PSF where a cylindrical lens introduces
z-dependent widths σ_x(z) and σ_y(z):

    σ_x(z) = σ₀ √(1 + ((z - c_x) / d_x)² + a_x ((z - c_x) / d_x)³)
    σ_y(z) = σ₀ √(1 + ((z - c_y) / d_y)² + a_y ((z - c_y) / d_y)³)

The calibration curve is fitted from z-stack bead data.  Once calibrated,
a 3D MLE fitter estimates (x, y, z, I, bg) per emitter by minimizing
the negative Poisson log-likelihood of the astigmatic Gaussian PSF model.

References:
    Huang et al., Science 319:810 (2008).
    Holtzer et al., Appl Phys Lett 90:053902 (2007).
"""

import numpy as np
from scipy.optimize import least_squares, minimize


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

class AstigmaticCalibration:
    """Z-calibration from measured σ_x(z), σ_y(z) vs. z-position.

    Parameters
    ----------
    sigma0 : float
        In-focus PSF sigma (pixels).
    cx, cy : float
        Focal offsets for x and y channels (nm).
    dx, dy : float
        Depth-of-focus parameters (nm).
    ax, ay : float
        Third-order correction coefficients (dimensionless).
    """

    def __init__(self, sigma0=1.3, cx=0.0, cy=0.0,
                 dx=400.0, dy=400.0, ax=0.0, ay=0.0):
        self.sigma0 = sigma0
        self.cx = cx
        self.cy = cy
        self.dx = dx
        self.dy = dy
        self.ax = ax
        self.ay = ay

    def sigma_x(self, z):
        """Compute σ_x(z)."""
        u = (z - self.cx) / self.dx
        return self.sigma0 * np.sqrt(1.0 + u ** 2 + self.ax * u ** 3)

    def sigma_y(self, z):
        """Compute σ_y(z)."""
        u = (z - self.cy) / self.dy
        return self.sigma0 * np.sqrt(1.0 + u ** 2 + self.ay * u ** 3)

    def sigma_xy(self, z):
        """Return (σ_x(z), σ_y(z)) arrays."""
        return self.sigma_x(z), self.sigma_y(z)

    @classmethod
    def fit_from_beads(cls, z_positions, sigma_x_measured, sigma_y_measured,
                       sigma0_init=1.3):
        """Fit calibration curve from bead z-stack measurements.

        Parameters
        ----------
        z_positions : array-like
            Known z positions (nm).
        sigma_x_measured, sigma_y_measured : array-like
            Measured PSF widths at each z position (pixels).
        sigma0_init : float
            Initial guess for in-focus sigma.

        Returns
        -------
        AstigmaticCalibration
            Fitted calibration object.
        """
        z = np.asarray(z_positions, dtype=np.float64)
        sx = np.asarray(sigma_x_measured, dtype=np.float64)
        sy = np.asarray(sigma_y_measured, dtype=np.float64)

        def _residuals(p):
            sigma0, cx, cy, dx, dy, ax, ay = p
            cal = cls(sigma0, cx, cy, dx, dy, ax, ay)
            rx = cal.sigma_x(z) - sx
            ry = cal.sigma_y(z) - sy
            return np.concatenate([rx, ry])

        p0 = [sigma0_init, 0.0, 0.0, 400.0, 400.0, 0.0, 0.0]
        result = least_squares(_residuals, p0, method='lm')
        p = result.x
        return cls(sigma0=p[0], cx=p[1], cy=p[2],
                   dx=p[3], dy=p[4], ax=p[5], ay=p[6])

    def to_dict(self):
        """Serialize to dict."""
        return dict(sigma0=self.sigma0, cx=self.cx, cy=self.cy,
                    dx=self.dx, dy=self.dy, ax=self.ax, ay=self.ay)

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dict."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Astigmatic Gaussian PSF model
# ---------------------------------------------------------------------------

def astigmatic_gaussian_psf(params, x_grid, y_grid, calibration):
    """Evaluate the integrated astigmatic Gaussian PSF.

    Parameters
    ----------
    params : array-like
        [x0, y0, z0, intensity, background] — emitter parameters.
    x_grid, y_grid : ndarray
        Pixel coordinate grids (from np.mgrid or meshgrid).
    calibration : AstigmaticCalibration
        Z-calibration curve.

    Returns
    -------
    ndarray
        Expected photon counts per pixel.
    """
    x0, y0, z0, intensity, bg = params
    sx = calibration.sigma_x(z0)
    sy = calibration.sigma_y(z0)

    # Erf-based pixel integration for accurate sub-pixel model
    from scipy.special import erf
    sqrt2 = np.sqrt(2.0)

    # Pixel edges: pixel centered at integer coords, edges at ±0.5
    ex = (erf((x_grid + 0.5 - x0) / (sqrt2 * sx)) -
          erf((x_grid - 0.5 - x0) / (sqrt2 * sx))) / 2.0
    ey = (erf((y_grid + 0.5 - y0) / (sqrt2 * sy)) -
          erf((y_grid - 0.5 - y0) / (sqrt2 * sy))) / 2.0

    return intensity * ex * ey + bg


def astigmatic_gaussian_jacobian(params, x_grid, y_grid, calibration):
    """Analytical Jacobian of the astigmatic PSF w.r.t. params.

    Returns
    -------
    list of ndarray
        [dM/dx0, dM/dy0, dM/dz0, dM/dI, dM/dbg], each same shape as x_grid.
    """
    from scipy.special import erf
    x0, y0, z0, intensity, bg = params
    sx = calibration.sigma_x(z0)
    sy = calibration.sigma_y(z0)
    sqrt2 = np.sqrt(2.0)
    sqrt2pi = np.sqrt(2.0 * np.pi)

    # Erf terms
    ex = (erf((x_grid + 0.5 - x0) / (sqrt2 * sx)) -
          erf((x_grid - 0.5 - x0) / (sqrt2 * sx))) / 2.0
    ey = (erf((y_grid + 0.5 - y0) / (sqrt2 * sy)) -
          erf((y_grid - 0.5 - y0) / (sqrt2 * sy))) / 2.0

    # Gaussian terms for derivatives
    def _gauss(u, s):
        return np.exp(-u ** 2 / (2.0 * s ** 2)) / (sqrt2pi * s)

    gx_lo = _gauss(x_grid - 0.5 - x0, sx)
    gx_hi = _gauss(x_grid + 0.5 - x0, sx)
    gy_lo = _gauss(y_grid - 0.5 - y0, sy)
    gy_hi = _gauss(y_grid + 0.5 - y0, sy)

    dex_dx = gx_lo - gx_hi  # derivative of ex w.r.t. x0
    dey_dy = gy_lo - gy_hi  # derivative of ey w.r.t. y0

    # d/dz requires chain rule through sigma_x(z), sigma_y(z)
    # dsigma_x/dz
    ux = (z0 - calibration.cx) / calibration.dx
    dux_dz = 1.0 / calibration.dx
    inner_x = 1.0 + ux ** 2 + calibration.ax * ux ** 3
    dinnerdz_x = (2.0 * ux + 3.0 * calibration.ax * ux ** 2) * dux_dz
    dsx_dz = calibration.sigma0 * 0.5 * dinnerdz_x / np.sqrt(max(inner_x, 1e-12))

    uy = (z0 - calibration.cy) / calibration.dy
    duy_dz = 1.0 / calibration.dy
    inner_y = 1.0 + uy ** 2 + calibration.ay * uy ** 3
    dinnerdz_y = (2.0 * uy + 3.0 * calibration.ay * uy ** 2) * duy_dz
    dsy_dz = calibration.sigma0 * 0.5 * dinnerdz_y / np.sqrt(max(inner_y, 1e-12))

    # dex/dsigma_x * dsigma_x/dz
    dex_dsx = ((x_grid - 0.5 - x0) * gx_lo -
               (x_grid + 0.5 - x0) * gx_hi) / sx
    dey_dsy = ((y_grid - 0.5 - y0) * gy_lo -
               (y_grid + 0.5 - y0) * gy_hi) / sy

    dM_dx = intensity * dex_dx * ey
    dM_dy = intensity * ex * dey_dy
    dM_dz = intensity * (dex_dsx * dsx_dz * ey + ex * dey_dsy * dsy_dz)
    dM_dI = ex * ey
    dM_dbg = np.ones_like(x_grid)

    return [dM_dx, dM_dy, dM_dz, dM_dI, dM_dbg]


# ---------------------------------------------------------------------------
# 3D MLE Fitter
# ---------------------------------------------------------------------------

class AstigmaticFitter3D:
    """Maximum likelihood fitter for 3D astigmatic PSF localization.

    Uses Poisson MLE: minimizes Σ[model - data·ln(model)] over
    the ROI around each candidate emitter.

    Parameters
    ----------
    calibration : AstigmaticCalibration
        Z-calibration curve.
    roi_size : int
        Half-size of fitting ROI (default 4 → 9×9 pixels).
    z_range : tuple
        (z_min, z_max) allowed range in nm (default (-800, 800)).
    max_iterations : int
        Maximum LM iterations (default 100).
    """

    def __init__(self, calibration, roi_size=4, z_range=(-800, 800),
                 max_iterations=100):
        self.calibration = calibration
        self.roi_size = roi_size
        self.z_range = z_range
        self.max_iterations = max_iterations

    def fit_emitter(self, image, x_init, y_init, z_init=0.0):
        """Fit a single emitter using Poisson MLE.

        Parameters
        ----------
        image : ndarray
            2D image (float).
        x_init, y_init : float
            Initial position estimate (pixels).
        z_init : float
            Initial z estimate (nm).  Default 0.

        Returns
        -------
        dict or None
            {'x': float, 'y': float, 'z': float, 'intensity': float,
             'background': float, 'sigma_x': float, 'sigma_y': float,
             'crlb': ndarray (5,)} or None if fit fails.
        """
        h, w = image.shape
        r = self.roi_size
        ix, iy = int(round(x_init)), int(round(y_init))

        x0 = max(0, ix - r)
        x1 = min(w, ix + r + 1)
        y0 = max(0, iy - r)
        y1 = min(h, iy + r + 1)

        if x1 - x0 < 3 or y1 - y0 < 3:
            return None

        roi = image[y0:y1, x0:x1].astype(np.float64)
        yg, xg = np.mgrid[y0:y1, x0:x1]
        xg = xg.astype(np.float64)
        yg = yg.astype(np.float64)

        # Initial estimates
        bg0 = max(np.percentile(roi, 10), 0.1)
        intensity0 = max(float(np.sum(roi) - bg0 * roi.size), 1.0)

        p0 = np.array([x_init, y_init, z_init, intensity0, bg0])

        def neg_poisson_loglik(p):
            model = astigmatic_gaussian_psf(p, xg, yg, self.calibration)
            model = np.maximum(model, 1e-10)
            return np.sum(model - roi * np.log(model))

        bounds = ([x0 - 0.5, y0 - 0.5, self.z_range[0], 0.1, 0.001],
                  [x1 + 0.5, y1 + 0.5, self.z_range[1], 1e8, 1e6])

        try:
            result = minimize(neg_poisson_loglik, p0, method='L-BFGS-B',
                              bounds=list(zip(bounds[0], bounds[1])),
                              options={'maxiter': self.max_iterations,
                                       'ftol': 1e-10})
        except Exception:
            return None

        if not result.success and result.fun > neg_poisson_loglik(p0):
            return None

        pf = result.x
        sx = self.calibration.sigma_x(pf[2])
        sy = self.calibration.sigma_y(pf[2])

        # Cramér-Rao lower bound from Fisher information
        crlb = self._compute_crlb(pf, xg, yg, roi)

        return {
            'x': pf[0], 'y': pf[1], 'z': pf[2],
            'intensity': pf[3], 'background': pf[4],
            'sigma_x': sx, 'sigma_y': sy,
            'crlb': crlb,
        }

    def _compute_crlb(self, params, xg, yg, data):
        """Compute CRLB from Poisson Fisher information matrix."""
        model = astigmatic_gaussian_psf(params, xg, yg, self.calibration)
        model = np.maximum(model, 1e-10)
        jac = astigmatic_gaussian_jacobian(params, xg, yg, self.calibration)

        fisher = np.zeros((5, 5))
        for i in range(5):
            for j in range(i, 5):
                val = np.sum(jac[i] * jac[j] / model)
                fisher[i, j] = val
                fisher[j, i] = val

        try:
            crlb = np.sqrt(np.diag(np.linalg.inv(fisher)))
        except np.linalg.LinAlgError:
            crlb = np.full(5, np.inf)

        return crlb

    def fit_frame(self, image, candidates):
        """Fit all candidate emitters in a frame.

        Parameters
        ----------
        image : ndarray
            2D image.
        candidates : ndarray
            (N, 2+) array with columns [x, y, ...].

        Returns
        -------
        list of dict
            Successful fit results.
        """
        results = []
        for i in range(len(candidates)):
            x0, y0 = candidates[i, 0], candidates[i, 1]
            z0 = candidates[i, 2] if candidates.shape[1] > 2 else 0.0
            r = self.fit_emitter(image, x0, y0, z0)
            if r is not None:
                results.append(r)
        return results

    def results_to_array(self, results):
        """Convert list of fit dicts to (N, 9) array.

        Columns: x, y, z, intensity, background, sigma_x, sigma_y,
                 crlb_x, crlb_y.
        """
        if not results:
            return np.empty((0, 9), dtype=np.float64)

        out = np.empty((len(results), 9), dtype=np.float64)
        for i, r in enumerate(results):
            out[i] = [r['x'], r['y'], r['z'], r['intensity'],
                      r['background'], r['sigma_x'], r['sigma_y'],
                      r['crlb'][0], r['crlb'][1]]
        return out
