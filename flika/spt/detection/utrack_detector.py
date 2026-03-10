"""U-Track statistical significance particle detection.

Implements the Jaqaman et al. detection methodology with:
  - Difference-of-Gaussians (DoG) bandpass pre-filtering
  - Local maxima finding via scipy maximum_filter
  - Statistical significance testing against local background
  - 2D Gaussian PSF fitting for sub-pixel localization
  - Mixture-model fitting for resolving overlapping particles
  - Localization uncertainty estimation (Cramér-Rao bound)

The ``detect_particles_single_frame`` method returns a pandas DataFrame with
columns ``['x', 'y', 'intensity', 'sigma', 'uncertainty', 'frame']``.

The ``detect_frame`` and ``detect_stack`` methods return numpy arrays for
backward compatibility with the rest of the SPT pipeline:
  - ``detect_frame``: (N, 3) array with columns ``[x, y, intensity]``
  - ``detect_stack``: (M, 4) array with columns ``[frame, x, y, intensity]``
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.ndimage import gaussian_filter, maximum_filter

from flika.logger import logger


# ---------------------------------------------------------------------------
# 2D Gaussian model for PSF fitting
# ---------------------------------------------------------------------------

def _gaussian_2d(coords, amplitude, x0, y0, sigma, background):
    """2D symmetric Gaussian model: A*exp(-r²/2σ²) + B."""
    y, x = coords
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma ** 2)) + background


def _gaussian_2d_residuals(params, coords, data):
    """Residuals for least-squares fitting."""
    return _gaussian_2d(coords, *params) - data


def _two_gaussian_2d(coords, a1, x1, y1, s1, a2, x2, y2, s2, bg):
    """Two overlapping 2D Gaussians + shared background."""
    y, x = coords
    g1 = a1 * np.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (2.0 * s1 ** 2))
    g2 = a2 * np.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (2.0 * s2 ** 2))
    return g1 + g2 + bg


def _two_gaussian_residuals(params, coords, data):
    """Residuals for 2-Gaussian mixture fit."""
    return _two_gaussian_2d(coords, *params) - data


def _fit_gaussian_psf(image, y0, x0, sigma_init, fit_radius):
    """Fit a single 2D Gaussian to a local patch.

    Parameters
    ----------
    image : 2D ndarray
        Full image (float64).
    y0, x0 : int
        Integer pixel coordinates of the candidate.
    sigma_init : float
        Initial PSF sigma estimate.
    fit_radius : int
        Half-size of the fitting window.

    Returns
    -------
    dict or None
        {'x': float, 'y': float, 'amplitude': float, 'sigma': float,
         'background': float, 'residual': float, 'uncertainty': float}
        Returns None if fit fails.
    """
    h, w = image.shape
    r0 = max(0, y0 - fit_radius)
    r1 = min(h, y0 + fit_radius + 1)
    c0 = max(0, x0 - fit_radius)
    c1 = min(w, x0 + fit_radius + 1)

    patch = image[r0:r1, c0:c1]
    if patch.size < 5:
        return None

    yy, xx = np.mgrid[r0:r1, c0:c1]
    coords = (yy.ravel(), xx.ravel())
    data = patch.ravel()

    # Initial guesses
    bg_init = float(np.percentile(patch, 25))
    amp_init = float(np.max(patch) - bg_init)
    if amp_init <= 0:
        return None

    p0 = [amp_init, float(x0), float(y0), sigma_init, bg_init]

    # Bounds: amplitude > 0, position within patch, sigma in [0.5, 3*sigma_init], bg >= 0
    lb = [0.0, c0 - 0.5, r0 - 0.5, 0.3, 0.0]
    ub = [amp_init * 5.0, c1 + 0.5, r1 + 0.5, sigma_init * 4.0, bg_init * 3.0 + 1.0]

    try:
        result = optimize.least_squares(
            _gaussian_2d_residuals, p0, args=(coords, data),
            bounds=(lb, ub), method='trf', max_nfev=100)

        if not result.success and result.cost > np.sum(data ** 2) * 0.5:
            return None

        amp, cx, cy, sigma_fit, bg = result.x
        residual = float(np.sqrt(result.cost / len(data)))

        # Cramér-Rao bound for localization uncertainty
        # σ_loc ≈ σ_psf / sqrt(N_photons) where N_photons ≈ 2π·A·σ²
        n_photons = max(2.0 * np.pi * amp * sigma_fit ** 2, 1.0)
        uncertainty = sigma_fit / np.sqrt(n_photons)

        return {
            'x': float(cx),
            'y': float(cy),
            'amplitude': float(amp),
            'sigma': float(sigma_fit),
            'background': float(bg),
            'residual': residual,
            'uncertainty': float(uncertainty),
        }
    except Exception:
        return None


def _try_mixture_fit(image, y1, x1, y2, x2, sigma_init, fit_radius):
    """Try fitting two overlapping Gaussians to a merged region.

    Parameters
    ----------
    image : 2D ndarray
    y1, x1, y2, x2 : int
        Pixel coordinates of the two candidate maxima.
    sigma_init : float
    fit_radius : int

    Returns
    -------
    list of dict, or None
        Two fit result dicts if successful, None otherwise.
    """
    h, w = image.shape
    # Bounding box covering both candidates
    r0 = max(0, min(y1, y2) - fit_radius)
    r1 = min(h, max(y1, y2) + fit_radius + 1)
    c0 = max(0, min(x1, x2) - fit_radius)
    c1 = min(w, max(x1, x2) + fit_radius + 1)

    patch = image[r0:r1, c0:c1]
    if patch.size < 9:
        return None

    yy, xx = np.mgrid[r0:r1, c0:c1]
    coords = (yy.ravel(), xx.ravel())
    data = patch.ravel()

    bg_init = float(np.percentile(patch, 10))
    a1_init = float(image[y1, x1] - bg_init)
    a2_init = float(image[y2, x2] - bg_init)
    if a1_init <= 0:
        a1_init = 1.0
    if a2_init <= 0:
        a2_init = 1.0

    p0 = [a1_init, float(x1), float(y1), sigma_init,
          a2_init, float(x2), float(y2), sigma_init, bg_init]

    lb = [0, c0 - 0.5, r0 - 0.5, 0.3,
          0, c0 - 0.5, r0 - 0.5, 0.3, 0]
    ub = [max(a1_init, a2_init) * 5, c1 + 0.5, r1 + 0.5, sigma_init * 4,
          max(a1_init, a2_init) * 5, c1 + 0.5, r1 + 0.5, sigma_init * 4,
          bg_init * 3 + 1]

    try:
        result = optimize.least_squares(
            _two_gaussian_residuals, p0, args=(coords, data),
            bounds=(lb, ub), method='trf', max_nfev=200)

        if not result.success:
            return None

        a1, cx1, cy1, s1, a2, cx2, cy2, s2, bg = result.x
        residual = float(np.sqrt(result.cost / len(data)))

        results = []
        for amp, cx, cy, sig in [(a1, cx1, cy1, s1), (a2, cx2, cy2, s2)]:
            if amp < 0.1:
                continue
            n_photons = max(2.0 * np.pi * amp * sig ** 2, 1.0)
            unc = sig / np.sqrt(n_photons)
            results.append({
                'x': float(cx), 'y': float(cy),
                'amplitude': float(amp), 'sigma': float(sig),
                'background': float(bg), 'residual': residual,
                'uncertainty': float(unc),
            })
        return results if len(results) == 2 else None
    except Exception:
        return None


class UTrackDetector:
    """Statistical significance-based particle detector.

    Implements the U-Track 2.5 detection pipeline: DoG bandpass filtering,
    local maxima finding, significance testing against local background,
    Gaussian PSF sub-pixel fitting, and mixture-model fitting for
    overlapping particles.

    Parameters
    ----------
    psf_sigma : float
        Expected PSF width in pixels (default 1.5).
    alpha : float
        Significance threshold for detection (default 0.05).  Also accepted
        as ``alpha_threshold`` for compatibility with the original plugin API.
    min_intensity : float
        Minimum absolute intensity cutoff (default 0.0).
    dog_ratio : float
        Ratio of large-to-small sigma for DoG bandpass (default 5.0).
        Set to 0 to disable DoG and use single Gaussian (legacy behavior).
    mixture_separation : float
        Maximum separation (in units of psf_sigma) for attempting
        mixture-model fitting of overlapping particles (default 3.0).
        Set to 0 to disable mixture fitting.
    local_bg_inner : float
        Inner radius (in psf_sigma) for local background annulus (default 3.0).
        Set to 0 to use global background estimation (legacy behavior).
    local_bg_outer : float
        Outer radius (in psf_sigma) for local background annulus (default 5.0).
    """

    def __init__(self, psf_sigma=1.5, alpha=0.05, min_intensity=0.0,
                 alpha_threshold=None, dog_ratio=5.0,
                 mixture_separation=3.0,
                 local_bg_inner=3.0, local_bg_outer=5.0):
        self.psf_sigma = psf_sigma
        self.alpha_threshold = alpha_threshold if alpha_threshold is not None else alpha
        self.min_intensity = min_intensity
        self.dog_ratio = dog_ratio
        self.mixture_separation = mixture_separation
        self.local_bg_inner = local_bg_inner
        self.local_bg_outer = local_bg_outer

    @property
    def alpha(self):
        return self.alpha_threshold

    @alpha.setter
    def alpha(self, value):
        self.alpha_threshold = value

    # -----------------------------------------------------------------------
    # Automatic parameter estimation
    # -----------------------------------------------------------------------

    @staticmethod
    def auto_estimate_psf_sigma(image, max_sigma=5.0):
        """Estimate PSF sigma from an image using autocorrelation.

        Fits a Gaussian to the central peak of the image autocorrelation
        to estimate the characteristic feature width.

        Parameters
        ----------
        image : 2D ndarray
            Representative image frame.
        max_sigma : float
            Maximum allowed sigma (default 5.0).

        Returns
        -------
        float
            Estimated PSF sigma in pixels.
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim > 2:
            image = np.squeeze(image)
        if image.ndim != 2:
            return 1.5  # fallback

        # Compute autocorrelation via FFT
        from scipy.fft import fft2, ifft2
        centered = image - np.mean(image)
        ft = fft2(centered)
        acf = np.real(ifft2(ft * np.conj(ft)))
        acf = np.fft.fftshift(acf)

        # Extract central region
        cy, cx = acf.shape[0] // 2, acf.shape[1] // 2
        r = min(15, cy, cx)
        patch = acf[cy - r:cy + r + 1, cx - r:cx + r + 1]
        patch = patch / patch.max()

        # Fit Gaussian to central peak
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        r2 = xx ** 2 + yy ** 2
        # Only use central part for fitting
        mask = r2 <= (r ** 2)
        vals = patch[mask]
        r2_vals = r2[mask].astype(np.float64)

        # Least-squares: log(vals) = -r²/(2σ²) => σ² = -r² / (2*log(vals))
        valid = vals > 0.1
        if np.sum(valid) < 3:
            return 1.5
        log_vals = np.log(np.clip(vals[valid], 1e-10, None))
        r2_fit = r2_vals[valid]

        # Linear regression: log(I) = a - r²/(2σ²)
        # y = a + b*x where b = -1/(2σ²), x = r²
        A = np.column_stack([np.ones(len(r2_fit)), r2_fit])
        try:
            result = np.linalg.lstsq(A, log_vals, rcond=None)
            b = result[0][1]
            if b < 0:
                sigma = np.sqrt(-1.0 / (2.0 * b))
                return float(np.clip(sigma, 0.5, max_sigma))
        except Exception:
            pass

        return 1.5  # fallback

    @staticmethod
    def auto_estimate_noise(image):
        """Estimate image noise level from median absolute deviation.

        Parameters
        ----------
        image : 2D ndarray

        Returns
        -------
        float
            Estimated noise standard deviation.
        """
        image = np.asarray(image, dtype=np.float64)
        if image.ndim > 2:
            image = np.squeeze(image)
        # Use high-pass residual (Laplacian) to estimate noise
        from scipy.ndimage import laplace
        residual = laplace(image)
        mad = np.median(np.abs(residual - np.median(residual)))
        # MAD to std conversion: σ = MAD * 1.4826
        # Laplacian amplifies noise by ~sqrt(20) for 3x3 kernel
        noise_std = mad * 1.4826 / np.sqrt(20.0)
        return max(float(noise_std), 1e-6)

    # -----------------------------------------------------------------------
    # Primary detection entry point
    # -----------------------------------------------------------------------

    def detect_particles_single_frame(self, image, frame_number=0):
        """Detect particles in a single 2D frame.

        Returns a DataFrame with columns
        ``['x', 'y', 'intensity', 'sigma', 'uncertainty', 'frame']``.

        Parameters
        ----------
        image : numpy.ndarray
            2D (or squeezable-to-2D) image.
        frame_number : int
            Frame index to record in the output DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        try:
            if image.ndim > 2:
                image = np.squeeze(image)
            if image.ndim != 2:
                return self._empty_detection_result(frame_number)
            image = image.astype(np.float64)

            # DoG bandpass filtering
            filtered_img = self._bandpass_filter(image)

            # Find local maxima
            try:
                max_coords = self._find_local_maxima_scipy(filtered_img)
            except Exception:
                window_size = max(3, int(2 * self.psf_sigma + 1))
                if window_size % 2 == 0:
                    window_size += 1
                max_coords = self._find_local_maxima_manual(
                    filtered_img, window_size)

            if len(max_coords[0]) == 0:
                return self._empty_detection_result(frame_number)

            # Statistical significance testing with local background
            y_candidates = max_coords[0]
            x_candidates = max_coords[1]
            significant_mask = self._significance_test(
                filtered_img, image, y_candidates, x_candidates)

            if not np.any(significant_mask):
                return self._empty_detection_result(frame_number)

            y_sig = y_candidates[significant_mask]
            x_sig = x_candidates[significant_mask]

            # Gaussian PSF fitting for sub-pixel localization
            fit_radius = max(int(np.ceil(3 * self.psf_sigma)), 3)
            detections = self._fit_all_candidates(
                image, y_sig, x_sig, fit_radius)

            # Mixture-model fitting for overlapping particles
            if self.mixture_separation > 0 and len(detections) > 0:
                detections = self._resolve_overlaps(
                    image, detections, fit_radius)

            if not detections:
                return self._empty_detection_result(frame_number)

            return pd.DataFrame({
                'x': [d['x'] for d in detections],
                'y': [d['y'] for d in detections],
                'intensity': [d['amplitude'] for d in detections],
                'sigma': [d['sigma'] for d in detections],
                'uncertainty': [d['uncertainty'] for d in detections],
                'frame': frame_number,
            })

        except Exception as e:
            logger.debug("UTrackDetector error on frame %d: %s",
                         frame_number, e)
            return self._empty_detection_result(frame_number)

    # -----------------------------------------------------------------------
    # Numpy-array interface (backward compatibility)
    # -----------------------------------------------------------------------

    def detect_frame(self, image):
        """Detect particles in a single 2D frame.

        Backward-compatible interface returning (N, 3) array [x, y, intensity].
        """
        df = self.detect_particles_single_frame(image, frame_number=0)
        if df.empty:
            return np.empty((0, 3))
        return df[['x', 'y', 'intensity']].to_numpy()

    def detect_stack(self, stack, callback=None):
        """Detect particles in a 3D image stack.

        Returns (M, 4) array with columns [frame, x, y, intensity].
        """
        if stack.ndim == 2:
            stack = stack[np.newaxis]

        all_detections = []
        for t in range(len(stack)):
            dets = self.detect_frame(stack[t])
            if len(dets) > 0:
                frame_col = np.full((len(dets), 1), t)
                all_detections.append(np.hstack([frame_col, dets]))
            if callback is not None:
                callback(t)

        if all_detections:
            return np.vstack(all_detections)
        return np.empty((0, 4))

    # -----------------------------------------------------------------------
    # DoG bandpass filtering
    # -----------------------------------------------------------------------

    def _bandpass_filter(self, image):
        """Apply Difference-of-Gaussians bandpass filter.

        Enhances features at PSF scale while suppressing both high-frequency
        noise and low-frequency background gradients.

        Falls back to single Gaussian if dog_ratio <= 0.
        """
        sigma_small = self.psf_sigma
        if self.dog_ratio > 0:
            sigma_large = sigma_small * self.dog_ratio
            g_small = gaussian_filter(image, sigma=sigma_small)
            g_large = gaussian_filter(image, sigma=sigma_large)
            filtered = g_small - g_large
        else:
            # Legacy: single Gaussian
            filtered = gaussian_filter(image, sigma=sigma_small)
        return filtered

    # -----------------------------------------------------------------------
    # Background estimation
    # -----------------------------------------------------------------------

    def _estimate_background_global(self, image):
        """Global background: 25th percentile + MAD std (legacy)."""
        bg_mean = float(np.percentile(image, 25))
        median_val = float(np.median(image))
        mad = float(np.median(np.abs(image - median_val)))
        bg_std = mad * 1.4826
        return bg_mean, max(bg_std, 1.0)

    def _estimate_background_local(self, image, y, x):
        """Local background from annular region around (y, x).

        Returns (bg_mean, bg_std) estimated from pixels in the annulus
        [inner_radius, outer_radius] around the candidate.
        """
        inner_r = self.local_bg_inner * self.psf_sigma
        outer_r = self.local_bg_outer * self.psf_sigma
        h, w = image.shape

        # Build annular mask
        r0 = max(0, int(y - outer_r))
        r1 = min(h, int(y + outer_r + 1))
        c0 = max(0, int(x - outer_r))
        c1 = min(w, int(x + outer_r + 1))

        yy, xx = np.mgrid[r0:r1, c0:c1]
        dist2 = (yy - y) ** 2 + (xx - x) ** 2
        annulus_mask = (dist2 >= inner_r ** 2) & (dist2 <= outer_r ** 2)

        if np.sum(annulus_mask) < 8:
            # Not enough pixels; fall back to global
            return self._estimate_background_global(image)

        pixels = image[r0:r1, c0:c1][annulus_mask]
        bg_mean = float(np.mean(pixels))
        bg_std = float(np.std(pixels))
        return bg_mean, max(bg_std, 1.0)

    # -----------------------------------------------------------------------
    # Statistical significance testing
    # -----------------------------------------------------------------------

    def _significance_test(self, filtered_img, raw_image, y_coords, x_coords):
        """Test each candidate against local or global background.

        Returns boolean mask of significant detections.
        """
        n = len(y_coords)
        mask = np.zeros(n, dtype=bool)

        if self.local_bg_inner > 0:
            # Local background per candidate
            for i in range(n):
                bg_mean, bg_std = self._estimate_background_local(
                    filtered_img, y_coords[i], x_coords[i])
                intensity = filtered_img[y_coords[i], x_coords[i]]
                p_value = 1.0 - stats.norm.cdf(intensity, bg_mean, bg_std)
                if p_value < self.alpha_threshold and intensity >= self.min_intensity:
                    mask[i] = True
        else:
            # Global background (legacy)
            bg_mean, bg_std = self._estimate_background_global(filtered_img)
            intensities = filtered_img[y_coords, x_coords]
            p_values = 1.0 - stats.norm.cdf(intensities, bg_mean, bg_std)
            mask = (p_values < self.alpha_threshold) & (intensities >= self.min_intensity)

        return mask

    # -----------------------------------------------------------------------
    # Gaussian PSF fitting
    # -----------------------------------------------------------------------

    def _fit_all_candidates(self, image, y_coords, x_coords, fit_radius):
        """Fit Gaussian PSF to each candidate location.

        Falls back to intensity-weighted centroid if fit fails.
        """
        detections = []
        for y, x in zip(y_coords, x_coords):
            result = _fit_gaussian_psf(
                image, int(y), int(x), self.psf_sigma, fit_radius)

            if result is not None:
                detections.append(result)
            else:
                # Fallback: intensity-weighted centroid
                centroid = self._centroid_fallback(image, int(x), int(y))
                if centroid is not None:
                    detections.append(centroid)

        return detections

    def _centroid_fallback(self, image, x, y):
        """Intensity-weighted centroid (legacy fallback)."""
        h, w = image.shape
        ws = int(2 * self.psf_sigma)
        r0 = max(0, y - ws)
        r1 = min(h, y + ws + 1)
        c0 = max(0, x - ws)
        c1 = min(w, x + ws + 1)
        region = image[r0:r1, c0:c1]
        total = region.sum()
        if total <= 0:
            return None
        yy, xx = np.mgrid[r0:r1, c0:c1]
        cx = float(np.sum(xx * region) / total)
        cy = float(np.sum(yy * region) / total)
        return {
            'x': cx, 'y': cy,
            'amplitude': float(np.max(region)),
            'sigma': self.psf_sigma,
            'background': float(np.percentile(region, 10)),
            'residual': 0.0,
            'uncertainty': self.psf_sigma,  # poor estimate for centroid
        }

    # -----------------------------------------------------------------------
    # Mixture-model fitting for overlapping particles
    # -----------------------------------------------------------------------

    def _resolve_overlaps(self, image, detections, fit_radius):
        """Check for overlapping detections and attempt mixture-model fit.

        When two detections are within mixture_separation * psf_sigma of
        each other, try a 2-Gaussian mixture fit. If the mixture model
        provides a significantly better fit (F-test), replace both
        single-Gaussian results with the mixture results.
        """
        sep_threshold = self.mixture_separation * self.psf_sigma
        n = len(detections)
        if n < 2:
            return detections

        # Find close pairs
        positions = np.array([[d['y'], d['x']] for d in detections])
        replaced = set()
        new_detections = []

        for i in range(n):
            if i in replaced:
                continue
            merged = False
            for j in range(i + 1, n):
                if j in replaced:
                    continue
                dist = np.sqrt((positions[i, 0] - positions[j, 0]) ** 2 +
                               (positions[i, 1] - positions[j, 1]) ** 2)
                if dist < sep_threshold:
                    # Try mixture fit
                    mix_result = _try_mixture_fit(
                        image,
                        int(round(positions[i, 0])),
                        int(round(positions[i, 1])),
                        int(round(positions[j, 0])),
                        int(round(positions[j, 1])),
                        self.psf_sigma, fit_radius)

                    if mix_result is not None:
                        # F-test: compare mixture residual to single residuals
                        single_res = (detections[i]['residual'] +
                                      detections[j]['residual']) / 2
                        mix_res = mix_result[0]['residual']

                        # Accept mixture if residual is lower (simpler test
                        # than formal F-test; avoids over-fitting since
                        # mixture has more parameters)
                        if mix_res < single_res * 0.9:
                            new_detections.extend(mix_result)
                            replaced.add(i)
                            replaced.add(j)
                            merged = True
                            break

            if not merged and i not in replaced:
                new_detections.append(detections[i])

        # Add any remaining un-replaced detections
        for j in range(n):
            if j not in replaced and detections[j] not in new_detections:
                new_detections.append(detections[j])

        return new_detections

    # -----------------------------------------------------------------------
    # Local maxima finding
    # -----------------------------------------------------------------------

    def _find_local_maxima_scipy(self, image):
        """Find local maxima using scipy maximum_filter with border exclusion."""
        window_size = max(3, int(2 * self.psf_sigma + 1))
        if window_size % 2 == 0:
            window_size += 1

        max_filt = maximum_filter(image, size=window_size)
        local_max = (image == max_filt)

        border = window_size // 2
        local_max[:border, :] = False
        local_max[-border:, :] = False
        local_max[:, :border] = False
        local_max[:, -border:] = False

        extra_border = max(3, int(self.psf_sigma))
        local_max[:extra_border, :] = False
        local_max[-extra_border:, :] = False
        local_max[:, :extra_border] = False
        local_max[:, -extra_border:] = False

        return np.where(local_max)

    def _find_local_maxima_manual(self, image, window_size):
        """Fallback local maxima finding."""
        max_filt = maximum_filter(image, size=window_size)
        local_max = (image == max_filt)
        border = window_size // 2
        local_max[:border, :] = False
        local_max[-border:, :] = False
        local_max[:, :border] = False
        local_max[:, -border:] = False
        return np.where(local_max)

    # -----------------------------------------------------------------------
    # Empty result helper
    # -----------------------------------------------------------------------

    def _empty_detection_result(self, frame_number):
        """Return an empty DataFrame with the standard detection columns."""
        return pd.DataFrame({
            'x': [], 'y': [], 'intensity': [], 'sigma': [],
            'uncertainty': [], 'frame': []
        })
