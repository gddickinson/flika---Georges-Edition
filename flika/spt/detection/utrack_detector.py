"""U-Track statistical significance particle detection.

Faithful replica of the original spt_batch_analysis plugin's UTrackDetector
class (Jaqaman et al. methodology).  Background estimation via percentile +
MAD, Gaussian pre-filtering (skimage preferred, scipy fallback), local maxima
via scipy maximum_filter with two-stage border exclusion, significance testing
against a normal model, and intensity-weighted centroid sub-pixel refinement.

The ``detect_particles_single_frame`` method returns a pandas DataFrame with
columns ``['x', 'y', 'intensity', 'frame']`` -- identical to the original
plugin output.

The ``detect_frame`` and ``detect_stack`` methods return numpy arrays for
backward compatibility with the rest of the SPT pipeline:
  - ``detect_frame``: (N, 3) array with columns ``[x, y, intensity]``
  - ``detect_stack``: (M, 4) array with columns ``[frame, x, y, intensity]``
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter, maximum_filter

from ...logger import logger


class UTrackDetector:
    """Statistical significance-based particle detector.

    This is a faithful replica of the original spt_batch_analysis plugin's
    UTrackDetector class.

    Parameters
    ----------
    psf_sigma : float
        Expected PSF width in pixels (default 1.5).
    alpha : float
        Significance threshold for detection (default 0.05).  Also accepted
        as ``alpha_threshold`` for compatibility with the original plugin API.
    min_intensity : float
        Minimum absolute intensity cutoff (default 0.0).
    """

    def __init__(self, psf_sigma=1.5, alpha=0.05, min_intensity=0.0,
                 alpha_threshold=None):
        self.psf_sigma = psf_sigma
        # Accept both 'alpha' and 'alpha_threshold' for compatibility
        self.alpha_threshold = alpha_threshold if alpha_threshold is not None else alpha
        self.min_intensity = min_intensity

    # -- Keep 'alpha' as a convenience property for callers that use it -----
    @property
    def alpha(self):
        return self.alpha_threshold

    @alpha.setter
    def alpha(self, value):
        self.alpha_threshold = value

    # -----------------------------------------------------------------------
    # Primary detection entry point (original plugin interface)
    # -----------------------------------------------------------------------

    def detect_particles_single_frame(self, image, frame_number=0):
        """Detect particles in a single 2D frame.

        Returns a DataFrame with columns ``['x', 'y', 'intensity', 'frame']``,
        exactly matching the original plugin output.

        Parameters
        ----------
        image : numpy.ndarray
            2D (or squeezable-to-2D) image.
        frame_number : int
            Frame index to record in the output DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns: x, y, intensity, frame.
        """
        try:
            if image.ndim > 2:
                image = np.squeeze(image)
            if image.ndim != 2:
                return self._empty_detection_result(frame_number)
            image = image.astype(np.float64)

            # Background estimation
            bg_mean, bg_std = self._estimate_background(image)

            # Gaussian pre-filter (prefer skimage, fall back to scipy)
            try:
                from skimage import filters
                filtered_img = filters.gaussian(image, sigma=self.psf_sigma)
            except Exception:
                filtered_img = gaussian_filter(
                    image.astype(np.float32), sigma=self.psf_sigma)

            # Find local maxima (prefer scipy method, fall back to manual)
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

            # Statistical significance testing
            intensities = filtered_img[max_coords]
            bg_mean_vals = (bg_mean[max_coords]
                           if isinstance(bg_mean, np.ndarray) else bg_mean)
            bg_std_vals = (bg_std[max_coords]
                          if isinstance(bg_std, np.ndarray) else bg_std)
            p_values = 1 - stats.norm.cdf(intensities, bg_mean_vals,
                                          bg_std_vals)
            significant_mask = p_values < self.alpha_threshold
            intensity_mask = intensities >= self.min_intensity
            final_mask = significant_mask & intensity_mask

            if not np.any(final_mask):
                return self._empty_detection_result(frame_number)

            y_coords = max_coords[0][final_mask]
            x_coords = max_coords[1][final_mask]

            # Sub-pixel localization
            sub_x, sub_y, sub_int = self._subpixel_localization(
                image, x_coords, y_coords)

            return pd.DataFrame({
                'x': sub_x, 'y': sub_y,
                'intensity': sub_int,
                'frame': frame_number
            })

        except Exception as e:
            logger.debug("UTrackDetector error on frame %d: %s",
                         frame_number, e)
            return self._empty_detection_result(frame_number)

    # -----------------------------------------------------------------------
    # Numpy-array interface (backward compatibility with SPT pipeline)
    # -----------------------------------------------------------------------

    def detect_frame(self, image):
        """Detect particles in a single 2D frame.

        Backward-compatible interface that returns a numpy array.

        Parameters
        ----------
        image : numpy.ndarray
            2D image array.

        Returns
        -------
        numpy.ndarray
            (N, 3) array with columns ``[x, y, intensity]``, or ``(0, 3)``
            if nothing is detected.
        """
        df = self.detect_particles_single_frame(image, frame_number=0)
        if df.empty:
            return np.empty((0, 3))
        return df[['x', 'y', 'intensity']].to_numpy()

    def detect_stack(self, stack, callback=None):
        """Detect particles in a 3D image stack.

        Parameters
        ----------
        stack : numpy.ndarray
            (T, H, W) array.  A 2D array is treated as a single frame.
        callback : callable, optional
            Progress callback ``callback(frame_idx)``.

        Returns
        -------
        numpy.ndarray
            (M, 4) array with columns ``[frame, x, y, intensity]``, or
            ``(0, 4)`` if nothing is detected.
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
    # Background estimation
    # -----------------------------------------------------------------------

    def _estimate_background(self, image):
        """Estimate background mean and standard deviation.

        Uses the 25th percentile as the mean estimate and the MAD-based
        robust standard deviation estimator.

        Parameters
        ----------
        image : numpy.ndarray
            2D float64 image.

        Returns
        -------
        tuple
            (bg_mean, bg_std) -- scalars.
        """
        bg_mean = np.percentile(image, 25)
        median_val = np.median(image)
        mad = np.median(np.abs(image - median_val))
        bg_std = mad * 1.4826
        return bg_mean, max(bg_std, 1.0)

    # -----------------------------------------------------------------------
    # Sub-pixel localization
    # -----------------------------------------------------------------------

    def _subpixel_localization(self, image, x_coords, y_coords):
        """Intensity-weighted centroid sub-pixel refinement.

        Uses a window of size ``int(2 * psf_sigma)`` around each candidate,
        exactly matching the original plugin.

        Parameters
        ----------
        image : numpy.ndarray
            Original (unfiltered) 2D image.
        x_coords, y_coords : numpy.ndarray
            Integer pixel coordinates of candidate detections.

        Returns
        -------
        tuple
            (sub_x, sub_y, sub_int) -- lists of refined coordinates and
            mean intensities.
        """
        sub_x, sub_y, sub_int = [], [], []
        h, w = image.shape
        window_size = int(2 * self.psf_sigma)

        for x, y in zip(x_coords, y_coords):
            r0 = max(0, y - window_size)
            r1 = min(h, y + window_size + 1)
            c0 = max(0, x - window_size)
            c1 = min(w, x + window_size + 1)
            region = image[r0:r1, c0:c1]
            total = region.sum()
            if total <= 0:
                sub_x.append(float(x))
                sub_y.append(float(y))
                sub_int.append(0.0)
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            cx = float(np.sum(xx * region) / total)
            cy = float(np.sum(yy * region) / total)
            intensity = float(total / region.size)
            sub_x.append(cx)
            sub_y.append(cy)
            sub_int.append(intensity)
        return sub_x, sub_y, sub_int

    # -----------------------------------------------------------------------
    # Local maxima finding
    # -----------------------------------------------------------------------

    def _find_local_maxima_scipy(self, image):
        """Find local maxima using scipy maximum_filter.

        Applies a two-stage border exclusion matching the original plugin:
        first the standard ``window_size // 2`` border, then an additional
        ``max(3, int(psf_sigma))`` border.

        Parameters
        ----------
        image : numpy.ndarray
            2D filtered image.

        Returns
        -------
        tuple
            ``(row_indices, col_indices)`` from ``np.where``.
        """
        window_size = max(3, int(2 * self.psf_sigma + 1))
        if window_size % 2 == 0:
            window_size += 1

        max_filt = maximum_filter(image, size=window_size)
        local_max = (image == max_filt)

        # Two-stage border exclusion (exact plugin behavior)
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
        """Fallback local maxima finding without two-stage border exclusion.

        Parameters
        ----------
        image : numpy.ndarray
            2D filtered image.
        window_size : int
            Odd integer window size for the maximum filter.

        Returns
        -------
        tuple
            ``(row_indices, col_indices)`` from ``np.where``.
        """
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
        return pd.DataFrame({'x': [], 'y': [], 'intensity': [], 'frame': []})
