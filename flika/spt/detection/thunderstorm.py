"""ThunderSTORM-style particle detection pipeline.

Faithful Python replica of the ThunderSTORM ImageJ plugin
(Ovesny et al., Bioinformatics 2014) detection pipeline, ported from the
thunderstorm_python plugin codebase.

Pipeline stages:

1. **Image filtering** -- wavelet (a trous B-spline), Gaussian,
   Difference-of-Gaussians, lowered Gaussian, difference of averaging,
   median, box, or no filter.
2. **Threshold evaluation** -- support for ThunderSTORM expressions such as
   ``'std(Wave.F1)'``, ``'2*std(Wave.F1)'``, or numeric values.
3. **Candidate detection** -- local maximum (peak_local_max /
   maximum_filter), non-maximum suppression, centroid (connected components
   with optional watershed), or grid detector.
4. **Sub-pixel fitting** -- erf-based pixel-integrated Gaussian PSF with
   true Levenberg-Marquardt optimizer: LSQ (with squared parameterization),
   WLSQ (Poisson weight), MLE (Poisson negative log-likelihood),
   elliptical Gaussian MLE, radial symmetry (Parthasarathy 2012),
   phasor (Martens 2018), centroid, or multi-emitter.
5. **Localization precision** -- Thompson-Mortensen-Quan formula with
   Mortensen 16/9 correction for LSQ/WLSQ and EMCCD excess noise factor.
6. **Post-processing** -- drift correction (cross-correlation), molecular
   merging, localization filtering, local density filtering, and duplicate
   removal.

Camera model: pixel_size, photons_per_adu, baseline, em_gain, is_emccd,
quantum_efficiency, with full ADU-to-photon conversion.
"""

import math
import numpy as np
from scipy import ndimage
from scipy.special import erf
from ...logger import logger

# Try optional imports for advanced features
try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False

try:
    from skimage.feature import peak_local_max
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    from skimage import measure
    _HAS_MEASURE = True
except ImportError:
    _HAS_MEASURE = False


# ============================================================================
# Constants
# ============================================================================

SQRT2 = math.sqrt(2.0)
TWO_OVER_SQRTPI = 2.0 / math.sqrt(math.pi)


# ============================================================================
# Section 1: IMAGE FILTERS
# ============================================================================

class BaseFilter:
    """Base class for all image filters."""

    def __init__(self, name="BaseFilter"):
        self.name = name

    def apply(self, image):
        """Apply filter to image and return filtered result."""
        raise NotImplementedError


class WaveletFilter(BaseFilter):
    """Wavelet filter using a trous (undecimated) B-spline algorithm.

    Implements the B3-spline wavelet transform as used in ThunderSTORM.
    The wavelet coefficient at the specified scale is the difference
    between successive B-spline smoothings.

    Parameters
    ----------
    scale : int
        Wavelet scale (1-5 typically). Default 2.
    order : int
        B-spline order (3 for cubic). Default 3.
    """

    def __init__(self, scale=2, order=3):
        super().__init__("Wavelet")
        self.scale = scale
        self.order = order

    def apply(self, image):
        """Apply wavelet filter using a trous algorithm.

        Stores the filtered result as ``self.F1`` so that threshold
        expressions like ``'std(Wave.F1)'`` can reference it.
        """
        img = image.astype(np.float64)

        # B3-spline 1D kernel [1, 4, 6, 4, 1] / 16
        if self.order == 3:
            h = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])
        elif self.order == 1:
            h = np.array([0.25, 0.5, 0.25])
        else:
            h = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])

        # Apply a trous algorithm: iterative upsampled convolution
        approximation = img.copy()

        for j in range(self.scale):
            # Upsample kernel by inserting zeros at scale j
            step = 2 ** j
            dilated_len = (len(h) - 1) * step + 1
            h_upsampled = np.zeros(dilated_len)
            for k in range(len(h)):
                h_upsampled[k * step] = h[k]

            # Separable convolution (rows then columns)
            temp = np.zeros_like(img)
            for row in range(img.shape[0]):
                temp[row] = np.convolve(approximation[row], h_upsampled,
                                        mode='same')
            approx_new = np.zeros_like(img)
            for col in range(img.shape[1]):
                approx_new[:, col] = np.convolve(temp[:, col], h_upsampled,
                                                  mode='same')

            wavelet = approximation - approx_new
            approximation = approx_new

        self.F1 = wavelet
        return wavelet


class GaussianFilter(BaseFilter):
    """Gaussian low-pass filter.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel. Default 1.6.
    """

    def __init__(self, sigma=1.6):
        super().__init__("Gaussian")
        self.sigma = sigma

    def apply(self, image):
        result = ndimage.gaussian_filter(image.astype(np.float64), self.sigma)
        self.F1 = result
        return result


class DifferenceOfGaussiansFilter(BaseFilter):
    """Difference of Gaussians (DoG) band-pass filter.

    Parameters
    ----------
    sigma1 : float
        Narrow Gaussian sigma. Default 1.0.
    sigma2 : float
        Wide Gaussian sigma. Default 1.6.
    """

    def __init__(self, sigma1=1.0, sigma2=1.6):
        super().__init__("DoG")
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply(self, image):
        img = image.astype(np.float64)
        g1 = ndimage.gaussian_filter(img, self.sigma1)
        g2 = ndimage.gaussian_filter(img, self.sigma2)
        result = g1 - g2
        self.F1 = result
        return result


class LoweredGaussianFilter(BaseFilter):
    """Lowered Gaussian band-pass filter (Gaussian minus averaging).

    Parameters
    ----------
    sigma : float
        Gaussian sigma. Default 1.6.
    size : int
        Averaging kernel size. Default 3.
    """

    def __init__(self, sigma=1.6, size=3):
        super().__init__("LoweredGaussian")
        self.sigma = sigma
        self.size = size

    def apply(self, image):
        img = image.astype(np.float64)
        gauss = ndimage.gaussian_filter(img, self.sigma)
        avg = ndimage.uniform_filter(img, self.size)
        result = gauss - avg
        self.F1 = result
        return result


class DifferenceOfAveragingFilter(BaseFilter):
    """Difference of averaging (box) filters.

    Parameters
    ----------
    size1 : int
        First box filter size. Default 3.
    size2 : int
        Second box filter size. Default 5.
    """

    def __init__(self, size1=3, size2=5):
        super().__init__("DiffAvg")
        self.size1 = size1
        self.size2 = size2

    def apply(self, image):
        img = image.astype(np.float64)
        avg1 = ndimage.uniform_filter(img, self.size1)
        avg2 = ndimage.uniform_filter(img, self.size2)
        result = avg1 - avg2
        self.F1 = result
        return result


class MedianFilter(BaseFilter):
    """Median filter for noise reduction.

    Parameters
    ----------
    size : int
        Kernel size. Default 3.
    """

    def __init__(self, size=3):
        super().__init__("Median")
        self.size = size

    def apply(self, image):
        result = ndimage.median_filter(image.astype(np.float64), self.size)
        self.F1 = result
        return result


class BoxFilter(BaseFilter):
    """Simple averaging (box) filter.

    Parameters
    ----------
    size : int
        Kernel size. Default 3.
    """

    def __init__(self, size=3):
        super().__init__("Box")
        self.size = size

    def apply(self, image):
        result = ndimage.uniform_filter(image.astype(np.float64), self.size)
        self.F1 = result
        return result


class NoFilter(BaseFilter):
    """Pass-through filter (no filtering)."""

    def __init__(self):
        super().__init__("NoFilter")

    def apply(self, image):
        result = image.astype(np.float64)
        self.F1 = result
        return result


class CompoundFilter(BaseFilter):
    """Compound (chained) filter — applies two filters sequentially.

    This matches the ThunderSTORM ImageJ plugin's compound filter concept,
    where a primary filter (typically wavelet) is combined with a secondary
    filter (e.g. Gaussian, DoG, averaging) for enhanced background
    suppression and noise rejection.

    In the original ThunderSTORM, the "Wavelet filter (B-spline)" can be
    combined with any other filter.  The compound filter applies the
    primary filter first, then the secondary filter on the result.

    The threshold expression variables (``Wave.F1``, ``F1``) reference
    the output of the full compound filter chain.

    Parameters
    ----------
    primary : BaseFilter
        First filter to apply (e.g. WaveletFilter).
    secondary : BaseFilter
        Second filter to apply on the primary output (e.g. GaussianFilter).

    Examples
    --------
    Wavelet + Gaussian compound::

        f = CompoundFilter(WaveletFilter(scale=2), GaussianFilter(sigma=1.0))
        result = f.apply(image)

    Wavelet + DoG compound::

        f = CompoundFilter(WaveletFilter(scale=2),
                           DifferenceOfGaussiansFilter(sigma1=1.0, sigma2=2.0))
        result = f.apply(image)
    """

    def __init__(self, primary, secondary):
        name = f"Compound({primary.name}+{secondary.name})"
        super().__init__(name)
        self.primary = primary
        self.secondary = secondary

    def apply(self, image):
        """Apply primary filter, then secondary filter on the result."""
        intermediate = self.primary.apply(image)
        result = self.secondary.apply(intermediate)
        # Store F1 from the final stage for threshold expression evaluation
        self.F1 = result
        return result


def create_filter(filter_type, **kwargs):
    """Factory function to create image filters.

    Parameters
    ----------
    filter_type : str
        One of: 'wavelet', 'gaussian', 'dog', 'lowered_gaussian',
        'difference_of_averaging', 'median', 'box', 'none'.
        Compound filters use '+' separator, e.g. 'wavelet+gaussian',
        'wavelet+dog'.
    **kwargs
        Filter-specific parameters.  For compound filters, prefix
        secondary filter params with ``secondary_``, e.g.
        ``secondary_sigma=1.0`` for a Gaussian secondary filter.

    Returns
    -------
    BaseFilter
        Configured filter object.
    """
    _filter_map = {
        'wavelet': WaveletFilter,
        'gaussian': GaussianFilter,
        'dog': DifferenceOfGaussiansFilter,
        'lowered_gaussian': LoweredGaussianFilter,
        'difference_of_averaging': DifferenceOfAveragingFilter,
        'median': MedianFilter,
        'box': BoxFilter,
        'none': NoFilter,
    }

    key = filter_type.lower()

    # Handle compound filters (e.g. 'wavelet+gaussian', 'wavelet+dog')
    if '+' in key:
        parts = [p.strip() for p in key.split('+', 1)]
        if len(parts) != 2:
            raise ValueError(f"Compound filter must have exactly two "
                             f"parts separated by '+': {filter_type!r}")
        primary_key, secondary_key = parts
        if primary_key not in _filter_map:
            raise ValueError(f"Unknown primary filter: {primary_key!r}")
        if secondary_key not in _filter_map:
            raise ValueError(f"Unknown secondary filter: {secondary_key!r}")

        # Split kwargs: secondary_ prefix goes to secondary filter
        primary_kwargs = {}
        secondary_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith('secondary_'):
                secondary_kwargs[k[len('secondary_'):]] = v
            else:
                primary_kwargs[k] = v

        primary = _filter_map[primary_key](**primary_kwargs)
        secondary = _filter_map[secondary_key](**secondary_kwargs)
        return CompoundFilter(primary, secondary)

    if key not in _filter_map:
        raise ValueError(
            f"Unknown filter type: {filter_type!r}. "
            f"Available: {list(_filter_map)}")
    return _filter_map[key](**kwargs)


# ============================================================================
# Section 2: THRESHOLD EXPRESSION EVALUATION
# ============================================================================

def compute_threshold_expression(image, filtered_image, expression,
                                 filter_obj=None):
    """Evaluate a ThunderSTORM threshold expression.

    Supports expressions like:
    - ``'std(Wave.F1)'`` -- std of wavelet level 1
    - ``'2*std(Wave.F1)'`` -- 2x std
    - ``'mean(Wave.F1) + 3*std(Wave.F1)'``
    - Numeric values (int or float)

    Parameters
    ----------
    image : ndarray
        Raw image.
    filtered_image : ndarray
        Filtered image.
    expression : str, int, or float
        Threshold expression or numeric value.
    filter_obj : BaseFilter, optional
        Filter object that may have an ``F1`` attribute.

    Returns
    -------
    float
        Computed threshold value.
    """
    if isinstance(expression, (int, float)):
        return float(expression)

    expr = str(expression)

    # Replace Wave.F1 with F1 for compatibility
    expr = expr.replace('Wave.F1', 'F1')

    # Determine what F1 refers to
    if filter_obj is not None and hasattr(filter_obj, 'F1'):
        f1 = filter_obj.F1
    else:
        f1 = filtered_image

    namespace = {
        'std': np.std,
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'I1': image,
        'F1': f1,
    }

    try:
        threshold = eval(expr, {"__builtins__": {}}, namespace)
        return float(threshold)
    except Exception as e:
        logger.warning("Invalid threshold expression '%s': %s. "
                       "Falling back to std-based threshold.", expression, e)
        # Fallback: use std of filtered image
        return float(np.std(filtered_image))


# ============================================================================
# Section 3: CANDIDATE DETECTORS
# ============================================================================

def _remove_border_detections(detections, image_shape, border=5):
    """Remove detections too close to image borders.

    Parameters
    ----------
    detections : ndarray
        (N, 2) array of (row, col) positions.
    image_shape : tuple
        (height, width).
    border : int
        Border exclusion width in pixels.

    Returns
    -------
    ndarray
        Filtered (M, 2) array.
    """
    if len(detections) == 0:
        return detections
    height, width = image_shape
    mask = ((detections[:, 0] >= border) &
            (detections[:, 0] < height - border) &
            (detections[:, 1] >= border) &
            (detections[:, 1] < width - border))
    return detections[mask]


class BaseDetector:
    """Base class for molecule detectors."""

    def __init__(self, name="BaseDetector"):
        self.name = name

    def detect(self, image, threshold):
        """Detect candidates. Returns (N, 2) array of (row, col)."""
        raise NotImplementedError


class LocalMaximumDetector(BaseDetector):
    """Detect molecules as local maxima.

    Uses ``peak_local_max`` from scikit-image when available, otherwise
    falls back to ``scipy.ndimage.maximum_filter``.

    Parameters
    ----------
    connectivity : str
        '4-neighbourhood' or '8-neighbourhood'. Default '8-neighbourhood'.
    min_distance : int
        Minimum distance between peaks. Default 1.
    exclude_border : int or bool
        Border exclusion width. Default True (auto-computed).
    """

    def __init__(self, connectivity='8-neighbourhood', min_distance=1,
                 exclude_border=True):
        super().__init__("LocalMaximum")
        self.connectivity = connectivity
        self.min_distance = min_distance
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Detect local maxima above threshold."""
        if self.exclude_border is True:
            border_pixels = max(3, self.min_distance)
        elif isinstance(self.exclude_border, int):
            border_pixels = self.exclude_border
        else:
            border_pixels = 0

        if _HAS_SKIMAGE:
            coordinates = peak_local_max(
                image,
                min_distance=self.min_distance,
                threshold_abs=threshold,
                exclude_border=border_pixels if border_pixels else False
            )
        else:
            # Fallback: scipy maximum_filter
            win_size = 2 * self.min_distance + 1
            local_max = (image == ndimage.maximum_filter(image,
                                                         size=win_size))
            above = image > threshold
            mask = local_max & above
            if border_pixels:
                b = border_pixels
                mask[:b, :] = False
                mask[-b:, :] = False
                mask[:, :b] = False
                mask[:, -b:] = False
            ys, xs = np.where(mask)
            if len(ys) > 0:
                coordinates = np.column_stack([ys, xs])
            else:
                coordinates = np.empty((0, 2), dtype=int)

        # Additional border filtering
        if border_pixels and len(coordinates) > 0:
            coordinates = _remove_border_detections(
                coordinates, image.shape, border=border_pixels)

        return coordinates


class NonMaximumSuppressionDetector(BaseDetector):
    """Non-maximum suppression detector.

    Finds local maxima using morphological grey dilation.

    Parameters
    ----------
    connectivity : int
        Connectivity for morphological operations (1 or 2). Default 2.
    exclude_border : int or bool
        Border exclusion width. Default True.
    """

    def __init__(self, connectivity=2, exclude_border=True):
        super().__init__("NonMaxSuppression")
        self.connectivity = connectivity
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Detect using non-maximum suppression."""
        binary = image > threshold
        struct = ndimage.generate_binary_structure(2, self.connectivity)
        dilated = ndimage.grey_dilation(image, footprint=struct)
        maxima = (image == dilated) & binary
        coordinates = np.column_stack(np.where(maxima))

        if self.exclude_border and len(coordinates) > 0:
            border_width = 3 if self.exclude_border is True else self.exclude_border
            coordinates = _remove_border_detections(
                coordinates, image.shape, border=border_width)

        return coordinates


class CentroidDetector(BaseDetector):
    """Centroid of connected components detector.

    Segments the thresholded image into connected components and returns
    the intensity-weighted centroid of each region.

    Parameters
    ----------
    connectivity : int
        Connectivity for connected components (1 or 2). Default 2.
    min_area : int
        Minimum component area. Default 1.
    use_watershed : bool
        Use watershed segmentation to split merged peaks. Default True.
    exclude_border : int or bool
        Border exclusion width. Default True.
    """

    def __init__(self, connectivity=2, min_area=1, use_watershed=True,
                 exclude_border=True):
        super().__init__("Centroid")
        self.connectivity = connectivity
        self.min_area = min_area
        self.use_watershed = use_watershed
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Detect centroids of connected components."""
        binary = image > threshold

        if self.use_watershed:
            # Watershed segmentation to split touching objects
            dist = ndimage.distance_transform_edt(binary)
            # Find local maxima of distance transform as markers
            from scipy.ndimage import maximum_filter
            local_max = (dist == maximum_filter(dist, size=3)) & binary
            markers, n_markers = ndimage.label(local_max)
            if n_markers > 1:
                try:
                    from skimage.segmentation import watershed as sk_watershed
                    labeled = sk_watershed(-dist, markers, mask=binary)
                except ImportError:
                    # Fallback: use scipy watershed_ift
                    try:
                        labeled = ndimage.watershed_ift(
                            (-dist * 1000).astype(np.int16), markers)
                        labeled[~binary] = 0
                    except Exception:
                        if _HAS_MEASURE:
                            labeled = measure.label(
                                binary, connectivity=self.connectivity)
                        else:
                            labeled, _ = ndimage.label(binary)
            else:
                if _HAS_MEASURE:
                    labeled = measure.label(
                        binary, connectivity=self.connectivity)
                else:
                    labeled, _ = ndimage.label(binary)
        elif _HAS_MEASURE:
            labeled = measure.label(binary, connectivity=self.connectivity)
        else:
            labeled, _ = ndimage.label(binary)

        if _HAS_MEASURE:
            regions = measure.regionprops(labeled, intensity_image=image)

            coordinates = []
            for region in regions:
                if region.area >= self.min_area:
                    coordinates.append([region.weighted_centroid[0],
                                        region.weighted_centroid[1]])
        else:
            n_labels = labeled.max()
            coordinates = []
            for label_id in range(1, n_labels + 1):
                mask = labeled == label_id
                area = np.sum(mask)
                if area < self.min_area:
                    continue
                ys, xs = np.where(mask)
                weights = image[ys, xs]
                total = np.sum(weights)
                if total > 0:
                    cy = np.sum(ys * weights) / total
                    cx = np.sum(xs * weights) / total
                    coordinates.append([cy, cx])

        if not coordinates:
            return np.empty((0, 2), dtype=np.float64)

        coordinates = np.array(coordinates)

        if self.exclude_border and len(coordinates) > 0:
            border_width = (3 if self.exclude_border is True
                            else self.exclude_border)
            coordinates = _remove_border_detections(
                coordinates, image.shape, border=border_width)

        return coordinates


class GridDetector(BaseDetector):
    """Detect molecules on a regular grid (for simulations).

    Parameters
    ----------
    spacing : int
        Grid spacing in pixels. Default 10.
    exclude_border : int or bool
        Border exclusion width. Default True.
    """

    def __init__(self, spacing=10, exclude_border=True):
        super().__init__("Grid")
        self.spacing = spacing
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        rows, cols = image.shape
        if self.exclude_border is True:
            border = self.spacing // 2
        elif isinstance(self.exclude_border, int):
            border = self.exclude_border
        else:
            border = 0

        row_coords = np.arange(border, rows - border, self.spacing)
        col_coords = np.arange(border, cols - border, self.spacing)
        grid_rows, grid_cols = np.meshgrid(row_coords, col_coords,
                                            indexing='ij')
        coordinates = np.column_stack([grid_rows.ravel(), grid_cols.ravel()])

        if len(coordinates) > 0:
            ri = coordinates[:, 0].astype(int)
            ci = coordinates[:, 1].astype(int)
            valid = ((ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols))
            coordinates = coordinates[valid]
            ri = ri[valid]
            ci = ci[valid]
            if len(coordinates) > 0:
                intensities = image[ri, ci]
                coordinates = coordinates[intensities > threshold]

        return coordinates


def create_detector(detector_type, **kwargs):
    """Factory function to create detectors.

    Parameters
    ----------
    detector_type : str
        One of: 'local_max', 'local_maximum', 'nms',
        'non_maximum_suppression', 'centroid', 'grid'.
    **kwargs
        Detector-specific parameters.

    Returns
    -------
    BaseDetector
        Configured detector object.
    """
    _detector_map = {
        'local_max': LocalMaximumDetector,
        'local_maximum': LocalMaximumDetector,
        'nms': NonMaximumSuppressionDetector,
        'non_maximum_suppression': NonMaximumSuppressionDetector,
        'centroid': CentroidDetector,
        'grid': GridDetector,
    }
    key = detector_type.lower()
    if key not in _detector_map:
        raise ValueError(
            f"Unknown detector type: {detector_type!r}. "
            f"Available: {list(_detector_map)}")
    return _detector_map[key](**kwargs)


# ============================================================================
# Section 4: INTEGRATED GAUSSIAN PSF MODEL (erf-based)
# ============================================================================
# Matches ImageJ ThunderSTORM: pixel value = integral of Gaussian over pixel.
# Parameters: [x0, y0, sigma, intensity, offset]

def _integrated_gaussian_value(jj, ii, x0, y0, sigma, intensity, offset):
    """Compute integrated Gaussian PSF value for pixel (jj, ii).

    This is the erf-based pixel-integrated Gaussian model matching
    ImageJ ThunderSTORM exactly::

        dx = 0.5 * (erf((x + 0.5 - x0) / (sqrt2 * sigma))
                   - erf((x - 0.5 - x0) / (sqrt2 * sigma)))
        dy = 0.5 * (erf((y + 0.5 - y0) / (sqrt2 * sigma))
                   - erf((y - 0.5 - y0) / (sqrt2 * sigma)))
        model = intensity * dx * dy + offset

    Parameters
    ----------
    jj : float
        Pixel column coordinate (x).
    ii : float
        Pixel row coordinate (y).
    x0, y0 : float
        Sub-pixel PSF center.
    sigma : float
        PSF standard deviation.
    intensity : float
        Total molecule intensity (photons under the PSF).
    offset : float
        Background per pixel.

    Returns
    -------
    float
        Expected pixel value.
    """
    s2 = SQRT2 * sigma
    ex = 0.5 * (math.erf((jj - x0 + 0.5) / s2) -
                math.erf((jj - x0 - 0.5) / s2))
    ey = 0.5 * (math.erf((ii - y0 + 0.5) / s2) -
                math.erf((ii - y0 - 0.5) / s2))
    return offset + intensity * ex * ey


def _integrated_gaussian_jacobian(jj, ii, x0, y0, sigma, intensity, offset):
    """Jacobian of integrated Gaussian for pixel (jj, ii).

    Returns partial derivatives w.r.t. [x0, y0, sigma, intensity, offset].

    Returns
    -------
    ndarray of shape (5,)
    """
    jac = np.zeros(5)
    s2 = SQRT2 * sigma

    erf_xp = math.erf((jj - x0 + 0.5) / s2)
    erf_xm = math.erf((jj - x0 - 0.5) / s2)
    erf_yp = math.erf((ii - y0 + 0.5) / s2)
    erf_ym = math.erf((ii - y0 - 0.5) / s2)

    ex = 0.5 * (erf_xp - erf_xm)
    ey = 0.5 * (erf_yp - erf_ym)

    inv_s2 = 1.0 / s2

    ux_p = (jj - x0 + 0.5) * inv_s2
    ux_m = (jj - x0 - 0.5) * inv_s2
    uy_p = (ii - y0 + 0.5) * inv_s2
    uy_m = (ii - y0 - 0.5) * inv_s2

    gx_p = TWO_OVER_SQRTPI * math.exp(-ux_p * ux_p)
    gx_m = TWO_OVER_SQRTPI * math.exp(-ux_m * ux_m)
    gy_p = TWO_OVER_SQRTPI * math.exp(-uy_p * uy_p)
    gy_m = TWO_OVER_SQRTPI * math.exp(-uy_m * uy_m)

    dex_dx0 = 0.5 * inv_s2 * (gx_m - gx_p)
    dey_dy0 = 0.5 * inv_s2 * (gy_m - gy_p)
    dex_dsigma = 0.5 * (-ux_p * gx_p + ux_m * gx_m) / sigma
    dey_dsigma = 0.5 * (-uy_p * gy_p + uy_m * gy_m) / sigma

    jac[0] = intensity * dex_dx0 * ey           # d/dx0
    jac[1] = intensity * ex * dey_dy0           # d/dy0
    jac[2] = intensity * (dex_dsigma * ey +
                          ex * dey_dsigma)      # d/dsigma
    jac[3] = ex * ey                            # d/dintensity
    jac[4] = 1.0                                # d/doffset
    return jac


# ============================================================================
# Section 4b: ELLIPTICAL INTEGRATED GAUSSIAN PSF MODEL
# ============================================================================

def _integrated_elliptical_gaussian_value(jj, ii, x0, y0, sigma1, sigma2,
                                           intensity, offset):
    """Elliptical integrated Gaussian PSF (independent sigma_x, sigma_y)."""
    s2x = SQRT2 * sigma1
    s2y = SQRT2 * sigma2
    ex = 0.5 * (math.erf((jj - x0 + 0.5) / s2x) -
                math.erf((jj - x0 - 0.5) / s2x))
    ey = 0.5 * (math.erf((ii - y0 + 0.5) / s2y) -
                math.erf((ii - y0 - 0.5) / s2y))
    return offset + intensity * ex * ey


def _integrated_elliptical_gaussian_jacobian(jj, ii, x0, y0, sigma1, sigma2,
                                              intensity, offset):
    """Jacobian w.r.t. [x0, y0, sigma1, sigma2, intensity, offset]."""
    jac = np.zeros(6)
    s2x = SQRT2 * sigma1
    s2y = SQRT2 * sigma2

    erf_xp = math.erf((jj - x0 + 0.5) / s2x)
    erf_xm = math.erf((jj - x0 - 0.5) / s2x)
    erf_yp = math.erf((ii - y0 + 0.5) / s2y)
    erf_ym = math.erf((ii - y0 - 0.5) / s2y)

    ex = 0.5 * (erf_xp - erf_xm)
    ey = 0.5 * (erf_yp - erf_ym)

    inv_s2x = 1.0 / s2x
    inv_s2y = 1.0 / s2y

    ux_p = (jj - x0 + 0.5) * inv_s2x
    ux_m = (jj - x0 - 0.5) * inv_s2x
    uy_p = (ii - y0 + 0.5) * inv_s2y
    uy_m = (ii - y0 - 0.5) * inv_s2y

    gx_p = TWO_OVER_SQRTPI * math.exp(-ux_p * ux_p)
    gx_m = TWO_OVER_SQRTPI * math.exp(-ux_m * ux_m)
    gy_p = TWO_OVER_SQRTPI * math.exp(-uy_p * uy_p)
    gy_m = TWO_OVER_SQRTPI * math.exp(-uy_m * uy_m)

    dex_dx0 = 0.5 * inv_s2x * (gx_m - gx_p)
    dey_dy0 = 0.5 * inv_s2y * (gy_m - gy_p)
    dex_dsigma1 = 0.5 * (-ux_p * gx_p + ux_m * gx_m) / sigma1
    dey_dsigma2 = 0.5 * (-uy_p * gy_p + uy_m * gy_m) / sigma2

    jac[0] = intensity * dex_dx0 * ey
    jac[1] = intensity * ex * dey_dy0
    jac[2] = intensity * dex_dsigma1 * ey
    jac[3] = intensity * ex * dey_dsigma2
    jac[4] = ex * ey
    jac[5] = 1.0
    return jac


# ============================================================================
# Section 5: LEVENBERG-MARQUARDT OPTIMIZER (Pure NumPy)
# ============================================================================
# Replaces the Numba-JIT version from the original plugin.
# Uses np.linalg.solve instead of custom solve_NxN.
# Squared parameterization for positivity of sigma, intensity, offset.

def _fit_integrated_gaussian_lsq(roi, row_offset, col_offset,
                                  initial_sigma=1.3, max_iterations=500):
    """Fit integrated Gaussian PSF using LSQ with true LM optimizer.

    5 parameters: [x0, y0, sigma, intensity, offset].
    Squared parameterization: optimizer works on
    p = [x0, y0, sqrt(sigma), sqrt(intensity), sqrt(offset)].

    Parameters
    ----------
    roi : ndarray
        2D ROI array.
    row_offset, col_offset : int
        Offset of ROI origin in full image coordinates.
    initial_sigma : float
        Initial PSF sigma estimate.
    max_iterations : int
        Maximum LM iterations.

    Returns
    -------
    dict or None
        Fit result with keys: x, y, intensity, sigma_x, sigma_y,
        background, chi_squared, or None if fit fails.
    """
    ny, nx = roi.shape
    n_pixels = ny * nx
    if n_pixels < 5:
        return None

    roi_f = roi.astype(np.float64)

    # Initial parameter estimates
    offset = max(float(np.min(roi_f)), 0.01)
    peak = float(np.max(roi_f)) - offset
    total = roi_f.sum()
    if total > 0:
        yy, xx = np.mgrid[0:ny, 0:nx]
        x0 = float(np.sum(xx * roi_f) / total)
        y0 = float(np.sum(yy * roi_f) / total)
    else:
        x0 = float(nx) / 2.0
        y0 = float(ny) / 2.0
    sigma = initial_sigma
    intensity = max(peak * 2.0 * math.pi * sigma * sigma, 1.0)

    # Squared parameterization
    p2 = math.sqrt(sigma)
    p3 = math.sqrt(intensity)
    p4 = math.sqrt(offset)

    lambda_lm = 0.001

    # Precompute pixel coordinate arrays
    jj_arr = np.arange(nx, dtype=np.float64)
    ii_arr = np.arange(ny, dtype=np.float64)

    # Compute initial chi-squared
    chi_sq = 0.0
    for ii in range(ny):
        for jj in range(nx):
            model_val = _integrated_gaussian_value(
                jj_arr[jj], ii_arr[ii], x0, y0, sigma, intensity, offset)
            r = roi_f[ii, jj] - model_val
            chi_sq += r * r

    for iteration in range(max_iterations):
        # Build Hessian approximation H = J^T J and gradient g = J^T r
        H = np.zeros((5, 5))
        g = np.zeros(5)

        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                model_val = _integrated_gaussian_value(
                    fj, fi, x0, y0, sigma, intensity, offset)
                residual = roi_f[ii, jj] - model_val
                J_orig = _integrated_gaussian_jacobian(
                    fj, fi, x0, y0, sigma, intensity, offset)

                # Transform Jacobian for squared parameterization
                J = np.empty(5)
                J[0] = J_orig[0]
                J[1] = J_orig[1]
                J[2] = J_orig[2] * 2.0 * p2
                J[3] = J_orig[3] * 2.0 * p3
                J[4] = J_orig[4] * 2.0 * p4

                g += J * residual
                H += np.outer(J, J)

        # Augment diagonal: H_aug = H + lambda * diag(H)
        H_aug = H.copy()
        diag_h = np.diag(H_aug).copy()
        diag_h[diag_h < 1e-30] = 1e-30
        H_aug[np.diag_indices(5)] = diag_h * (1.0 + lambda_lm)

        # Solve for parameter update
        try:
            dp = np.linalg.solve(H_aug, g)
        except np.linalg.LinAlgError:
            break

        # Candidate parameters
        x0_new = x0 + dp[0]
        y0_new = y0 + dp[1]
        p2_new = p2 + dp[2]
        p3_new = p3 + dp[3]
        p4_new = p4 + dp[4]

        # Back-transform (always non-negative)
        sigma_new = p2_new * p2_new
        intensity_new = p3_new * p3_new
        offset_new = p4_new * p4_new

        # Sanity bounds
        sigma_new = min(max(sigma_new, 0.1), 10.0)
        if sigma_new == 0.1:
            p2_new = math.sqrt(0.1)

        # Compute new chi-squared
        chi_sq_new = 0.0
        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_gaussian_value(
                    fj, fi, x0_new, y0_new, sigma_new,
                    intensity_new, offset_new)
                r = roi_f[ii, jj] - mv
                chi_sq_new += r * r

        if chi_sq_new < chi_sq:
            rel_change = abs(chi_sq - chi_sq_new) / max(chi_sq, 1e-30)
            x0, y0 = x0_new, y0_new
            p2, p3, p4 = p2_new, p3_new, p4_new
            sigma = sigma_new
            intensity = intensity_new
            offset = offset_new
            chi_sq = chi_sq_new
            lambda_lm = max(lambda_lm / 10.0, 1e-7)

            if rel_change < 1e-8:
                break
        else:
            lambda_lm *= 10.0
            if lambda_lm > 1e10:
                break

    return {
        'x': x0 + col_offset,
        'y': y0 + row_offset,
        'intensity': intensity,
        'sigma_x': sigma,
        'sigma_y': sigma,
        'background': offset,
        'chi_squared': chi_sq / n_pixels,
    }


def _fit_integrated_gaussian_wlsq(roi, row_offset, col_offset,
                                    initial_sigma=1.3, max_iterations=500):
    """Fit integrated Gaussian PSF using WLSQ with true LM optimizer.

    Weights: w = 1/max(data, 1) (Poisson variance model).
    Squared parameterization for positivity.
    """
    ny, nx = roi.shape
    n_pixels = ny * nx
    if n_pixels < 5:
        return None

    roi_f = roi.astype(np.float64)

    offset = max(float(np.min(roi_f)), 0.01)
    peak = float(np.max(roi_f)) - offset
    total = roi_f.sum()
    if total > 0:
        yy, xx = np.mgrid[0:ny, 0:nx]
        x0 = float(np.sum(xx * roi_f) / total)
        y0 = float(np.sum(yy * roi_f) / total)
    else:
        x0 = float(nx) / 2.0
        y0 = float(ny) / 2.0
    sigma = initial_sigma
    intensity = max(peak * 2.0 * math.pi * sigma * sigma, 1.0)

    p2 = math.sqrt(sigma)
    p3 = math.sqrt(intensity)
    p4 = math.sqrt(offset)
    lambda_lm = 0.001

    jj_arr = np.arange(nx, dtype=np.float64)
    ii_arr = np.arange(ny, dtype=np.float64)

    # Initial weighted chi-squared
    chi_sq = 0.0
    for ii in range(ny):
        for jj in range(nx):
            mv = _integrated_gaussian_value(
                jj_arr[jj], ii_arr[ii], x0, y0, sigma, intensity, offset)
            data = roi_f[ii, jj]
            r = data - mv
            w = 1.0 / max(data, 1.0)
            chi_sq += w * r * r

    for iteration in range(max_iterations):
        H = np.zeros((5, 5))
        g = np.zeros(5)

        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_gaussian_value(
                    fj, fi, x0, y0, sigma, intensity, offset)
                data = roi_f[ii, jj]
                residual = data - mv
                w = 1.0 / max(data, 1.0)
                J_orig = _integrated_gaussian_jacobian(
                    fj, fi, x0, y0, sigma, intensity, offset)

                J = np.empty(5)
                J[0] = J_orig[0]
                J[1] = J_orig[1]
                J[2] = J_orig[2] * 2.0 * p2
                J[3] = J_orig[3] * 2.0 * p3
                J[4] = J_orig[4] * 2.0 * p4

                g += w * J * residual
                H += w * np.outer(J, J)

        H_aug = H.copy()
        diag_h = np.diag(H_aug).copy()
        diag_h[diag_h < 1e-30] = 1e-30
        H_aug[np.diag_indices(5)] = diag_h * (1.0 + lambda_lm)

        try:
            dp = np.linalg.solve(H_aug, g)
        except np.linalg.LinAlgError:
            break

        x0_new = x0 + dp[0]
        y0_new = y0 + dp[1]
        p2_new = p2 + dp[2]
        p3_new = p3 + dp[3]
        p4_new = p4 + dp[4]

        sigma_new = min(max(p2_new * p2_new, 0.1), 10.0)
        intensity_new = p3_new * p3_new
        offset_new = p4_new * p4_new
        if sigma_new == 0.1:
            p2_new = math.sqrt(0.1)

        chi_sq_new = 0.0
        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_gaussian_value(
                    fj, fi, x0_new, y0_new, sigma_new,
                    intensity_new, offset_new)
                data = roi_f[ii, jj]
                r = data - mv
                w = 1.0 / max(data, 1.0)
                chi_sq_new += w * r * r

        if chi_sq_new < chi_sq:
            rel_change = abs(chi_sq - chi_sq_new) / max(chi_sq, 1e-30)
            x0, y0 = x0_new, y0_new
            p2, p3, p4 = p2_new, p3_new, p4_new
            sigma = sigma_new
            intensity = intensity_new
            offset = offset_new
            chi_sq = chi_sq_new
            lambda_lm = max(lambda_lm / 10.0, 1e-7)
            if rel_change < 1e-8:
                break
        else:
            lambda_lm *= 10.0
            if lambda_lm > 1e10:
                break

    return {
        'x': x0 + col_offset,
        'y': y0 + row_offset,
        'intensity': intensity,
        'sigma_x': sigma,
        'sigma_y': sigma,
        'background': offset,
        'chi_squared': chi_sq / n_pixels,
    }


def _fit_integrated_gaussian_mle(roi, row_offset, col_offset,
                                  initial_sigma=1.3, max_iterations=1000):
    """Fit integrated Gaussian PSF using MLE with true LM optimizer.

    Poisson noise model: -log L = sum(model - data * log(model)).
    Squared parameterization for positivity.
    Higher max iterations and tighter convergence for MLE.
    """
    ny, nx = roi.shape
    n_pixels = ny * nx
    if n_pixels < 5:
        return None

    roi_f = roi.astype(np.float64)

    offset = max(float(np.min(roi_f)), 0.1)
    peak = float(np.max(roi_f)) - offset
    total = roi_f.sum()
    if total > 0:
        yy, xx = np.mgrid[0:ny, 0:nx]
        x0 = float(np.sum(xx * roi_f) / total)
        y0 = float(np.sum(yy * roi_f) / total)
    else:
        x0 = float(nx) / 2.0
        y0 = float(ny) / 2.0
    sigma = initial_sigma
    intensity = max(peak * 2.0 * math.pi * sigma * sigma, 1.0)

    p2 = math.sqrt(sigma)
    p3 = math.sqrt(intensity)
    p4 = math.sqrt(offset)
    lambda_lm = 0.001

    jj_arr = np.arange(nx, dtype=np.float64)
    ii_arr = np.arange(ny, dtype=np.float64)

    # Initial negative log-likelihood
    neg_ll = 0.0
    for ii in range(ny):
        for jj in range(nx):
            mv = _integrated_gaussian_value(
                jj_arr[jj], ii_arr[ii], x0, y0, sigma, intensity, offset)
            mv = max(mv, 1e-10)
            data = roi_f[ii, jj]
            neg_ll += mv - data * math.log(mv)

    for iteration in range(max_iterations):
        H = np.zeros((5, 5))
        g = np.zeros(5)

        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_gaussian_value(
                    fj, fi, x0, y0, sigma, intensity, offset)
                mv = max(mv, 1e-10)
                data = roi_f[ii, jj]
                J_orig = _integrated_gaussian_jacobian(
                    fj, fi, x0, y0, sigma, intensity, offset)

                J = np.empty(5)
                J[0] = J_orig[0]
                J[1] = J_orig[1]
                J[2] = J_orig[2] * 2.0 * p2
                J[3] = J_orig[3] * 2.0 * p3
                J[4] = J_orig[4] * 2.0 * p4

                # Fisher information weight = 1/model
                w = 1.0 / mv
                factor = data / mv - 1.0

                g += factor * J
                H += w * np.outer(J, J)

        H_aug = H.copy()
        diag_h = np.diag(H_aug).copy()
        diag_h[diag_h < 1e-30] = 1e-30
        H_aug[np.diag_indices(5)] = diag_h * (1.0 + lambda_lm)

        try:
            dp = np.linalg.solve(H_aug, g)
        except np.linalg.LinAlgError:
            break

        x0_new = x0 + dp[0]
        y0_new = y0 + dp[1]
        p2_new = p2 + dp[2]
        p3_new = p3 + dp[3]
        p4_new = p4 + dp[4]

        sigma_new = min(max(p2_new * p2_new, 0.1), 10.0)
        intensity_new = p3_new * p3_new
        offset_new = max(p4_new * p4_new, 1e-10)
        if sigma_new == 0.1:
            p2_new = math.sqrt(0.1)

        neg_ll_new = 0.0
        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_gaussian_value(
                    fj, fi, x0_new, y0_new, sigma_new,
                    intensity_new, offset_new)
                mv = max(mv, 1e-10)
                data = roi_f[ii, jj]
                neg_ll_new += mv - data * math.log(mv)

        if neg_ll_new < neg_ll:
            rel_change = abs(neg_ll - neg_ll_new) / max(abs(neg_ll), 1e-30)
            x0, y0 = x0_new, y0_new
            p2, p3, p4 = p2_new, p3_new, p4_new
            sigma = sigma_new
            intensity = intensity_new
            offset = offset_new
            neg_ll = neg_ll_new
            lambda_lm = max(lambda_lm / 10.0, 1e-7)
            if rel_change < 1e-8:
                break
        else:
            lambda_lm *= 10.0
            if lambda_lm > 1e10:
                break

    return {
        'x': x0 + col_offset,
        'y': y0 + row_offset,
        'intensity': intensity,
        'sigma_x': sigma,
        'sigma_y': sigma,
        'background': offset,
        'chi_squared': neg_ll / n_pixels,
    }


def _fit_elliptical_gaussian_mle(roi, row_offset, col_offset,
                                   initial_sigma=1.3, max_iterations=1000):
    """Fit elliptical integrated Gaussian PSF using MLE with LM.

    6 parameters: [x0, y0, sigma1, sigma2, intensity, offset].
    Squared parameterization for sigma1, sigma2, intensity, offset.
    """
    ny, nx = roi.shape
    n_pixels = ny * nx
    if n_pixels < 6:
        return None

    roi_f = roi.astype(np.float64)

    offset = max(float(np.min(roi_f)), 0.1)
    peak = float(np.max(roi_f)) - offset
    total = roi_f.sum()
    if total > 0:
        yy, xx = np.mgrid[0:ny, 0:nx]
        x0 = float(np.sum(xx * roi_f) / total)
        y0 = float(np.sum(yy * roi_f) / total)
    else:
        x0 = float(nx) / 2.0
        y0 = float(ny) / 2.0
    sigma1 = initial_sigma
    sigma2 = initial_sigma
    intensity = max(peak * 2.0 * math.pi * sigma1 * sigma2, 1.0)

    p_s1 = math.sqrt(sigma1)
    p_s2 = math.sqrt(sigma2)
    p_I = math.sqrt(intensity)
    p_bg = math.sqrt(offset)
    lambda_lm = 0.001

    jj_arr = np.arange(nx, dtype=np.float64)
    ii_arr = np.arange(ny, dtype=np.float64)

    neg_ll = 0.0
    for ii in range(ny):
        for jj in range(nx):
            mv = _integrated_elliptical_gaussian_value(
                jj_arr[jj], ii_arr[ii], x0, y0, sigma1, sigma2,
                intensity, offset)
            mv = max(mv, 1e-10)
            neg_ll += mv - roi_f[ii, jj] * math.log(mv)

    for iteration in range(max_iterations):
        H = np.zeros((6, 6))
        g = np.zeros(6)

        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_elliptical_gaussian_value(
                    fj, fi, x0, y0, sigma1, sigma2, intensity, offset)
                mv = max(mv, 1e-10)
                data = roi_f[ii, jj]
                J_orig = _integrated_elliptical_gaussian_jacobian(
                    fj, fi, x0, y0, sigma1, sigma2, intensity, offset)

                J = np.empty(6)
                J[0] = J_orig[0]
                J[1] = J_orig[1]
                J[2] = J_orig[2] * 2.0 * p_s1
                J[3] = J_orig[3] * 2.0 * p_s2
                J[4] = J_orig[4] * 2.0 * p_I
                J[5] = J_orig[5] * 2.0 * p_bg

                w = 1.0 / mv
                factor = data / mv - 1.0

                g += factor * J
                H += w * np.outer(J, J)

        H_aug = H.copy()
        diag_h = np.diag(H_aug).copy()
        diag_h[diag_h < 1e-30] = 1e-30
        H_aug[np.diag_indices(6)] = diag_h * (1.0 + lambda_lm)

        try:
            dp = np.linalg.solve(H_aug, g)
        except np.linalg.LinAlgError:
            break

        x0_n = x0 + dp[0]
        y0_n = y0 + dp[1]
        ps1_n = p_s1 + dp[2]
        ps2_n = p_s2 + dp[3]
        pI_n = p_I + dp[4]
        pbg_n = p_bg + dp[5]

        s1_n = min(max(ps1_n * ps1_n, 0.1), 10.0)
        s2_n = min(max(ps2_n * ps2_n, 0.1), 10.0)
        I_n = pI_n * pI_n
        bg_n = max(pbg_n * pbg_n, 1e-10)

        if s1_n == 0.1:
            ps1_n = math.sqrt(0.1)
        if s2_n == 0.1:
            ps2_n = math.sqrt(0.1)

        neg_ll_n = 0.0
        for ii in range(ny):
            fi = ii_arr[ii]
            for jj in range(nx):
                fj = jj_arr[jj]
                mv = _integrated_elliptical_gaussian_value(
                    fj, fi, x0_n, y0_n, s1_n, s2_n, I_n, bg_n)
                mv = max(mv, 1e-10)
                neg_ll_n += mv - roi_f[ii, jj] * math.log(mv)

        if neg_ll_n < neg_ll:
            rel = abs(neg_ll - neg_ll_n) / max(abs(neg_ll), 1e-30)
            x0, y0 = x0_n, y0_n
            p_s1, p_s2, p_I, p_bg = ps1_n, ps2_n, pI_n, pbg_n
            sigma1, sigma2, intensity, offset = s1_n, s2_n, I_n, bg_n
            neg_ll = neg_ll_n
            lambda_lm = max(lambda_lm / 10.0, 1e-7)
            if rel < 1e-8:
                break
        else:
            lambda_lm *= 10.0
            if lambda_lm > 1e10:
                break

    return {
        'x': x0 + col_offset,
        'y': y0 + row_offset,
        'intensity': intensity,
        'sigma_x': sigma1,
        'sigma_y': sigma2,
        'background': offset,
        'chi_squared': neg_ll / n_pixels,
    }


def _fit_radial_symmetry(roi, row_offset, col_offset):
    """Parthasarathy (2012) radial symmetry localization.

    Computes image gradients along 45-degree rotated coordinates, fits
    gradient lines via weighted least squares to find the point of maximal
    radial symmetry. Model-free, non-iterative, very fast.
    """
    ny, nx = roi.shape
    if ny < 3 or nx < 3:
        return None

    roi_f = roi.astype(np.float64)

    # Diagonal gradients (45-degree rotated coordinates)
    dIdu = roi_f[1:, 1:] - roi_f[:-1, :-1]
    dIdv = roi_f[1:, :-1] - roi_f[:-1, 1:]

    # Smooth gradients
    if dIdu.shape[0] >= 3 and dIdu.shape[1] >= 3:
        dIdu = ndimage.uniform_filter(dIdu, size=3)
        dIdv = ndimage.uniform_filter(dIdv, size=3)

    # Half-pixel grid positions
    rows_g, cols_g = np.mgrid[0:dIdu.shape[0], 0:dIdu.shape[1]]
    rows_g = rows_g.astype(np.float64) + 0.5
    cols_g = cols_g.astype(np.float64) + 0.5

    denom = dIdu - dIdv
    valid = np.abs(denom) > 1e-6
    if np.sum(valid) < 3:
        return None

    m = np.zeros_like(dIdu)
    m[valid] = -(dIdu[valid] + dIdv[valid]) / denom[valid]
    b = rows_g - m * cols_g

    # Weights: gradient magnitude squared
    grad_mag2 = dIdu ** 2 + dIdv ** 2
    w = grad_mag2[valid]
    total_w = w.sum()
    if total_w < 1e-30:
        return None
    w = w / total_w

    mv = m[valid]
    bv = b[valid]

    sw = np.sum(w)
    smw = np.sum(mv * w)
    smmw = np.sum(mv * mv * w)
    sbw = np.sum(bv * w)
    smbw = np.sum(mv * bv * w)

    det = smmw * sw - smw * smw
    if abs(det) < 1e-30:
        return None

    xc = (smbw * sw - smw * sbw) / det
    yc = (smbw * smw - smmw * sbw) / det

    bg = float(np.min(roi_f))
    intensity = float(np.sum(roi_f) - bg * ny * nx)

    return {
        'x': xc + col_offset,
        'y': yc + row_offset,
        'intensity': max(intensity, 0.0),
        'sigma_x': 1.5,
        'sigma_y': 1.5,
        'background': bg,
        'chi_squared': 0.0,
    }


def _fit_phasor(roi, row_offset, col_offset):
    """Phasor-based localization (Martens et al. 2018).

    Converts ROI to phase vectors via first Fourier coefficients.
    Model-free, non-iterative, extremely fast.
    """
    ny, nx = roi.shape
    if ny < 2 or nx < 2:
        return None

    roi_f = roi.astype(np.float64)
    col_indices = np.arange(nx, dtype=np.float64)
    row_indices = np.arange(ny, dtype=np.float64)

    # X projection and phasor
    proj_x = roi_f.sum(axis=0)
    phase_x = np.sum(proj_x * np.exp(-2j * np.pi * col_indices / nx))
    angle_x = np.angle(phase_x)
    if angle_x < 0:
        angle_x += 2 * np.pi
    xc = angle_x * nx / (2 * np.pi)

    # Y projection and phasor
    proj_y = roi_f.sum(axis=1)
    phase_y = np.sum(proj_y * np.exp(-2j * np.pi * row_indices / ny))
    angle_y = np.angle(phase_y)
    if angle_y < 0:
        angle_y += 2 * np.pi
    yc = angle_y * ny / (2 * np.pi)

    bg = float(np.min(roi_f))
    intensity = float(np.sum(roi_f) - bg * ny * nx)

    return {
        'x': xc + col_offset,
        'y': yc + row_offset,
        'intensity': max(intensity, 0.0),
        'sigma_x': 1.5,
        'sigma_y': 1.5,
        'background': bg,
        'chi_squared': 0.0,
    }


def _fit_centroid(roi, row_offset, col_offset):
    """Simple intensity-weighted centroid (fastest, least accurate)."""
    ny, nx = roi.shape
    roi_f = roi.astype(np.float64)
    total = roi_f.sum()
    if total <= 0:
        return None

    rows, cols = np.mgrid[0:ny, 0:nx]
    yc = float((rows * roi_f).sum() / total)
    xc = float((cols * roi_f).sum() / total)

    bg = float(np.min(roi_f))

    return {
        'x': xc + col_offset,
        'y': yc + row_offset,
        'intensity': float(np.max(roi_f)) - bg,
        'sigma_x': 1.0,
        'sigma_y': 1.0,
        'background': bg,
        'chi_squared': 0.0,
    }


# ============================================================================
# Section 6: LOCALIZATION PRECISION (Thompson-Mortensen-Quan)
# ============================================================================

def compute_localization_precision(intensity, background, sigma,
                                    pixel_size=1.0, fitting_method='lsq',
                                    is_emccd=False):
    """Compute localization precision using Thompson-Mortensen-Quan formula.

    Base variance (Thompson et al., 2002)::

        sigma_a2 = sigma**2 + pixel_size**2 / 12   # pixelation correction
        term1 = sigma_a2 / N
        term2 = 8 * pi * sigma_a2**2 * bg_std**2 / (pixel_size**2 * N**2)
        variance = term1 + term2

    Corrections:
    - Mortensen et al. (2010): multiply by 16/9 for LSQ/WLSQ fitting.
    - EMCCD excess noise: multiply by F=2 (Ulbrich & Bhatt, 2015).

    Parameters
    ----------
    intensity : float
        Total molecule intensity (photons). For integrated Gaussian this is
        the total intensity parameter, not peak amplitude.
    background : float
        Background per pixel.
    sigma : float
        PSF standard deviation (same units as pixel_size).
    pixel_size : float
        Pixel size. Set to 1.0 for pixel-unit computation.
    fitting_method : str
        'lsq', 'wlsq', or 'mle'.
    is_emccd : bool
        Whether camera is EMCCD (applies F=2 excess noise factor).

    Returns
    -------
    float
        Localization precision (sqrt of variance).
    """
    if intensity <= 0:
        return np.nan

    N = intensity
    b = background
    s = sigma
    a = pixel_size

    # Effective variance including pixelation
    sa2 = s * s + a * a / 12.0

    # Base variance
    term1 = sa2 / N
    term2 = (8.0 * math.pi * sa2 * sa2 * b * b) / (a * a * N * N)
    variance = term1 + term2

    # Mortensen correction for least-squares fitting
    if fitting_method in ('lsq', 'wlsq'):
        variance *= 16.0 / 9.0

    # EMCCD excess noise factor F=2
    if is_emccd:
        variance *= 2.0

    return math.sqrt(variance)


# ============================================================================
# Section 7: CAMERA MODEL
# ============================================================================

class CameraModel:
    """Full ThunderSTORM camera model.

    Handles ADU-to-photon conversion and provides parameters for
    localization precision computation.

    Parameters
    ----------
    pixel_size : float
        Camera pixel size in nm. Default 108.0.
    photons_per_adu : float
        Photoelectrons per A/D count. Default 3.6.
    baseline : float
        Camera baseline offset in ADU. Default 100.0.
    em_gain : float
        EM multiplication gain. Default 100.0.
    is_emccd : bool
        Whether camera is EMCCD. Default False.
    quantum_efficiency : float
        Sensor quantum efficiency (0-1). Default 1.0.
    """

    def __init__(self, pixel_size=108.0, photons_per_adu=3.6,
                 baseline=100.0, em_gain=100.0, is_emccd=False,
                 quantum_efficiency=1.0):
        self.pixel_size = pixel_size
        self.photons_per_adu = photons_per_adu
        self.baseline = baseline
        self.em_gain = em_gain
        self.is_emccd = is_emccd
        self.quantum_efficiency = quantum_efficiency

    def subtract_baseline(self, image):
        """Subtract camera baseline offset from image."""
        if self.baseline != 0:
            return image.astype(np.float64) - float(self.baseline)
        return image.astype(np.float64)

    def adu_to_photons(self, digital_counts):
        """Convert offset-subtracted ADU to photons.

        Uses the same formula as ThunderSTORM's CameraSetupPlugIn::

            photons = digital_counts * photons_per_adu
                      / quantum_efficiency / em_gain

        Parameters
        ----------
        digital_counts : float or ndarray
            Intensity values in offset-subtracted ADU.

        Returns
        -------
        float or ndarray
            Estimated photon counts.
        """
        em = self.em_gain if self.is_emccd else 1.0
        qe = max(self.quantum_efficiency, 1e-10)
        em = max(em, 1e-10)
        return digital_counts * self.photons_per_adu / qe / em


# ============================================================================
# Section 8: MULTI-EMITTER FITTING
# ============================================================================

def _fit_multi_emitter(roi, row_offset, col_offset, initial_sigma=1.3,
                       max_emitters=5, p_value_threshold=1e-6):
    """Multi-emitter fitting: iteratively add emitters until the
    improvement is no longer statistically significant.

    Uses F-test to determine whether adding another emitter significantly
    improves the fit (p-value < threshold).

    Parameters
    ----------
    roi : ndarray
        2D ROI array.
    row_offset, col_offset : int
        ROI origin offset in full image.
    initial_sigma : float
        Initial PSF sigma.
    max_emitters : int
        Maximum number of emitters to try.
    p_value_threshold : float
        P-value threshold for F-test.

    Returns
    -------
    list of dict
        List of fit results, one per detected emitter.
    """
    from scipy import stats as scipy_stats

    ny, nx = roi.shape
    n_pixels = ny * nx
    roi_f = roi.astype(np.float64)

    jj_arr = np.arange(nx, dtype=np.float64)
    ii_arr = np.arange(ny, dtype=np.float64)

    # Start with single-emitter fit
    result_1 = _fit_integrated_gaussian_mle(
        roi_f, row_offset, col_offset, initial_sigma)
    if result_1 is None:
        return []

    results = [result_1]
    best_chi_sq = result_1['chi_squared'] * n_pixels

    for n_emitters in range(2, max_emitters + 1):
        # Find the pixel with largest residual as seed for new emitter
        model_image = np.zeros_like(roi_f)
        for res in results:
            local_x = res['x'] - col_offset
            local_y = res['y'] - row_offset
            for ii in range(ny):
                for jj in range(nx):
                    model_image[ii, jj] += _integrated_gaussian_value(
                        float(jj), float(ii), local_x, local_y,
                        res['sigma_x'], res['intensity'], 0.0)
        # Add background from last fit
        model_image += results[-1]['background']

        residual = roi_f - model_image
        max_idx = np.unravel_index(np.argmax(residual), residual.shape)

        # Seed new emitter at residual peak position
        seed_y = float(max_idx[0])
        seed_x = float(max_idx[1])

        # Fit new emitter individually, seeded at residual peak
        new_result = _fit_integrated_gaussian_mle(
            roi_f, row_offset, col_offset, initial_sigma)

        if new_result is None:
            break

        # Override initial position with residual peak
        new_result_local = dict(new_result)
        new_result_local['x'] = seed_x + col_offset
        new_result_local['y'] = seed_y + row_offset

        candidate_results = list(results) + [new_result_local]

        # --- Joint re-optimization of ALL emitters ---
        n_emit = len(candidate_results)
        # Parameter vector: [x1, y1, sigma1, I1, x2, y2, sigma2, I2, ..., bg]
        n_params = n_emit * 4 + 1
        dof_new = n_pixels - n_params
        if dof_new <= 0:
            break

        # Build initial parameter vector from individual fits
        params = np.zeros(n_params)
        for k, res in enumerate(candidate_results):
            params[k * 4 + 0] = res['x'] - col_offset  # local x
            params[k * 4 + 1] = res['y'] - row_offset  # local y
            params[k * 4 + 2] = math.sqrt(res['sigma_x'])  # sqrt(sigma)
            params[k * 4 + 3] = math.sqrt(res['intensity'])  # sqrt(I)
        params[-1] = math.sqrt(candidate_results[-1]['background'])  # sqrt(bg)

        # Joint LM optimization
        lambda_lm = 0.001
        max_joint_iter = 200

        def _joint_model_and_residual(p, roi_data):
            """Compute chi-squared for joint parameters."""
            bg = p[-1] * p[-1]
            chi_sq = 0.0
            for ii in range(ny):
                fi = ii_arr[ii]
                for jj in range(nx):
                    fj = jj_arr[jj]
                    mv = bg
                    for k in range(n_emit):
                        lx = p[k * 4 + 0]
                        ly = p[k * 4 + 1]
                        sig = p[k * 4 + 2] * p[k * 4 + 2]
                        inten = p[k * 4 + 3] * p[k * 4 + 3]
                        mv += _integrated_gaussian_value(
                            fj, fi, lx, ly, sig, inten, 0.0)
                    r = roi_data[ii, jj] - mv
                    chi_sq += r * r
            return chi_sq

        joint_chi_sq = _joint_model_and_residual(params, roi_f)

        joint_success = False
        for iteration in range(max_joint_iter):
            H = np.zeros((n_params, n_params))
            g_vec = np.zeros(n_params)

            bg_val = params[-1] * params[-1]

            for ii in range(ny):
                fi = ii_arr[ii]
                for jj in range(nx):
                    fj = jj_arr[jj]
                    mv = bg_val
                    for k in range(n_emit):
                        lx = params[k * 4 + 0]
                        ly = params[k * 4 + 1]
                        sig = params[k * 4 + 2] * params[k * 4 + 2]
                        inten = params[k * 4 + 3] * params[k * 4 + 3]
                        mv += _integrated_gaussian_value(
                            fj, fi, lx, ly, sig, inten, 0.0)

                    residual_val = roi_f[ii, jj] - mv

                    J = np.zeros(n_params)
                    for k in range(n_emit):
                        lx = params[k * 4 + 0]
                        ly = params[k * 4 + 1]
                        sig = params[k * 4 + 2] * params[k * 4 + 2]
                        inten = params[k * 4 + 3] * params[k * 4 + 3]
                        J_orig = _integrated_gaussian_jacobian(
                            fj, fi, lx, ly, sig, inten, 0.0)
                        # dx, dy are direct; dsigma and dI use chain rule
                        J[k * 4 + 0] = J_orig[0]
                        J[k * 4 + 1] = J_orig[1]
                        J[k * 4 + 2] = J_orig[2] * 2.0 * params[k * 4 + 2]
                        J[k * 4 + 3] = J_orig[3] * 2.0 * params[k * 4 + 3]
                    # Background derivative: d/d(p_bg) of bg = 2*p_bg
                    J[-1] = 2.0 * params[-1]

                    g_vec += J * residual_val
                    H += np.outer(J, J)

            # Augment diagonal
            H_aug = H.copy()
            diag_h = np.diag(H_aug).copy()
            diag_h[diag_h < 1e-30] = 1e-30
            H_aug[np.diag_indices(n_params)] = diag_h * (1.0 + lambda_lm)

            try:
                dp = np.linalg.solve(H_aug, g_vec)
            except np.linalg.LinAlgError:
                break

            params_new = params + dp

            # Enforce sigma bounds
            for k in range(n_emit):
                sig_new = params_new[k * 4 + 2] * params_new[k * 4 + 2]
                sig_new = min(max(sig_new, 0.1), 10.0)
                params_new[k * 4 + 2] = math.sqrt(sig_new)

            chi_sq_new = _joint_model_and_residual(params_new, roi_f)

            if chi_sq_new < joint_chi_sq:
                rel_change = abs(joint_chi_sq - chi_sq_new) / max(
                    joint_chi_sq, 1e-30)
                params = params_new
                joint_chi_sq = chi_sq_new
                lambda_lm = max(lambda_lm / 10.0, 1e-7)
                if rel_change < 1e-8:
                    joint_success = True
                    break
            else:
                lambda_lm *= 10.0
                if lambda_lm > 1e10:
                    break
        else:
            # Completed all iterations without early break
            joint_success = True

        if joint_success:
            new_chi_sq = joint_chi_sq
        else:
            # Fall back to individual fits
            new_chi_sq = new_result['chi_squared'] * n_pixels

        # F-test: is the improvement significant?
        dof_old = n_pixels - (len(results) * 4 + 1)
        dof_new = n_pixels - n_params

        if dof_new <= 0 or dof_old <= 0:
            break

        f_stat = ((best_chi_sq - new_chi_sq) / 5.0) / (new_chi_sq / dof_new)

        if f_stat <= 0:
            break

        # Proper F-test using F-distribution survival function
        p_value = scipy_stats.f.sf(f_stat, dfn=5, dfd=dof_new)

        if p_value < p_value_threshold:
            if joint_success:
                # Build results from joint parameters
                bg_final = params[-1] * params[-1]
                joint_results = []
                for k in range(n_emit):
                    joint_results.append({
                        'x': params[k * 4 + 0] + col_offset,
                        'y': params[k * 4 + 1] + row_offset,
                        'intensity': params[k * 4 + 3] * params[k * 4 + 3],
                        'sigma_x': params[k * 4 + 2] * params[k * 4 + 2],
                        'sigma_y': params[k * 4 + 2] * params[k * 4 + 2],
                        'background': bg_final,
                        'chi_squared': new_chi_sq / n_pixels,
                    })
                results = joint_results
            else:
                results.append(new_result)
            best_chi_sq = new_chi_sq
        else:
            break

    return results


# ============================================================================
# Section 9: POST-PROCESSING
# ============================================================================

class DriftCorrector:
    """Correct for sample drift using cross-correlation of reconstructed
    images from localization data.

    Parameters
    ----------
    smoothing : float
        Smoothing parameter for drift trajectory. Default 0.25.
    segment_frames : int
        Number of frames per segment for cross-correlation. Default 500.
    """

    def __init__(self, smoothing=0.25, segment_frames=500):
        self.smoothing = smoothing
        self.segment_frames = segment_frames
        self.drift_x = None
        self.drift_y = None

    def compute_drift(self, localizations, n_frames, pixel_size_render=10.0):
        """Compute drift using cross-correlation.

        Parameters
        ----------
        localizations : ndarray
            (M, 6+) array [frame, x, y, intensity, sigma, uncertainty].
        n_frames : int
            Total number of frames.
        pixel_size_render : float
            Pixel size for rendering histograms (in nm).
        """
        if len(localizations) == 0:
            self.drift_x = np.zeros(n_frames)
            self.drift_y = np.zeros(n_frames)
            return

        frames = localizations[:, 0]
        x = localizations[:, 1]
        y = localizations[:, 2]

        x_max = int(np.max(x) / pixel_size_render) + 1
        y_max = int(np.max(y) / pixel_size_render) + 1
        img_size = (max(y_max, 1), max(x_max, 1))

        n_segments = max(1, n_frames // self.segment_frames + 1)
        segment_drift_x = []
        segment_drift_y = []
        segment_centers = []

        # Reference: first segment
        ref_mask = frames < self.segment_frames
        ref_img = self._render_histogram(
            x[ref_mask], y[ref_mask], pixel_size_render, img_size)

        for seg in range(n_segments):
            start = seg * self.segment_frames
            end = min((seg + 1) * self.segment_frames, n_frames)
            if start >= n_frames:
                break

            seg_mask = (frames >= start) & (frames < end)
            seg_img = self._render_histogram(
                x[seg_mask], y[seg_mask], pixel_size_render, img_size)

            # Cross-correlate
            from scipy.signal import correlate2d
            xcorr = correlate2d(ref_img, seg_img, mode='same')
            peak_y, peak_x = np.unravel_index(np.argmax(xcorr), xcorr.shape)
            cy, cx = np.array(xcorr.shape) // 2

            # Sub-pixel refinement via parabolic interpolation
            sub_x, sub_y = float(peak_x), float(peak_y)
            if 0 < peak_x < xcorr.shape[1] - 1:
                left = xcorr[peak_y, peak_x - 1]
                center = xcorr[peak_y, peak_x]
                right = xcorr[peak_y, peak_x + 1]
                denom = 2.0 * center - left - right
                if abs(denom) > 1e-10:
                    sub_x += (left - right) / (2.0 * denom)
            if 0 < peak_y < xcorr.shape[0] - 1:
                top = xcorr[peak_y - 1, peak_x]
                center = xcorr[peak_y, peak_x]
                bottom = xcorr[peak_y + 1, peak_x]
                denom = 2.0 * center - top - bottom
                if abs(denom) > 1e-10:
                    sub_y += (top - bottom) / (2.0 * denom)

            segment_drift_x.append((sub_x - cx) * pixel_size_render)
            segment_drift_y.append((sub_y - cy) * pixel_size_render)
            segment_centers.append((start + end) / 2.0)

        if len(segment_centers) < 2:
            self.drift_x = np.zeros(n_frames)
            self.drift_y = np.zeros(n_frames)
            return

        # Cubic spline smoothing
        from scipy.interpolate import UnivariateSpline
        try:
            s_factor = max(len(segment_centers) * self.smoothing, 1.0)
            spl_x = UnivariateSpline(segment_centers, segment_drift_x, s=s_factor)
            spl_y = UnivariateSpline(segment_centers, segment_drift_y, s=s_factor)
            all_frames = np.arange(n_frames)
            self.drift_x = spl_x(all_frames)
            self.drift_y = spl_y(all_frames)
        except Exception:
            # Fallback to linear interpolation + Gaussian smoothing
            all_frames = np.arange(n_frames)
            self.drift_x = np.interp(all_frames, segment_centers, segment_drift_x)
            self.drift_y = np.interp(all_frames, segment_centers, segment_drift_y)
            if n_frames > 3:
                sigma_smooth = max(n_frames * self.smoothing / 6.0, 1.0)
                self.drift_x = ndimage.gaussian_filter1d(
                    self.drift_x, sigma_smooth)
                self.drift_y = ndimage.gaussian_filter1d(
                    self.drift_y, sigma_smooth)

    def apply_correction(self, localizations):
        """Apply drift correction to localizations.

        Parameters
        ----------
        localizations : ndarray
            (M, 6+) array [frame, x, y, ...].

        Returns
        -------
        ndarray
            Corrected localizations (copy).
        """
        if self.drift_x is None:
            return localizations.copy()

        corrected = localizations.copy()
        for i in range(len(corrected)):
            frame_idx = int(corrected[i, 0])
            if frame_idx < len(self.drift_x):
                corrected[i, 1] -= self.drift_x[frame_idx]
                corrected[i, 2] -= self.drift_y[frame_idx]
        return corrected

    @staticmethod
    def _render_histogram(x, y, pixel_size, img_size):
        """Render localization histogram."""
        img = np.zeros(img_size)
        if len(x) == 0:
            return img
        xi = (x / pixel_size).astype(int)
        yi = (y / pixel_size).astype(int)
        valid = ((xi >= 0) & (xi < img_size[1]) &
                 (yi >= 0) & (yi < img_size[0]))
        xi, yi = xi[valid], yi[valid]
        for xv, yv in zip(xi, yi):
            img[yv, xv] += 1
        return img


class MolecularMerger:
    """Merge re-appearing molecules across consecutive frames.

    Parameters
    ----------
    max_distance : float
        Maximum distance (same units as x, y) to merge. Default 50.
    max_frame_gap : int
        Maximum frame gap for re-appearance. Default 1.
    """

    def __init__(self, max_distance=50.0, max_frame_gap=1):
        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap

    def merge(self, localizations):
        """Merge re-appearing molecules.

        Parameters
        ----------
        localizations : ndarray
            (M, 6+) array [frame, x, y, intensity, sigma, uncertainty].

        Returns
        -------
        ndarray
            Merged localizations.
        """
        if len(localizations) == 0 or not _HAS_KDTREE:
            return localizations

        sort_idx = np.argsort(localizations[:, 0])
        locs = localizations[sort_idx]
        keep = np.ones(len(locs), dtype=bool)

        frames = locs[:, 0]
        unique_frames = np.unique(frames)

        for i, frame in enumerate(unique_frames[:-1]):
            curr_mask = frames == frame
            curr_idx = np.where(curr_mask)[0]
            if len(curr_idx) == 0:
                continue

            curr_pos = locs[curr_idx, 1:3]
            tree = cKDTree(curr_pos)

            for gap in range(1, self.max_frame_gap + 1):
                if i + gap >= len(unique_frames):
                    break
                next_frame = unique_frames[i + gap]
                next_mask = frames == next_frame
                next_idx = np.where(next_mask)[0]

                for j, nidx in enumerate(next_idx):
                    if not keep[nidx]:
                        continue
                    dist, idx = tree.query(locs[nidx, 1:3])
                    if dist <= self.max_distance:
                        keep[nidx] = False

        return locs[keep]


class LocalizationFilter:
    """Filter localizations by quality criteria.

    Parameters
    ----------
    min_intensity : float, optional
    max_intensity : float, optional
    max_uncertainty : float, optional
    min_sigma : float, optional
    max_sigma : float, optional
    """

    def __init__(self, min_intensity=None, max_intensity=None,
                 max_uncertainty=None, min_sigma=None, max_sigma=None):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.max_uncertainty = max_uncertainty
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def filter(self, localizations):
        """Apply filters. Columns: [frame, x, y, intensity, sigma, unc]."""
        if len(localizations) == 0:
            return localizations

        mask = np.ones(len(localizations), dtype=bool)

        # intensity is column 3
        if self.min_intensity is not None:
            mask &= localizations[:, 3] >= self.min_intensity
        if self.max_intensity is not None:
            mask &= localizations[:, 3] <= self.max_intensity

        # sigma is column 4
        if self.min_sigma is not None:
            mask &= localizations[:, 4] >= self.min_sigma
        if self.max_sigma is not None:
            mask &= localizations[:, 4] <= self.max_sigma

        # uncertainty is column 5
        if self.max_uncertainty is not None and localizations.shape[1] > 5:
            mask &= localizations[:, 5] <= self.max_uncertainty

        return localizations[mask]


class LocalDensityFilter:
    """Filter by local density (removes isolated localizations).

    Parameters
    ----------
    radius : float
        Search radius. Default 50.
    min_neighbors : int
        Minimum number of neighbors required. Default 3.
    """

    def __init__(self, radius=50.0, min_neighbors=3):
        self.radius = radius
        self.min_neighbors = min_neighbors

    def filter(self, localizations):
        """Filter by local density."""
        if len(localizations) == 0 or not _HAS_KDTREE:
            return localizations

        positions = localizations[:, 1:3]
        tree = cKDTree(positions)
        neighbors = tree.query_ball_point(positions, self.radius)
        counts = np.array([len(n) - 1 for n in neighbors])
        mask = counts >= self.min_neighbors
        return localizations[mask]


class DuplicateRemover:
    """Remove duplicate localizations from multi-emitter fitting.

    Parameters
    ----------
    max_distance : float
        Maximum distance to consider duplicates. Default 20.
    """

    def __init__(self, max_distance=20.0):
        self.max_distance = max_distance

    def remove(self, localizations):
        """Remove duplicates within same frame, keeping brighter one."""
        if len(localizations) == 0 or not _HAS_KDTREE:
            return localizations

        keep = np.ones(len(localizations), dtype=bool)

        for frame in np.unique(localizations[:, 0]):
            frame_mask = localizations[:, 0] == frame
            frame_idx = np.where(frame_mask)[0]
            if len(frame_idx) <= 1:
                continue

            positions = localizations[frame_idx, 1:3]
            tree = cKDTree(positions)
            pairs = tree.query_pairs(self.max_distance)

            for i, j in pairs:
                idx_i = frame_idx[i]
                idx_j = frame_idx[j]
                if localizations[idx_i, 3] < localizations[idx_j, 3]:
                    keep[idx_i] = False
                else:
                    keep[idx_j] = False

        return localizations[keep]


# ============================================================================
# Section 10: MAIN DETECTOR CLASS
# ============================================================================

class ThunderSTORMDetector:
    """ThunderSTORM-style multi-stage particle detection pipeline.

    Faithfully replicates the ThunderSTORM ImageJ plugin pipeline
    (Ovesny et al., Bioinformatics 2014) including erf-based integrated
    Gaussian PSF model, Levenberg-Marquardt optimizer with squared
    parameterization, Thompson-Mortensen-Quan localization precision, full
    camera model, and all filter/detector/fitter types.

    Pipeline stages per frame:
    1. Subtract camera baseline.
    2. Apply image filter.
    3. Evaluate threshold expression.
    4. Detect candidates.
    5. Remove border detections.
    6. Fit sub-pixel PSF.
    7. Validate localizations within bounds.
    8. Compute localization precision.
    9. (Optional) Scale to physical units (nm).

    Parameters
    ----------
    filter_type : str
        Image filter: 'wavelet' (default), 'gaussian', 'dog',
        'lowered_gaussian', 'difference_of_averaging', 'median',
        'box', 'none'.
    detector_type : str
        Candidate detector: 'local_max' (default), 'nms',
        'centroid', 'grid'.
    fitter_type : str
        Sub-pixel fitter: 'gaussian_lsq' (default), 'gaussian_wlsq',
        'gaussian_mle', 'elliptical_gaussian_mle', 'radial_symmetry',
        'phasor', 'centroid', 'multi_emitter'.
    threshold : str, float, or None
        Detection threshold. Can be a ThunderSTORM expression string
        like ``'std(Wave.F1)'`` or ``'2*std(Wave.F1)'``, a numeric
        value, or None for automatic (defaults to ``'std(Wave.F1)'``).
    roi_size : int
        Side length of fitting ROI in pixels (odd integer). Default 5.
    camera_params : dict, optional
        Camera model parameters: pixel_size, photons_per_adu, baseline,
        em_gain, is_emccd, quantum_efficiency.
    filter_params : dict, optional
        Additional parameters for the selected filter.
    detector_params : dict, optional
        Additional parameters for the selected detector.
    fitter_params : dict, optional
        Additional parameters for the selected fitter (e.g.
        initial_sigma, max_emitters, p_value_threshold).
    convert_to_nm : bool
        If True, multiply x, y, sigma, uncertainty by pixel_size.
        Default False.
    """

    def __init__(self, filter_type='wavelet', detector_type='local_max',
                 fitter_type='gaussian_lsq', threshold=None,
                 roi_size=5, camera_params=None,
                 filter_params=None, detector_params=None,
                 fitter_params=None, convert_to_nm=False):

        # Camera model
        if camera_params is None:
            camera_params = {}
        self.camera = CameraModel(
            pixel_size=camera_params.get('pixel_size', 108.0),
            photons_per_adu=camera_params.get('photons_per_adu', 3.6),
            baseline=camera_params.get('baseline', 0.0),
            em_gain=camera_params.get('em_gain', 1.0),
            is_emccd=camera_params.get('is_emccd', False),
            quantum_efficiency=camera_params.get('quantum_efficiency', 1.0),
        )

        # Filter
        if filter_params is None:
            filter_params = {}
        self.image_filter = create_filter(filter_type, **filter_params)

        # Detector
        if detector_params is None:
            detector_params = {}
        self.detector = create_detector(detector_type, **detector_params)

        # Fitter parameters
        self.fitter_type = fitter_type.lower()
        if fitter_params is None:
            fitter_params = {}
        self.initial_sigma = fitter_params.get('initial_sigma', 1.3)
        self.max_emitters = fitter_params.get('max_emitters', 5)
        self.p_value_threshold = fitter_params.get('p_value_threshold', 1e-6)

        # Threshold
        if threshold is None:
            self.threshold_expression = 'std(Wave.F1)'
        else:
            self.threshold_expression = threshold

        # ROI size
        self.roi_size = roi_size if roi_size % 2 == 1 else roi_size + 1
        self.fit_radius = self.roi_size // 2

        # Output options
        self.convert_to_nm = convert_to_nm

        # Supported fitter types
        self._fitter_types = {
            'gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle',
            'elliptical_gaussian_mle', 'radial_symmetry', 'phasor',
            'centroid', 'multi_emitter',
        }
        if self.fitter_type not in self._fitter_types:
            raise ValueError(
                f"Unknown fitter_type {fitter_type!r}. "
                f"Available: {sorted(self._fitter_types)}")

    def _get_fitting_method_name(self):
        """Return short fitting method name for precision computation."""
        if self.fitter_type in ('gaussian_lsq',):
            return 'lsq'
        elif self.fitter_type in ('gaussian_wlsq',):
            return 'wlsq'
        else:
            return 'mle'

    def _fit_single(self, image, row, col):
        """Fit a single candidate at (row, col) in the image.

        Returns
        -------
        dict or list of dict
            Fit result(s), or None on failure.
        """
        h, w = image.shape
        r = self.fit_radius

        r0 = max(0, row - r)
        r1 = min(h, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(w, col + r + 1)

        roi = image[r0:r1, c0:c1]
        if roi.shape[0] < 3 or roi.shape[1] < 3:
            return None

        if self.fitter_type == 'gaussian_lsq':
            return _fit_integrated_gaussian_lsq(
                roi, r0, c0, self.initial_sigma)
        elif self.fitter_type == 'gaussian_wlsq':
            return _fit_integrated_gaussian_wlsq(
                roi, r0, c0, self.initial_sigma)
        elif self.fitter_type == 'gaussian_mle':
            return _fit_integrated_gaussian_mle(
                roi, r0, c0, self.initial_sigma)
        elif self.fitter_type == 'elliptical_gaussian_mle':
            return _fit_elliptical_gaussian_mle(
                roi, r0, c0, self.initial_sigma)
        elif self.fitter_type == 'radial_symmetry':
            return _fit_radial_symmetry(roi, r0, c0)
        elif self.fitter_type == 'phasor':
            return _fit_phasor(roi, r0, c0)
        elif self.fitter_type == 'centroid':
            return _fit_centroid(roi, r0, c0)
        elif self.fitter_type == 'multi_emitter':
            results = _fit_multi_emitter(
                roi, r0, c0, self.initial_sigma,
                self.max_emitters, self.p_value_threshold)
            return results if results else None
        return None

    def _validate_localization(self, fit, image_shape):
        """Check that a localization is within image bounds."""
        height, width = image_shape
        x = fit['x']
        y = fit['y']
        return (0 <= x < width) and (0 <= y < height)

    def detect_frame(self, frame_2d):
        """Detect and localise particles in a single 2D frame.

        Parameters
        ----------
        frame_2d : ndarray
            2D numpy array (single image frame), in ADU.

        Returns
        -------
        ndarray
            (N, 7) array with columns
            ``[x, y, intensity, sigma_x, sigma_y, background, uncertainty]``.
            Returns ``(0, 7)`` if no particles found. If ``convert_to_nm``
            is True, x/y/sigma/uncertainty are in nm.
        """
        image = np.asarray(frame_2d, dtype=np.float64)
        if image.ndim != 2 or image.size == 0:
            return np.empty((0, 7))

        # Step 1: Subtract camera baseline
        image = self.camera.subtract_baseline(image)

        # Step 2: Apply image filter
        filtered = self.image_filter.apply(image)

        # Step 3: Compute threshold
        threshold = compute_threshold_expression(
            image, filtered, self.threshold_expression,
            filter_obj=self.image_filter)

        # Step 4: Detect candidates
        positions = self.detector.detect(filtered, threshold)

        if len(positions) == 0:
            return np.empty((0, 7))

        # Step 5: Remove border detections before fitting
        border_width = max(5, int(self.fit_radius * 1.5))
        positions = _remove_border_detections(
            positions, image.shape, border=border_width)

        if len(positions) == 0:
            return np.empty((0, 7))

        # Step 6: Fit PSF at each candidate
        fitting_method = self._get_fitting_method_name()
        results = []

        for pos in positions:
            row = int(round(pos[0]))
            col = int(round(pos[1]))

            fit = self._fit_single(image, row, col)
            if fit is None:
                continue

            # Handle multi-emitter (returns list)
            if isinstance(fit, list):
                for f in fit:
                    if f is not None and self._validate_localization(
                            f, image.shape):
                        unc = compute_localization_precision(
                            f['intensity'], f['background'],
                            (f['sigma_x'] + f['sigma_y']) / 2.0,
                            1.0, fitting_method, self.camera.is_emccd)
                        results.append([
                            f['x'], f['y'], f['intensity'],
                            f['sigma_x'], f['sigma_y'],
                            f['background'], unc,
                        ])
            else:
                # Step 7: Validate localization within bounds
                if not self._validate_localization(fit, image.shape):
                    continue

                # Step 8: Compute localization precision
                unc = compute_localization_precision(
                    fit['intensity'], fit['background'],
                    (fit['sigma_x'] + fit['sigma_y']) / 2.0,
                    1.0, fitting_method, self.camera.is_emccd)

                results.append([
                    fit['x'], fit['y'], fit['intensity'],
                    fit['sigma_x'], fit['sigma_y'],
                    fit['background'], unc,
                ])

        if not results:
            return np.empty((0, 7))

        result_array = np.array(results)

        # Step 9: Convert to nm if requested
        if self.convert_to_nm:
            ps = self.camera.pixel_size
            result_array[:, 0] *= ps  # x
            result_array[:, 1] *= ps  # y
            result_array[:, 3] *= ps  # sigma_x
            result_array[:, 4] *= ps  # sigma_y
            result_array[:, 6] *= ps  # uncertainty

        return result_array

    def detect_stack(self, stack_3d, callback=None):
        """Detect and localise particles in a 3D image stack.

        Parameters
        ----------
        stack_3d : ndarray
            (T, H, W) array of image frames.
        callback : callable, optional
            Called as ``callback(frame_idx)`` after each frame.

        Returns
        -------
        ndarray
            (M, 8) array with columns
            ``[frame, x, y, intensity, sigma_x, sigma_y, background,
            uncertainty]``. Returns ``(0, 8)`` if no particles found.
        """
        stack = np.asarray(stack_3d)
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        if stack.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {stack.ndim}D")

        n_frames = len(stack)
        all_detections = []

        logger.info("ThunderSTORM detection: %d frames, filter=%s, "
                     "detector=%s, fitter=%s",
                     n_frames, self.image_filter.name,
                     self.detector.name, self.fitter_type)

        for t in range(n_frames):
            dets = self.detect_frame(stack[t])
            if len(dets) > 0:
                frame_col = np.full((len(dets), 1), t, dtype=np.float64)
                all_detections.append(np.hstack([frame_col, dets]))

            if callback is not None:
                callback(t)

        if all_detections:
            result = np.vstack(all_detections)
            logger.info("ThunderSTORM detection complete: %d localisations "
                        "in %d frames", len(result), n_frames)
            return result

        logger.info("ThunderSTORM detection complete: 0 localisations")
        return np.empty((0, 8))

    def detect_and_fit(self, image_stack, show_progress=False):
        """Compatibility method matching the original plugin API.

        Parameters
        ----------
        image_stack : ndarray
            3D (frames, H, W) or 2D image.
        show_progress : bool
            If True, log progress.

        Returns
        -------
        dict
            Localization results with keys: x, y, frame, intensity,
            background, sigma_x, sigma_y, uncertainty, chi_squared.
            x and y are in nm if convert_to_nm is True.
        """
        stack = np.asarray(image_stack)
        if stack.ndim == 2:
            stack = stack[np.newaxis]

        n_frames = stack.shape[0]
        all_x = []
        all_y = []
        all_frame = []
        all_intensity = []
        all_background = []
        all_sigma_x = []
        all_sigma_y = []
        all_uncertainty = []
        all_chi_squared = []

        for t in range(n_frames):
            dets = self.detect_frame(stack[t])
            if len(dets) > 0:
                # Columns: x, y, intensity, sigma_x, sigma_y, background, unc
                all_x.append(dets[:, 0])
                all_y.append(dets[:, 1])
                all_frame.append(np.full(len(dets), t, dtype=np.float64))
                all_intensity.append(dets[:, 2])
                all_sigma_x.append(dets[:, 3])
                all_sigma_y.append(dets[:, 4])
                all_background.append(dets[:, 5])
                all_uncertainty.append(dets[:, 6])
                # chi_squared not directly in output; use 0
                all_chi_squared.append(np.zeros(len(dets)))

            if show_progress and (t + 1) % max(1, n_frames // 10) == 0:
                logger.info("  Frame %d/%d", t + 1, n_frames)

        def _concat(arrays):
            return np.concatenate(arrays) if arrays else np.array([])

        return {
            'x': _concat(all_x),
            'y': _concat(all_y),
            'frame': _concat(all_frame),
            'intensity': _concat(all_intensity),
            'background': _concat(all_background),
            'sigma_x': _concat(all_sigma_x),
            'sigma_y': _concat(all_sigma_y),
            'uncertainty': _concat(all_uncertainty),
            'chi_squared': _concat(all_chi_squared),
        }

    # Post-processing convenience methods

    def apply_drift_correction(self, localizations, n_frames=None,
                               **kwargs):
        """Apply drift correction to stack localizations.

        Parameters
        ----------
        localizations : ndarray
            (M, 8) array from ``detect_stack``.
        n_frames : int, optional
            Total number of frames.
        **kwargs
            Additional parameters for DriftCorrector.

        Returns
        -------
        ndarray
            Corrected localizations.
        """
        if len(localizations) == 0:
            return localizations
        if n_frames is None:
            n_frames = int(localizations[:, 0].max()) + 1
        corrector = DriftCorrector(**kwargs)
        corrector.compute_drift(localizations, n_frames)
        return corrector.apply_correction(localizations)

    def merge_molecules(self, localizations, max_distance=50.0,
                        max_frame_gap=1):
        """Merge re-appearing molecules.

        Parameters
        ----------
        localizations : ndarray
            (M, 8) array from ``detect_stack``.
        max_distance : float
            Maximum merge distance.
        max_frame_gap : int
            Maximum frame gap.

        Returns
        -------
        ndarray
            Merged localizations.
        """
        merger = MolecularMerger(max_distance, max_frame_gap)
        return merger.merge(localizations)

    def filter_localizations(self, localizations, **filter_params):
        """Filter localizations by quality criteria.

        Parameters
        ----------
        localizations : ndarray
            (M, 8) array from ``detect_stack``.
        **filter_params
            Parameters for LocalizationFilter.

        Returns
        -------
        ndarray
            Filtered localizations.
        """
        filt = LocalizationFilter(**filter_params)
        return filt.filter(localizations)

    def filter_by_density(self, localizations, radius=50.0,
                          min_neighbors=3):
        """Filter by local density.

        Parameters
        ----------
        localizations : ndarray
            (M, 8) array from ``detect_stack``.
        radius : float
            Search radius.
        min_neighbors : int
            Minimum neighbor count.

        Returns
        -------
        ndarray
            Filtered localizations.
        """
        filt = LocalDensityFilter(radius, min_neighbors)
        return filt.filter(localizations)


# ============================================================================
# Section 11: CONVENIENCE FUNCTIONS
# ============================================================================

def is_thunderstorm_available():
    """Return True. The pipeline is always available (built-in)."""
    return True


def get_available_filters():
    """Return list of available filter types."""
    return ['wavelet', 'gaussian', 'dog', 'lowered_gaussian',
            'difference_of_averaging', 'median', 'box', 'none']


def get_available_detectors():
    """Return list of available detector types."""
    return ['local_max', 'nms', 'centroid', 'grid']


def get_available_fitters():
    """Return list of available PSF fitters."""
    return ['gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle',
            'elliptical_gaussian_mle', 'radial_symmetry', 'phasor',
            'centroid', 'multi_emitter']


def get_default_threshold_expressions():
    """Return list of common threshold expressions."""
    return [
        'std(Wave.F1)',
        '2*std(Wave.F1)',
        '3*std(Wave.F1)',
        'mean(Wave.F1) + 3*std(Wave.F1)',
        '100',
    ]
