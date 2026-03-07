"""Geometric track features: radius of gyration, asymmetry, fractal dimension, etc.

Exact replica of the algorithms from locsAndTracksPlotter/joinTracks.py
(RadiusGyrationAsymmetrySkewnessKurtosis, FractalDimension,
NetDisplacementEfficiency, SummedSinesCosines) and from
spt_batch_analysis (FeatureCalculator, GeometricAnalyzer).
"""
import numpy as np
import math
from scipy import stats, spatial
from ...logger import logger


# ---------------------------------------------------------------------------
# Radius of gyration (tensor method) -- exact joinTracks.py algorithm
# ---------------------------------------------------------------------------

def radius_of_gyration(positions, method='tensor'):
    """Compute radius of gyration and related shape features.

    Uses the exact tensor eigenvalue decomposition from the original
    locsAndTracksPlotter/joinTracks.py::RadiusGyrationAsymmetrySkewnessKurtosis
    and spt_batch_analysis::FeatureCalculator.radius_gyration_asymmetry.

    Args:
        positions: (N, 2) array of [x, y] positions
        method: 'tensor' (eigenvalue decomposition) or 'simple' (RMS distance)

    Returns:
        dict with keys: rg, asymmetry, skewness, kurtosis
        Returns default values (rg=0, etc.) if positions has fewer than 3 points.
    """
    defaults = {'rg': 0.0, 'asymmetry': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
    if len(positions) < 3:
        return defaults

    positions = np.asarray(positions, dtype=np.float64)

    if method == 'simple':
        rg = get_radius_of_gyration_simple(positions)
        return {'rg': float(rg), 'asymmetry': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}

    # Tensor method -- exact plugin code
    center = positions.mean(0)
    normed_points = positions - center[None, :]
    radiusGyration_tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(positions)
    eig_values, eig_vectors = np.linalg.eig(radiusGyration_tensor)

    radius_gyration_value = np.sqrt(np.sum(eig_values))

    # Asymmetry -- exact plugin formula
    asymmetry_numerator = pow((eig_values[0] - eig_values[1]), 2)
    asymmetry_denominator = 2 * (pow((eig_values[0] + eig_values[1]), 2))
    if asymmetry_denominator > 0:
        asymmetry_value = -math.log(1 - (asymmetry_numerator / asymmetry_denominator))
    else:
        asymmetry_value = 0.0

    # Skewness and kurtosis from projection onto dominant eigenvector
    maxcol = list(eig_values).index(max(eig_values))
    dominant_eig_vect = eig_vectors[:, maxcol]

    points_a = positions[:-1]
    points_b = positions[1:]
    ba = points_b - points_a
    proj_ba_dom_eig_vect = np.dot(ba, dominant_eig_vect) / np.power(np.linalg.norm(dominant_eig_vect), 2)

    skewness_value = float(stats.skew(proj_ba_dom_eig_vect)) if len(proj_ba_dom_eig_vect) > 0 else 0.0
    kurtosis_value = float(stats.kurtosis(proj_ba_dom_eig_vect)) if len(proj_ba_dom_eig_vect) > 0 else 0.0

    return {
        'rg': float(np.real(radius_gyration_value)),
        'asymmetry': float(np.real(asymmetry_value)),
        'skewness': skewness_value,
        'kurtosis': kurtosis_value,
    }


# ---------------------------------------------------------------------------
# Fractal dimension -- exact joinTracks.py::FractalDimension
# ---------------------------------------------------------------------------

def fractal_dimension(points_array):
    """Fractal dimension of a track.

    Exact replica of joinTracks.py::FractalDimension (Vivek's code).

    Returns 1.0 for degenerate cases (< 3 points, collinear, QHull failure).
    """
    points_array = np.asarray(points_array, dtype=np.float64)
    if len(points_array) < 3:
        return 1.0

    try:
        # Check if points are on the same line -- exact plugin code
        x0, y0 = points_array[0]
        points = [(x, y) for x, y in points_array if ((x != x0) or (y != y0))]
        if len(points) < 2:
            return 1.0
        slopes = [((y - y0) / (x - x0)) if (x != x0) else None for x, y in points]
        if all(s == slopes[0] for s in slopes):
            return 1.0  # Collinear

        total_path_length = np.sum(
            pow(np.sum(pow(points_array[1:, :] - points_array[:-1, :], 2), axis=1), 0.5)
        )
        stepCount = len(points_array)

        candidates = points_array[spatial.ConvexHull(points_array).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        maxIndex = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        largestDistance = dist_mat[maxIndex]

        if total_path_length > 0 and largestDistance > 0:
            fractal_dimension_value = math.log(stepCount) / math.log(
                stepCount * largestDistance * math.pow(total_path_length, -1)
            )
            return float(fractal_dimension_value)
        else:
            return 1.0
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Net displacement & efficiency -- exact joinTracks.py::NetDisplacementEfficiency
# ---------------------------------------------------------------------------

def net_displacement_efficiency(points_array):
    """Net displacement and path efficiency.

    Exact replica of joinTracks.py::NetDisplacementEfficiency (Vivek's code).

    Returns:
        dict with keys: net_displacement, efficiency
    """
    points_array = np.asarray(points_array, dtype=np.float64)
    if len(points_array) < 2:
        return {'net_displacement': 0.0, 'efficiency': 0.0}

    net_displacement_value = float(np.linalg.norm(points_array[0] - points_array[-1]))

    if len(points_array) < 3:
        return {'net_displacement': net_displacement_value, 'efficiency': 1.0}

    netDispSquared = pow(net_displacement_value, 2)
    points_a = points_array[1:, :]
    points_b = points_array[:-1, :]
    dist_ab_SumSquared = sum(pow(np.linalg.norm(points_a - points_b, axis=1), 2))

    if dist_ab_SumSquared > 0:
        efficiency_value = netDispSquared / ((len(points_array) - 1) * dist_ab_SumSquared)
    else:
        efficiency_value = 0.0

    return {'net_displacement': net_displacement_value, 'efficiency': float(efficiency_value)}


# ---------------------------------------------------------------------------
# Straightness / SummedSinesCosines -- exact joinTracks.py::SummedSinesCosines
# ---------------------------------------------------------------------------

def straightness(points_array):
    """Bending and straightness features using sine/cosine analysis.

    Exact replica of joinTracks.py::SummedSinesCosines (Vivek's code).

    Returns:
        dict with keys: straightness (mean cosine), mean_sin, mean_cos,
                        sin_vals, cos_vals
    """
    points_array = np.asarray(points_array, dtype=np.float64)
    if len(points_array) < 3:
        return {'straightness': 0.0, 'mean_sin': 0.0, 'mean_cos': 0.0,
                'sin_vals': np.array([]), 'cos_vals': np.array([])}

    # Remove consecutive duplicates -- exact plugin code
    compare_against = points_array[:-1]
    duplicates_table = points_array[1:] == compare_against
    duplicates_table = duplicates_table.sum(axis=1)
    duplicate_indices = np.where(duplicates_table == 2)
    points_array = np.delete(points_array, duplicate_indices, axis=0)

    if len(points_array) < 3:
        return {'straightness': 0.0, 'mean_sin': 0.0, 'mean_cos': 0.0,
                'sin_vals': np.array([]), 'cos_vals': np.array([])}

    # Generate three sets of points
    points_set_a = points_array[:-2]
    points_set_b = points_array[1:-1]
    points_set_c = points_array[2:]

    # Generate two sets of vectors
    ab = points_set_b - points_set_a
    bc = points_set_c - points_set_b

    # Evaluate sin and cos values
    cross_products = np.cross(ab, bc)
    dot_products = np.einsum('ij,ij->i', ab, bc)
    product_magnitudes_ab_bc = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)

    # Avoid division by zero
    valid_magnitudes = product_magnitudes_ab_bc > 0
    cos_vals = np.zeros_like(product_magnitudes_ab_bc)
    sin_vals = np.zeros_like(product_magnitudes_ab_bc)

    cos_vals[valid_magnitudes] = dot_products[valid_magnitudes] / product_magnitudes_ab_bc[valid_magnitudes]
    sin_vals[valid_magnitudes] = cross_products[valid_magnitudes] / product_magnitudes_ab_bc[valid_magnitudes]

    cos_mean_val = float(np.mean(cos_vals))
    sin_mean_val = float(np.mean(sin_vals))

    return {
        'straightness': cos_mean_val,
        'mean_sin': sin_mean_val,
        'mean_cos': cos_mean_val,
        'sin_vals': sin_vals,
        'cos_vals': cos_vals,
    }


# ---------------------------------------------------------------------------
# Scaled Rg variants -- exact joinTracks.py::addLagDisplacementToDF columns
# ---------------------------------------------------------------------------

def radius_gyration_scaled(rg, mean_lag):
    """Rg scaled by mean lag displacement.

    Exact: tracksDF['radius_gyration_scaled'] = tracksDF['radius_gyration']/tracksDF['meanLag']
    """
    if mean_lag == 0 or np.isnan(mean_lag):
        return 0.0
    return rg / mean_lag


def radius_gyration_scaled_nSegments(rg, n_segments):
    """Rg scaled by number of segments.

    Exact: tracksDF['radius_gyration_scaled_nSegments'] = tracksDF['radius_gyration']/tracksDF['n_segments']
    """
    if n_segments == 0:
        return 0.0
    return rg / n_segments


def radius_gyration_scaled_trackLength(rg, track_length):
    """Rg scaled by track length.

    Exact: tracksDF['radius_gyration_scaled_trackLength'] = tracksDF['radius_gyration']/tracksDF['track_length']
    """
    if track_length == 0 or np.isnan(track_length):
        return 0.0
    return rg / track_length


# ---------------------------------------------------------------------------
# GeometricAnalyzer methods -- exact spt_batch_analysis::GeometricAnalyzer
# ---------------------------------------------------------------------------

def get_radius_of_gyration_simple(xy):
    """Calculate radius of gyration using simple geometric formula.

    Exact replica of spt_batch_analysis::GeometricAnalyzer.get_radius_of_gyration_simple.

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)

    Returns:
        Radius of gyration value (float), or NaN if < 2 points.
    """
    xy = np.asarray(xy, dtype=np.float64)
    # Remove NaN values
    xy = xy[~np.isnan(xy).any(axis=1)]

    if len(xy) < 2:
        return np.nan

    # Calculate average position
    avg = np.nanmean(xy, axis=0)

    # Calculate average squared position
    avg2 = np.nanmean(xy ** 2, axis=0)

    # Calculate radius of gyration
    rg = np.sqrt(np.sum(avg2 - avg ** 2))

    return float(rg)


def get_scaled_rg(rg, mean_step_length):
    """Calculate scaled radius of gyration as in Golan and Sherman Nat Comm 2017.

    Exact replica of spt_batch_analysis::GeometricAnalyzer.get_scaled_rg.

    sRg = sqrt(pi/2) * Rg / mean_step_length

    Args:
        rg: Radius of gyration
        mean_step_length: Mean step length

    Returns:
        Scaled radius of gyration value, or NaN if invalid.
    """
    if np.isnan(rg) or np.isnan(mean_step_length) or mean_step_length == 0:
        return np.nan

    s_rg = np.sqrt(np.pi / 2) * rg / mean_step_length
    return float(s_rg)


def classify_linear_motion_simple(xy, directionality_threshold=0.8, perpendicular_threshold=0.15):
    """Classify trajectory as linear or non-linear using geometric methods.

    Exact replica of spt_batch_analysis::GeometricAnalyzer.classify_linear_motion_simple.

    Uses:
    1. Directionality ratio: net displacement / total path length
    2. Mean perpendicular distance: average distance of points from straight line

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)
        directionality_threshold: Minimum directionality ratio for linear classification
        perpendicular_threshold: Maximum normalized perpendicular distance

    Returns:
        Dictionary with classification and metrics.
    """
    xy = np.asarray(xy, dtype=np.float64)
    # Remove NaN values
    xy = xy[~np.isnan(xy).any(axis=1)]

    if len(xy) < 3:
        return {
            'classification': 'unclassified',
            'directionality_ratio': np.nan,
            'mean_perpendicular_distance': np.nan,
            'normalized_perpendicular_distance': np.nan,
        }

    # Calculate directionality ratio
    start_point = xy[0]
    end_point = xy[-1]
    net_displacement = np.linalg.norm(end_point - start_point)

    # Calculate total path length
    steps = np.diff(xy, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    total_path_length = np.sum(step_lengths)

    if total_path_length == 0:
        return {
            'classification': 'unclassified',
            'directionality_ratio': np.nan,
            'mean_perpendicular_distance': np.nan,
            'normalized_perpendicular_distance': np.nan,
        }

    directionality_ratio = net_displacement / total_path_length

    # Calculate perpendicular distances from straight line
    if net_displacement > 0:
        direction = (end_point - start_point) / net_displacement

        perpendicular_distances = []
        for point in xy:
            vec_to_point = point - start_point
            projection_length = np.dot(vec_to_point, direction)
            projection = projection_length * direction
            perpendicular = vec_to_point - projection
            perpendicular_dist = np.linalg.norm(perpendicular)
            perpendicular_distances.append(perpendicular_dist)

        mean_perpendicular_distance = np.mean(perpendicular_distances)
        normalized_perpendicular_distance = (
            mean_perpendicular_distance / net_displacement
            if net_displacement > 0 else np.inf
        )
    else:
        mean_perpendicular_distance = np.nan
        normalized_perpendicular_distance = np.nan

    # Classify based on metrics
    is_directional = directionality_ratio >= directionality_threshold
    is_straight = (
        normalized_perpendicular_distance <= perpendicular_threshold
        if not np.isnan(normalized_perpendicular_distance) else False
    )

    if is_directional and is_straight:
        classification = 'linear_unidirectional'
    elif not is_directional and is_straight:
        classification = 'linear_bidirectional'
    else:
        classification = 'non_linear'

    return {
        'classification': classification,
        'directionality_ratio': float(directionality_ratio),
        'mean_perpendicular_distance': float(mean_perpendicular_distance) if not np.isnan(mean_perpendicular_distance) else np.nan,
        'normalized_perpendicular_distance': float(normalized_perpendicular_distance) if not np.isnan(normalized_perpendicular_distance) else np.nan,
    }


# ---------------------------------------------------------------------------
# Legacy alias for backward compatibility
# ---------------------------------------------------------------------------

def scaled_radius_of_gyration(positions, mean_step_length=None):
    """Radius of gyration normalized by mean step length (Golan & Sherman).

    sRg = sqrt(pi/2) * Rg / mean_step_length

    This is a convenience wrapper for get_scaled_rg().
    """
    rg_result = radius_of_gyration(positions)
    rg = rg_result['rg']

    if mean_step_length is None:
        steps = np.diff(np.asarray(positions, dtype=np.float64), axis=0)
        step_lengths = np.linalg.norm(steps, axis=1)
        mean_step_length = float(np.mean(step_lengths)) if len(step_lengths) > 0 else 0.0

    return get_scaled_rg(rg, mean_step_length)


def classify_linear_motion(positions, dir_thresh=0.8, perp_thresh=0.15):
    """Classify track as linear unidirectional, bidirectional, or non-linear.

    Convenience wrapper for classify_linear_motion_simple().

    Returns: 'linear_unidirectional', 'linear_bidirectional', or 'non_linear'
    """
    result = classify_linear_motion_simple(
        positions,
        directionality_threshold=dir_thresh,
        perpendicular_threshold=perp_thresh,
    )
    return result['classification']
