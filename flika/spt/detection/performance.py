"""Performance evaluation tools for localization microscopy.

Compares detected localizations against ground truth to compute
standard metrics used in SMLM benchmarks (e.g. ISBI SMLM Challenge):

- **Jaccard index**: TP / (TP + FP + FN) at a given distance threshold.
- **Recall** (sensitivity): TP / (TP + FN).
- **Precision**: TP / (TP + FP).
- **F1 score**: harmonic mean of precision and recall.
- **RMSE**: root mean square error of matched localizations.
- **Lateral / axial RMSE**: separate x,y and z accuracy.
- **Efficiency**: combined metric E = 100 · √(Jaccard · (1 - RMSE/threshold)).

References:
    Sage et al., Nat Methods 12:717 (2015) — SMLM Challenge.
    Sage et al., Nat Methods 16:387 (2019) — SMLM Challenge 2.
"""

import numpy as np
from scipy.spatial.distance import cdist


def match_localizations(detected, ground_truth, threshold=1.0,
                        use_3d=False):
    """Match detected localizations to ground truth using Hungarian algorithm.

    Parameters
    ----------
    detected : ndarray
        (N, 2+) array: [x, y, ...] or [x, y, z, ...].
    ground_truth : ndarray
        (M, 2+) array: [x, y, ...] or [x, y, z, ...].
    threshold : float
        Maximum matching distance (pixels).  Default 1.0.
    use_3d : bool
        If True, use 3D Euclidean distance (columns 0,1,2).

    Returns
    -------
    dict
        'tp_det_idx': indices into detected for true positives,
        'tp_gt_idx': indices into ground_truth for true positives,
        'fp_idx': indices into detected for false positives,
        'fn_idx': indices into ground_truth for false negatives,
        'distances': matched distances.
    """
    if len(detected) == 0 or len(ground_truth) == 0:
        return {
            'tp_det_idx': np.array([], dtype=int),
            'tp_gt_idx': np.array([], dtype=int),
            'fp_idx': np.arange(len(detected)),
            'fn_idx': np.arange(len(ground_truth)),
            'distances': np.array([]),
        }

    ndim = 3 if use_3d else 2
    det_coords = detected[:, :ndim]
    gt_coords = ground_truth[:, :ndim]

    # Compute distance matrix
    dist_matrix = cdist(det_coords, gt_coords)

    # Greedy matching (Hungarian is O(n³) which is slow for large N;
    # greedy with sorting is standard in SMLM challenges)
    tp_det = []
    tp_gt = []
    distances = []
    used_gt = set()
    used_det = set()

    # Sort all pairs by distance
    n_det, n_gt = dist_matrix.shape
    flat_idx = np.argsort(dist_matrix, axis=None)

    for flat in flat_idx:
        i = flat // n_gt
        j = flat % n_gt
        d = dist_matrix[i, j]
        if d > threshold:
            break
        if i in used_det or j in used_gt:
            continue
        tp_det.append(i)
        tp_gt.append(j)
        distances.append(d)
        used_det.add(i)
        used_gt.add(j)

    tp_det = np.array(tp_det, dtype=int)
    tp_gt = np.array(tp_gt, dtype=int)
    fp_idx = np.array([i for i in range(n_det) if i not in used_det],
                      dtype=int)
    fn_idx = np.array([j for j in range(n_gt) if j not in used_gt],
                      dtype=int)

    return {
        'tp_det_idx': tp_det,
        'tp_gt_idx': tp_gt,
        'fp_idx': fp_idx,
        'fn_idx': fn_idx,
        'distances': np.array(distances),
    }


def compute_metrics(detected, ground_truth, threshold=1.0, use_3d=False):
    """Compute full set of localization performance metrics.

    Parameters
    ----------
    detected : ndarray
        (N, 2+) array of detected positions.
    ground_truth : ndarray
        (M, 2+) array of true positions.
    threshold : float
        Matching distance threshold (pixels).
    use_3d : bool
        Use 3D matching.

    Returns
    -------
    dict
        'jaccard': float, 'precision': float, 'recall': float,
        'f1': float, 'rmse': float, 'rmse_lateral': float,
        'rmse_axial': float (if 3D), 'efficiency': float,
        'tp': int, 'fp': int, 'fn': int, 'n_detected': int,
        'n_ground_truth': int.
    """
    match = match_localizations(detected, ground_truth, threshold, use_3d)

    tp = len(match['tp_det_idx'])
    fp = len(match['fp_idx'])
    fn = len(match['fn_idx'])

    jaccard = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    if tp > 0:
        # Compute RMSE from matched pairs
        ndim = 3 if use_3d else 2
        det_matched = detected[match['tp_det_idx'], :ndim]
        gt_matched = ground_truth[match['tp_gt_idx'], :ndim]
        errors = det_matched - gt_matched

        rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        rmse_lateral = float(np.sqrt(np.mean(errors[:, :2] ** 2)))

        if use_3d and errors.shape[1] >= 3:
            rmse_axial = float(np.sqrt(np.mean(errors[:, 2] ** 2)))
        else:
            rmse_axial = 0.0
    else:
        rmse = float('inf')
        rmse_lateral = float('inf')
        rmse_axial = float('inf') if use_3d else 0.0

    # Efficiency (ISBI challenge metric)
    if rmse < threshold and jaccard > 0:
        efficiency = 100.0 * np.sqrt(jaccard * (1.0 - rmse / threshold))
    else:
        efficiency = 0.0

    return {
        'jaccard': jaccard,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rmse': rmse,
        'rmse_lateral': rmse_lateral,
        'rmse_axial': rmse_axial,
        'efficiency': efficiency,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_detected': len(detected),
        'n_ground_truth': len(ground_truth),
    }


def compute_metrics_per_frame(detected, ground_truth, threshold=1.0,
                              use_3d=False):
    """Compute metrics per frame for time-lapse data.

    Parameters
    ----------
    detected : ndarray
        (N, 3+) array: [frame, x, y, ...].
    ground_truth : ndarray
        (M, 3+) array: [frame, x, y, ...].
    threshold : float
        Matching distance.
    use_3d : bool
        Use 3D matching (columns 1,2,3).

    Returns
    -------
    dict
        'per_frame': list of per-frame metric dicts,
        'aggregate': dict with mean metrics across frames.
    """
    det_frames = detected[:, 0].astype(int) if len(detected) > 0 else np.array([])
    gt_frames = ground_truth[:, 0].astype(int) if len(ground_truth) > 0 else np.array([])

    all_frames = set()
    if len(det_frames) > 0:
        all_frames.update(det_frames.tolist())
    if len(gt_frames) > 0:
        all_frames.update(gt_frames.tolist())

    per_frame = []
    for f in sorted(all_frames):
        det_f = detected[det_frames == f, 1:] if len(detected) > 0 else np.empty((0, 2))
        gt_f = ground_truth[gt_frames == f, 1:] if len(ground_truth) > 0 else np.empty((0, 2))
        metrics = compute_metrics(det_f, gt_f, threshold, use_3d)
        metrics['frame'] = f
        per_frame.append(metrics)

    # Aggregate
    if per_frame:
        agg = {}
        for key in ['jaccard', 'precision', 'recall', 'f1', 'rmse',
                     'rmse_lateral', 'efficiency']:
            vals = [m[key] for m in per_frame if np.isfinite(m[key])]
            agg[key] = float(np.mean(vals)) if vals else 0.0
        agg['tp'] = sum(m['tp'] for m in per_frame)
        agg['fp'] = sum(m['fp'] for m in per_frame)
        agg['fn'] = sum(m['fn'] for m in per_frame)
    else:
        agg = {k: 0.0 for k in ['jaccard', 'precision', 'recall', 'f1',
                                  'rmse', 'rmse_lateral', 'efficiency']}
        agg.update({'tp': 0, 'fp': 0, 'fn': 0})

    return {'per_frame': per_frame, 'aggregate': agg}


def format_metrics_report(metrics):
    """Format metrics dict as a human-readable string.

    Parameters
    ----------
    metrics : dict
        From ``compute_metrics`` or ``compute_metrics_per_frame['aggregate']``.

    Returns
    -------
    str
        Formatted report.
    """
    lines = [
        "Localization Performance Metrics",
        "=" * 40,
        f"  True Positives:   {metrics.get('tp', 'N/A')}",
        f"  False Positives:  {metrics.get('fp', 'N/A')}",
        f"  False Negatives:  {metrics.get('fn', 'N/A')}",
        f"  Jaccard Index:    {metrics['jaccard']:.4f}",
        f"  Precision:        {metrics['precision']:.4f}",
        f"  Recall:           {metrics['recall']:.4f}",
        f"  F1 Score:         {metrics['f1']:.4f}",
        f"  RMSE:             {metrics['rmse']:.4f} px",
        f"  RMSE (lateral):   {metrics['rmse_lateral']:.4f} px",
    ]
    if metrics.get('rmse_axial', 0) > 0:
        lines.append(f"  RMSE (axial):     {metrics['rmse_axial']:.4f} nm")
    lines.append(f"  Efficiency:       {metrics['efficiency']:.1f}")
    return "\n".join(lines)
