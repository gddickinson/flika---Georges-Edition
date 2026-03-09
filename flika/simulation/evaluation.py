# -*- coding: utf-8 -*-
"""Evaluation metrics for comparing analysis results against ground truth.

All functions are pure numpy (no Qt dependency) for standalone testing.
"""
import numpy as np
from dataclasses import dataclass, field


# -----------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""
    name: str
    category: str
    metrics: dict
    parameters: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def summary_line(self):
        """One-line summary for console output."""
        key_metric = next(iter(self.metrics), None)
        val = self.metrics.get(key_metric, 'N/A')
        if isinstance(val, float):
            val = f'{val:.3f}'
        return f"[{self.category}] {self.name}: {key_metric}={val}"

    def to_dict(self):
        """Serializable dict for JSON export."""
        d = {
            'name': self.name,
            'category': self.category,
            'parameters': self.parameters,
            'elapsed_seconds': self.elapsed_seconds,
            'metrics': {},
        }
        for k, v in self.metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                d['metrics'][k] = float(v)
            elif isinstance(v, np.ndarray):
                d['metrics'][k] = v.tolist()
            else:
                d['metrics'][k] = v
        return d


def format_benchmark_report(results):
    """Format a human-readable benchmark report.

    Parameters
    ----------
    results : list of BenchmarkResult

    Returns
    -------
    str
    """
    lines = ["=" * 70, "BENCHMARK REPORT", "=" * 70]
    by_cat = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat, items in sorted(by_cat.items()):
        lines.append(f"\n--- {cat.upper()} ---")
        for r in items:
            lines.append(f"  {r.name}  ({r.elapsed_seconds:.2f}s)")
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
    lines.append("=" * 70)
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Detection metrics
# -----------------------------------------------------------------------

def match_detections(gt_positions, detected_positions, max_distance=3.0):
    """Optimal matching between ground truth and detected positions.

    Uses Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Parameters
    ----------
    gt_positions : ndarray
        (N, 2) ground truth positions [y, x].
    detected_positions : ndarray
        (M, 2) detected positions [y, x].
    max_distance : float
        Maximum matching distance in pixels.

    Returns
    -------
    dict
        tp, fp, fn, precision, recall, f1, rmse, matched_gt, matched_det.
    """
    from scipy.optimize import linear_sum_assignment

    n_gt = len(gt_positions)
    n_det = len(detected_positions)

    if n_gt == 0 and n_det == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0,
                'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'rmse': 0.0, 'matched_gt': np.array([]),
                'matched_det': np.array([])}
    if n_gt == 0:
        return {'tp': 0, 'fp': n_det, 'fn': 0,
                'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
                'rmse': 0.0, 'matched_gt': np.array([]),
                'matched_det': np.array([])}
    if n_det == 0:
        return {'tp': 0, 'fp': 0, 'fn': n_gt,
                'precision': 1.0, 'recall': 0.0, 'f1': 0.0,
                'rmse': 0.0, 'matched_gt': np.array([]),
                'matched_det': np.array([])}

    # Cost matrix
    gt = np.asarray(gt_positions, dtype=float)
    det = np.asarray(detected_positions, dtype=float)
    diff = gt[:, np.newaxis, :] - det[np.newaxis, :, :]  # (N, M, 2)
    cost = np.sqrt((diff**2).sum(axis=2))  # (N, M)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Filter by max distance
    matched_gt = []
    matched_det = []
    distances = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_distance:
            matched_gt.append(r)
            matched_det.append(c)
            distances.append(cost[r, c])

    tp = len(matched_gt)
    fp = n_det - tp
    fn = n_gt - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    rmse = np.sqrt(np.mean(np.array(distances)**2)) if distances else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'rmse': rmse,
        'matched_gt': np.array(matched_gt),
        'matched_det': np.array(matched_det),
    }


def detection_metrics_stack(gt_table, det_table, max_distance=3.0):
    """Per-frame detection metrics over a full stack.

    Parameters
    ----------
    gt_table : ndarray
        (N, 3+) array [frame, y, x, ...].
    det_table : ndarray
        (M, 3+) array [frame, y, x, ...].
    max_distance : float
        Maximum matching distance.

    Returns
    -------
    dict
        Aggregate and per-frame metrics.
    """
    frames = set()
    if len(gt_table) > 0:
        frames.update(gt_table[:, 0].astype(int))
    if len(det_table) > 0:
        frames.update(det_table[:, 0].astype(int))

    total_tp = total_fp = total_fn = 0
    all_rmse = []
    per_frame = []

    for f in sorted(frames):
        gt_mask = gt_table[:, 0].astype(int) == f if len(gt_table) > 0 \
            else np.array([], dtype=bool)
        det_mask = det_table[:, 0].astype(int) == f if len(det_table) > 0 \
            else np.array([], dtype=bool)

        gt_pos = gt_table[gt_mask, 1:3] if np.any(gt_mask) \
            else np.empty((0, 2))
        det_pos = det_table[det_mask, 1:3] if np.any(det_mask) \
            else np.empty((0, 2))

        m = match_detections(gt_pos, det_pos, max_distance)
        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']
        if m['rmse'] > 0:
            all_rmse.extend([m['rmse']] * m['tp'])
        per_frame.append(m)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    rmse = np.mean(all_rmse) if all_rmse else 0.0

    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'rmse': rmse, 'n_frames': len(frames),
        'per_frame': per_frame,
    }


# -----------------------------------------------------------------------
# Tracking metrics (MOT-style)
# -----------------------------------------------------------------------

def tracking_metrics(gt_tracks, pred_tracks, max_distance=3.0):
    """Compute Multi-Object Tracking metrics.

    Parameters
    ----------
    gt_tracks : dict
        {id: (N, 3) array [frame, x, y]} ground truth tracks.
    pred_tracks : dict
        {id: (N, 3) array [frame, x, y]} predicted tracks.
    max_distance : float
        Maximum matching distance.

    Returns
    -------
    dict
        MOTA, MOTP, id_switches, mostly_tracked, mostly_lost,
        precision, recall.
    """
    # Build per-frame lookup tables
    gt_by_frame = {}
    for gid, arr in gt_tracks.items():
        for row in arr:
            f = int(row[0])
            gt_by_frame.setdefault(f, []).append((gid, row[1], row[2]))

    pred_by_frame = {}
    for pid, arr in pred_tracks.items():
        for row in arr:
            f = int(row[0])
            pred_by_frame.setdefault(f, []).append((pid, row[1], row[2]))

    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_matches = 0
    total_dist = 0.0
    id_switches = 0

    # Track gt_id -> last matched pred_id
    last_match = {}

    for f in all_frames:
        gt_list = gt_by_frame.get(f, [])
        pred_list = pred_by_frame.get(f, [])
        total_gt += len(gt_list)

        if not gt_list or not pred_list:
            total_fn += len(gt_list)
            total_fp += len(pred_list)
            continue

        # Build cost matrix
        gt_ids = [g[0] for g in gt_list]
        gt_pos = np.array([[g[1], g[2]] for g in gt_list])
        pred_ids = [p[0] for p in pred_list]
        pred_pos = np.array([[p[1], p[2]] for p in pred_list])

        diff = gt_pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :]
        cost = np.sqrt((diff**2).sum(axis=2))

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_this_frame = 0
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_distance:
                matched_this_frame += 1
                total_dist += cost[r, c]

                gid = gt_ids[r]
                pid = pred_ids[c]
                if gid in last_match and last_match[gid] != pid:
                    id_switches += 1
                last_match[gid] = pid

        total_matches += matched_this_frame
        fn_this = len(gt_list) - matched_this_frame
        fp_this = len(pred_list) - matched_this_frame
        total_fn += fn_this
        total_fp += fp_this

    motp = total_dist / max(total_matches, 1)
    mota = 1 - (total_fn + total_fp + id_switches) / max(total_gt, 1)

    # Track completeness
    mostly_tracked = 0
    mostly_lost = 0
    for gid, arr in gt_tracks.items():
        gt_frames = set(arr[:, 0].astype(int))
        matched_frames = 0
        for f in gt_frames:
            # Check if this gt was matched in this frame
            pred_list = pred_by_frame.get(f, [])
            if pred_list:
                matched_frames += 1  # approximate
        frac = matched_frames / max(len(gt_frames), 1)
        if frac >= 0.8:
            mostly_tracked += 1
        elif frac < 0.2:
            mostly_lost += 1

    precision = total_matches / max(total_matches + total_fp, 1)
    recall = total_matches / max(total_gt, 1)

    return {
        'MOTA': mota, 'MOTP': motp,
        'id_switches': id_switches,
        'mostly_tracked': mostly_tracked,
        'mostly_lost': mostly_lost,
        'precision': precision, 'recall': recall,
        'total_gt': total_gt, 'total_matches': total_matches,
    }


# -----------------------------------------------------------------------
# Segmentation metrics
# -----------------------------------------------------------------------

def segmentation_metrics(gt_mask, pred_mask):
    """Binary segmentation evaluation.

    Parameters
    ----------
    gt_mask : ndarray
        Ground truth binary mask.
    pred_mask : ndarray
        Predicted binary mask.

    Returns
    -------
    dict
        jaccard, dice, precision, recall, pixel_accuracy.
    """
    gt = (np.asarray(gt_mask) > 0).ravel()
    pred = (np.asarray(pred_mask) > 0).ravel()

    tp = np.sum(gt & pred)
    fp = np.sum(~gt & pred)
    fn = np.sum(gt & ~pred)
    tn = np.sum(~gt & ~pred)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    jaccard = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        'jaccard': float(jaccard),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'pixel_accuracy': float(accuracy),
    }


def instance_segmentation_metrics(gt_labels, pred_labels):
    """Instance segmentation via label matching.

    Parameters
    ----------
    gt_labels : ndarray
        Integer label mask (0 = background).
    pred_labels : ndarray
        Integer label mask (0 = background).

    Returns
    -------
    dict
        mean_iou, panoptic_quality, segmentation_quality,
        recognition_quality, over_segmentation, under_segmentation.
    """
    gt_ids = set(np.unique(gt_labels)) - {0}
    pred_ids = set(np.unique(pred_labels)) - {0}

    if not gt_ids and not pred_ids:
        return {'mean_iou': 1.0, 'panoptic_quality': 1.0,
                'segmentation_quality': 1.0, 'recognition_quality': 1.0,
                'over_segmentation': 0, 'under_segmentation': 0}

    # Compute IoU for all gt-pred pairs
    ious = {}
    for gid in gt_ids:
        gt_mask = gt_labels == gid
        for pid in pred_ids:
            pred_mask = pred_labels == pid
            intersection = np.sum(gt_mask & pred_mask)
            if intersection > 0:
                union = np.sum(gt_mask | pred_mask)
                ious[(gid, pid)] = intersection / union

    # Greedy matching by IoU > 0.5
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    for (gid, pid), iou in sorted(ious.items(), key=lambda x: -x[1]):
        if gid not in matched_gt and pid not in matched_pred and iou > 0.5:
            matched_gt.add(gid)
            matched_pred.add(pid)
            matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    sq = np.mean(matched_ious) if matched_ious else 0.0
    rq = 2 * tp / max(2 * tp + fp + fn, 1)
    pq = sq * rq

    return {
        'mean_iou': float(sq),
        'panoptic_quality': float(pq),
        'segmentation_quality': float(sq),
        'recognition_quality': float(rq),
        'over_segmentation': fp,
        'under_segmentation': fn,
    }


# -----------------------------------------------------------------------
# Calcium metrics
# -----------------------------------------------------------------------

def calcium_event_metrics(gt_spike_frames, detected_events,
                          tolerance_frames=3):
    """Compare ground truth and detected calcium events.

    Parameters
    ----------
    gt_spike_frames : array-like
        Frame indices of ground truth spikes.
    detected_events : list of dict
        Each dict must have 'peak' key (frame index).
    tolerance_frames : int
        Temporal tolerance for matching.

    Returns
    -------
    dict
        event_precision, event_recall, event_f1, timing_rmse.
    """
    gt_frames = np.asarray(gt_spike_frames, dtype=float)
    if len(detected_events) == 0:
        det_frames = np.array([])
    else:
        det_frames = np.array([e['peak'] for e in detected_events],
                              dtype=float)

    n_gt = len(gt_frames)
    n_det = len(det_frames)

    if n_gt == 0 and n_det == 0:
        return {'event_precision': 1.0, 'event_recall': 1.0,
                'event_f1': 1.0, 'timing_rmse': 0.0}
    if n_gt == 0:
        return {'event_precision': 0.0, 'event_recall': 1.0,
                'event_f1': 0.0, 'timing_rmse': 0.0}
    if n_det == 0:
        return {'event_precision': 1.0, 'event_recall': 0.0,
                'event_f1': 0.0, 'timing_rmse': 0.0}

    # Match by temporal proximity
    matched_gt = set()
    matched_det = set()
    timing_errors = []

    # Sort detections for greedy matching
    for di, df in enumerate(det_frames):
        dists = np.abs(gt_frames - df)
        best_gi = np.argmin(dists)
        if dists[best_gi] <= tolerance_frames and best_gi not in matched_gt:
            matched_gt.add(best_gi)
            matched_det.add(di)
            timing_errors.append(dists[best_gi])

    tp = len(matched_gt)
    fp = n_det - tp
    fn = n_gt - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    timing_rmse = float(np.sqrt(np.mean(np.array(timing_errors)**2))) \
        if timing_errors else 0.0

    return {
        'event_precision': precision, 'event_recall': recall,
        'event_f1': f1, 'timing_rmse': timing_rmse,
    }


def trace_accuracy(gt_trace, estimated_trace):
    """Compare ground truth and estimated fluorescence traces.

    Parameters
    ----------
    gt_trace : ndarray
        (T,) ground truth trace.
    estimated_trace : ndarray
        (T,) estimated trace.

    Returns
    -------
    dict
        correlation, rmse, snr.
    """
    gt = np.asarray(gt_trace, dtype=float)
    est = np.asarray(estimated_trace, dtype=float)
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]

    corr = float(np.corrcoef(gt, est)[0, 1]) if n > 1 else 0.0
    rmse = float(np.sqrt(np.mean((gt - est)**2)))
    signal = np.std(gt)
    noise = np.std(gt - est)
    snr = signal / max(noise, 1e-10)

    return {'correlation': corr, 'rmse': rmse, 'snr': snr}


# -----------------------------------------------------------------------
# Diffusion metrics
# -----------------------------------------------------------------------

def diffusion_coefficient_error(gt_D, estimated_D):
    """Compare ground truth and estimated diffusion coefficients.

    Parameters
    ----------
    gt_D : array-like
        Ground truth D values.
    estimated_D : array-like
        Estimated D values.

    Returns
    -------
    dict
        rmse, relative_error, correlation.
    """
    gt = np.asarray(gt_D, dtype=float).ravel()
    est = np.asarray(estimated_D, dtype=float).ravel()
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]

    rmse = float(np.sqrt(np.mean((gt - est)**2)))
    rel_err = float(np.mean(np.abs(gt - est) / np.maximum(gt, 1e-10)))
    corr = float(np.corrcoef(gt, est)[0, 1]) if n > 1 else 0.0

    return {'rmse': rmse, 'relative_error': rel_err, 'correlation': corr}
