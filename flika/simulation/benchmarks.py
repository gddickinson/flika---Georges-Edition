# -*- coding: utf-8 -*-
"""Predefined benchmark scenarios for simulation + analysis evaluation.

Each benchmark generates synthetic data, runs analysis, and evaluates
against ground truth. All benchmarks run without Qt/Window dependencies.
"""
import time
import numpy as np

from .engine import SimulationConfig, SimulationEngine
from .noise import CameraConfig
from .evaluation import (
    BenchmarkResult, match_detections, detection_metrics_stack,
    tracking_metrics, segmentation_metrics, calcium_event_metrics,
)
from .ground_truth import (
    positions_to_detection_table, trajectories_to_track_dict,
)

# Registry of benchmarks
BENCHMARKS = {}


def register_benchmark(name, category):
    """Decorator to register a benchmark function."""
    def decorator(func):
        BENCHMARKS[name] = {'func': func, 'category': category}
        return func
    return decorator


def run_benchmark(name, **kwargs):
    """Run a single named benchmark.

    Parameters
    ----------
    name : str
        Benchmark name.
    **kwargs
        Passed to the benchmark function.

    Returns
    -------
    BenchmarkResult
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{name}'. "
                         f"Available: {list(BENCHMARKS.keys())}")
    return BENCHMARKS[name]['func'](**kwargs)


def run_all_benchmarks(categories=None):
    """Run all registered benchmarks.

    Parameters
    ----------
    categories : list of str or None
        Filter by category. None = run all.

    Returns
    -------
    list of BenchmarkResult
    """
    results = []
    for name, info in BENCHMARKS.items():
        if categories and info['category'] not in categories:
            continue
        results.append(info['func']())
    return results


# -----------------------------------------------------------------------
# Detection benchmarks
# -----------------------------------------------------------------------

def _run_detection(config, detector_kwargs=None):
    """Helper: run simulation + detection + evaluation."""
    t0 = time.time()
    engine = SimulationEngine(config)
    stack, gt = engine.run()

    positions = gt.get('positions')
    trajectories = gt.get('trajectories')
    if positions is None:
        return None, None, None, time.time() - t0

    gt_table = positions_to_detection_table(positions, trajectories, config)

    # Run detection (UTrackDetector operates on z-first internal format)
    # The stack may have been transposed for flika display; undo for analysis
    analysis_stack = stack
    if analysis_stack.ndim == 4:
        # (T, Y, X, Z) or (1, Y, X, Z) -> take z=0 slice for 2D detection
        analysis_stack = analysis_stack[..., 0]
    if analysis_stack.ndim == 2:
        analysis_stack = analysis_stack[np.newaxis]

    from ..spt.detection.utrack_detector import UTrackDetector
    dk = detector_kwargs or {}
    detector = UTrackDetector(
        psf_sigma=dk.get('psf_sigma', 1.5),
        alpha=dk.get('alpha', 0.05),
    )
    det_locs = detector.detect_stack(analysis_stack.astype(float))

    # det_locs is (M, 4) [frame, x, y, intensity]
    # gt_table is (N, 4) [frame, y, x, particle_id]
    # Convert det_locs to [frame, y, x] format
    if len(det_locs) > 0:
        det_table = np.column_stack([
            det_locs[:, 0],  # frame
            det_locs[:, 2],  # y
            det_locs[:, 1],  # x
        ])
    else:
        det_table = np.empty((0, 3))

    metrics = detection_metrics_stack(gt_table[:, :3], det_table,
                                      max_distance=3.0)
    elapsed = time.time() - t0
    return metrics, gt_table, det_table, elapsed


@register_benchmark('detection_beads_easy', 'detection')
def benchmark_detection_beads_easy():
    """Well-separated bright beads, low noise."""
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=5,
        pixel_size=0.1, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 30, 'bead_radius': 0.1},
        fluorophore='Alexa488',
        camera=CameraConfig(type='sCMOS', read_noise=1.0,
                             quantum_efficiency=0.9),
        modality_params={'background': 5},
    )
    metrics, _, _, elapsed = _run_detection(config)
    return BenchmarkResult(
        name='detection_beads_easy', category='detection',
        metrics=metrics, elapsed_seconds=elapsed,
        parameters={'n_beads': 30, 'snr': 'high'},
    )


@register_benchmark('detection_beads_crowded', 'detection')
def benchmark_detection_beads_crowded():
    """Dense field with potential PSF overlaps."""
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=3,
        pixel_size=0.1, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 100, 'bead_radius': 0.1},
        fluorophore='Alexa488',
        camera=CameraConfig(type='sCMOS'),
        modality_params={'background': 10},
    )
    metrics, _, _, elapsed = _run_detection(config)
    return BenchmarkResult(
        name='detection_beads_crowded', category='detection',
        metrics=metrics, elapsed_seconds=elapsed,
        parameters={'n_beads': 100, 'snr': 'medium'},
    )


@register_benchmark('detection_low_snr', 'detection')
def benchmark_detection_low_snr():
    """Low photon count, high noise."""
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=3,
        pixel_size=0.1, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 20, 'bead_radius': 0.1},
        fluorophore='EGFP',  # lower photon count than Alexa
        camera=CameraConfig(type='sCMOS', read_noise=3.0,
                             quantum_efficiency=0.6),
        modality_params={'background': 20},
    )
    metrics, _, _, elapsed = _run_detection(config)
    return BenchmarkResult(
        name='detection_low_snr', category='detection',
        metrics=metrics, elapsed_seconds=elapsed,
        parameters={'n_beads': 20, 'snr': 'low'},
    )


# -----------------------------------------------------------------------
# Tracking benchmarks
# -----------------------------------------------------------------------

def _run_tracking(config, link_kwargs=None):
    """Helper: run simulation + detection + linking + evaluation."""
    t0 = time.time()
    engine = SimulationEngine(config)
    stack, gt = engine.run()

    positions = gt.get('positions')
    trajectories = gt.get('trajectories')
    if positions is None or trajectories is None:
        return None, time.time() - t0

    gt_tracks = trajectories_to_track_dict(positions, trajectories, config)

    # Detection
    analysis_stack = stack
    if analysis_stack.ndim == 4:
        analysis_stack = analysis_stack[..., 0]
    if analysis_stack.ndim == 2:
        analysis_stack = analysis_stack[np.newaxis]

    from ..spt.detection.utrack_detector import UTrackDetector
    detector = UTrackDetector(psf_sigma=1.5, alpha=0.05)
    det_locs = detector.detect_stack(analysis_stack.astype(float))

    if len(det_locs) == 0:
        metrics = {'MOTA': 0.0, 'MOTP': 0.0, 'id_switches': 0,
                   'precision': 0.0, 'recall': 0.0}
        return BenchmarkResult(
            name='tracking', category='tracking',
            metrics=metrics, elapsed_seconds=time.time() - t0)

    # Linking
    from ..spt.linking.greedy_linker import link_particles
    lk = link_kwargs or {}
    tracks_result, stats = link_particles(
        det_locs,
        max_distance=lk.get('max_distance', 5.0),
        max_gap=lk.get('max_gap', 2),
        min_track_length=lk.get('min_track_length', 3),
    )

    # Convert linked tracks to dict format {id: (N, 3) [frame, x, y]}
    pred_tracks = {}
    for tid, indices in enumerate(tracks_result):
        if len(indices) >= 2:
            track_data = det_locs[indices]  # (N, 4) [frame, x, y, int]
            pred_tracks[tid] = track_data[:, :3]  # [frame, x, y]

    metrics = tracking_metrics(gt_tracks, pred_tracks, max_distance=5.0)
    elapsed = time.time() - t0
    return metrics, elapsed


@register_benchmark('tracking_brownian_easy', 'tracking')
def benchmark_tracking_brownian():
    """Well-separated Brownian particles."""
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=100,
        pixel_size=0.1, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 20, 'bead_radius': 0.1},
        fluorophore='Alexa488',
        motion_type='brownian',
        motion_params={'D': 0.05},
        camera=CameraConfig(type='sCMOS', read_noise=1.0),
        modality_params={'background': 5},
    )
    metrics, elapsed = _run_tracking(config)
    return BenchmarkResult(
        name='tracking_brownian_easy', category='tracking',
        metrics=metrics, elapsed_seconds=elapsed,
        parameters={'n_particles': 20, 'D': 0.05},
    )


@register_benchmark('tracking_directed', 'tracking')
def benchmark_tracking_directed():
    """Directed motion with potential crossing."""
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=80,
        pixel_size=0.1, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 15, 'bead_radius': 0.1},
        fluorophore='Alexa488',
        motion_type='directed',
        motion_params={'D': 0.02, 'velocity': [0.3, 0.1]},
        camera=CameraConfig(type='sCMOS'),
        modality_params={'background': 5},
    )
    metrics, elapsed = _run_tracking(config)
    return BenchmarkResult(
        name='tracking_directed', category='tracking',
        metrics=metrics, elapsed_seconds=elapsed,
        parameters={'n_particles': 15, 'motion': 'directed'},
    )


# -----------------------------------------------------------------------
# Segmentation benchmarks
# -----------------------------------------------------------------------

@register_benchmark('segmentation_binary_beads', 'segmentation')
def benchmark_segmentation_binary():
    """Binary threshold on bright beads."""
    t0 = time.time()
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=1,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 30, 'bead_radius': 2},
        fluorophore='Alexa488',
        camera=CameraConfig(type='sCMOS'),
        modality_params={'background': 5},
    )
    engine = SimulationEngine(config)
    stack, gt = engine.run()

    gt_mask = gt['structure_mask'] > 0
    image = stack.astype(float)
    if image.ndim > 2:
        image = image.squeeze()

    # Otsu threshold
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(image)
    pred_mask = image > thresh

    metrics = segmentation_metrics(gt_mask, pred_mask)
    return BenchmarkResult(
        name='segmentation_binary_beads', category='segmentation',
        metrics=metrics, elapsed_seconds=time.time() - t0,
        parameters={'method': 'otsu'},
    )


@register_benchmark('segmentation_cells', 'segmentation')
def benchmark_segmentation_cells():
    """Cell field segmentation via threshold."""
    t0 = time.time()
    config = SimulationConfig(
        nx=128, ny=128, nz=1, nt=1,
        modality='widefield', psf_model='gaussian',
        structure_type='cell_field',
        structure_params={'n_cells': 8, 'cell_type': 'round'},
        fluorophore='EGFP',
        camera=CameraConfig(type='sCMOS'),
        modality_params={'background': 5},
    )
    engine = SimulationEngine(config)
    stack, gt = engine.run()

    gt_mask = gt['structure_mask'] > 0
    image = stack.astype(float)
    if image.ndim > 2:
        image = image.squeeze()

    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(image)
    pred_mask = image > thresh

    metrics = segmentation_metrics(gt_mask, pred_mask)
    return BenchmarkResult(
        name='segmentation_cells', category='segmentation',
        metrics=metrics, elapsed_seconds=time.time() - t0,
        parameters={'method': 'otsu', 'n_cells': 8},
    )


# -----------------------------------------------------------------------
# Calcium benchmarks
# -----------------------------------------------------------------------

@register_benchmark('calcium_spike_detection', 'calcium')
def benchmark_calcium_detection():
    """Calcium spike detection on simulated neuron."""
    t0 = time.time()
    config = SimulationConfig(
        nx=64, ny=64, nz=1, nt=500,
        pixel_size=0.5, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='cells',
        structure_params={'cell_type': 'round'},
        fluorophore='GCaMP6f',
        modality_params={'calcium_rate': 1.0, 'calcium_tau': 0.3,
                         'background': 5},
        camera=CameraConfig(type='sCMOS'),
    )
    engine = SimulationEngine(config)
    stack, gt = engine.run()

    calcium_trace = gt.get('calcium_trace')
    calcium_spikes = gt.get('calcium_spike_frames')

    if calcium_trace is None or calcium_spikes is None or len(calcium_spikes) == 0:
        return BenchmarkResult(
            name='calcium_spike_detection', category='calcium',
            metrics={'event_f1': 0.0, 'note': 'no GT spikes generated'},
            elapsed_seconds=time.time() - t0,
        )

    # Extract mean trace from image (ROI = whole cell)
    image_stack = stack.astype(float)
    if image_stack.ndim > 3:
        image_stack = image_stack.squeeze()
    if image_stack.ndim == 2:
        image_stack = image_stack[np.newaxis]

    # Use structure mask as ROI
    cell_mask = gt['structure_mask'] > 0
    if cell_mask.ndim > 2:
        cell_mask = cell_mask.squeeze()
    if cell_mask.sum() == 0:
        cell_mask = np.ones(image_stack.shape[-2:], dtype=bool)

    trace = np.array([
        image_stack[t][cell_mask].mean()
        for t in range(image_stack.shape[0])
    ])

    # Compute dF/F and detect events
    from ..process.calcium import compute_dff, detect_calcium_events
    dff = compute_dff(trace)
    events = detect_calcium_events(dff, threshold=2.0, min_duration=2)

    metrics = calcium_event_metrics(calcium_spikes, events,
                                    tolerance_frames=5)
    metrics['n_gt_spikes'] = len(calcium_spikes)
    metrics['n_detected'] = len(events)

    return BenchmarkResult(
        name='calcium_spike_detection', category='calcium',
        metrics=metrics, elapsed_seconds=time.time() - t0,
        parameters={'rate': 1.0, 'threshold': 2.0},
    )
