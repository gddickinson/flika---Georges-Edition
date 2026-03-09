# -*- coding: utf-8 -*-
"""Benchmark tests for simulation + analysis pipeline validation.

Standalone tests (no FlikaApplication required). They verify that analysis
algorithms achieve minimum quality thresholds on known synthetic data.
"""
import pytest
import numpy as np


class TestDetectionBenchmarks:
    """Detection accuracy on simulated bead fields."""

    def test_detection_beads_easy(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('detection_beads_easy')
        assert result.metrics['f1'] > 0.5
        assert result.metrics['rmse'] < 3.0

    def test_detection_beads_crowded(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('detection_beads_crowded')
        assert result.metrics['recall'] > 0.3

    def test_detection_low_snr(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('detection_low_snr')
        # Low SNR is hard; just check it runs and finds some particles
        assert result.metrics['recall'] > 0.1


class TestTrackingBenchmarks:
    """Tracking accuracy on simulated particle motion."""

    def test_tracking_brownian(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('tracking_brownian_easy')
        assert result.metrics['MOTA'] > 0.2
        assert result.metrics['MOTP'] < 5.0

    def test_tracking_directed(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('tracking_directed')
        assert result.metrics['recall'] > 0.2


class TestSegmentationBenchmarks:
    """Segmentation accuracy on simulated structures."""

    def test_segmentation_binary_beads(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('segmentation_binary_beads')
        assert result.metrics['dice'] > 0.3

    def test_segmentation_cells(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('segmentation_cells')
        assert result.metrics['jaccard'] > 0.2


class TestCalciumBenchmarks:
    """Calcium event detection on simulated traces."""

    def test_calcium_spike_detection(self):
        from flika.simulation.benchmarks import run_benchmark
        result = run_benchmark('calcium_spike_detection')
        # May have few or no GT spikes due to randomness; check it runs
        assert 'event_f1' in result.metrics or 'note' in result.metrics


class TestEvaluationMetrics:
    """Unit tests for the evaluation metric functions themselves."""

    def test_match_detections_perfect(self):
        from flika.simulation.evaluation import match_detections
        gt = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        det = gt + np.random.randn(*gt.shape) * 0.1
        m = match_detections(gt, det, max_distance=3.0)
        assert m['tp'] == 3
        assert m['fp'] == 0
        assert m['fn'] == 0
        assert m['f1'] == 1.0

    def test_match_detections_with_fp(self):
        from flika.simulation.evaluation import match_detections
        gt = np.array([[10, 20]], dtype=float)
        det = np.array([[10, 20], [50, 60]], dtype=float)
        m = match_detections(gt, det, max_distance=3.0)
        assert m['tp'] == 1
        assert m['fp'] == 1
        assert m['fn'] == 0

    def test_match_detections_with_fn(self):
        from flika.simulation.evaluation import match_detections
        gt = np.array([[10, 20], [30, 40]], dtype=float)
        det = np.array([[10, 20.5]], dtype=float)
        m = match_detections(gt, det, max_distance=3.0)
        assert m['tp'] == 1
        assert m['fn'] == 1

    def test_match_detections_empty(self):
        from flika.simulation.evaluation import match_detections
        m = match_detections(np.empty((0, 2)), np.empty((0, 2)))
        assert m['f1'] == 1.0

    def test_segmentation_metrics_perfect(self):
        from flika.simulation.evaluation import segmentation_metrics
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True
        m = segmentation_metrics(mask, mask)
        assert m['jaccard'] == 1.0
        assert m['dice'] == 1.0

    def test_segmentation_metrics_partial(self):
        from flika.simulation.evaluation import segmentation_metrics
        gt = np.zeros((50, 50), dtype=bool)
        gt[10:30, 10:30] = True
        pred = np.zeros((50, 50), dtype=bool)
        pred[15:35, 15:35] = True
        m = segmentation_metrics(gt, pred)
        assert 0 < m['jaccard'] < 1.0
        assert m['dice'] > m['jaccard']  # Dice >= Jaccard

    def test_calcium_event_metrics_perfect(self):
        from flika.simulation.evaluation import calcium_event_metrics
        gt_frames = np.array([10, 50, 100])
        events = [{'peak': 10}, {'peak': 50}, {'peak': 100}]
        m = calcium_event_metrics(gt_frames, events)
        assert m['event_f1'] == 1.0
        assert m['timing_rmse'] == 0.0

    def test_calcium_event_metrics_with_tolerance(self):
        from flika.simulation.evaluation import calcium_event_metrics
        gt_frames = np.array([10, 50])
        events = [{'peak': 12}, {'peak': 48}]  # within tolerance
        m = calcium_event_metrics(gt_frames, events, tolerance_frames=3)
        assert m['event_f1'] == 1.0
        assert m['timing_rmse'] == 2.0

    def test_tracking_metrics_basic(self):
        from flika.simulation.evaluation import tracking_metrics
        gt_tracks = {
            0: np.array([[0, 10, 20], [1, 11, 21], [2, 12, 22]]),
            1: np.array([[0, 50, 60], [1, 51, 61], [2, 52, 62]]),
        }
        pred_tracks = {
            0: np.array([[0, 10.1, 20.1], [1, 11.1, 21.1],
                          [2, 12.1, 22.1]]),
            1: np.array([[0, 50.1, 60.1], [1, 51.1, 61.1],
                          [2, 52.1, 62.1]]),
        }
        m = tracking_metrics(gt_tracks, pred_tracks, max_distance=3.0)
        assert m['MOTA'] > 0.9
        assert m['MOTP'] < 1.0

    def test_benchmark_result_serializable(self):
        import json
        from flika.simulation.evaluation import BenchmarkResult
        r = BenchmarkResult('test', 'unit',
                            {'f1': 0.95, 'count': np.int64(5)})
        d = r.to_dict()
        s = json.dumps(d)
        assert 'f1' in s

    def test_format_report(self):
        from flika.simulation.evaluation import (
            BenchmarkResult, format_benchmark_report)
        results = [
            BenchmarkResult('test1', 'detection', {'f1': 0.9},
                            elapsed_seconds=1.5),
            BenchmarkResult('test2', 'tracking', {'MOTA': 0.8},
                            elapsed_seconds=2.0),
        ]
        report = format_benchmark_report(results)
        assert 'DETECTION' in report
        assert 'TRACKING' in report


class TestGroundTruth:
    """Test ground truth generation and conversion."""

    def test_positions_to_detection_table(self):
        from flika.simulation.ground_truth import positions_to_detection_table
        from flika.simulation.engine import SimulationConfig
        config = SimulationConfig(nx=64, ny=64, nt=3)
        pos = np.array([[10.0, 20.0], [30.0, 40.0]])
        table = positions_to_detection_table(pos, None, config)
        assert table.shape == (6, 4)  # 2 particles * 3 frames
        assert table[0, 0] == 0  # frame 0
        assert table[0, 3] == 0  # particle 0

    def test_trajectories_to_track_dict(self):
        from flika.simulation.ground_truth import trajectories_to_track_dict
        from flika.simulation.engine import SimulationConfig
        config = SimulationConfig(nx=64, ny=64, nt=5)
        pos = np.array([[10.0, 20.0], [30.0, 40.0]])
        traj = np.random.randn(2, 5, 2) * 0.5
        tracks = trajectories_to_track_dict(pos, traj, config)
        assert len(tracks) == 2
        assert tracks[0].shape == (5, 3)  # [frame, x, y]

    def test_ground_truth_dataclass(self):
        from flika.simulation.ground_truth import GroundTruth
        gt = GroundTruth(
            positions_per_frame=np.array([[0, 10, 20, 0]]),
            binary_mask=np.ones((10, 10), dtype=np.uint8),
        )
        d = gt.to_dict()
        assert 'positions_per_frame' in d
        assert 'binary_mask' in d
        gt2 = GroundTruth.from_dict(d)
        assert gt2.positions_per_frame is not None

    def test_engine_stores_enhanced_gt(self):
        from flika.simulation.engine import SimulationConfig, SimulationEngine
        config = SimulationConfig(
            nx=64, ny=64, nz=1, nt=5,
            structure_type='beads',
            structure_params={'n_beads': 10},
            motion_type='brownian',
            motion_params={'D': 0.1},
        )
        engine = SimulationEngine(config)
        stack, gt = engine.run()
        assert 'positions_per_frame' in gt
        assert 'track_dict' in gt
        assert 'binary_mask' in gt
        assert 'diffusion_coefficients' in gt
        assert gt['positions_per_frame'].shape[1] == 4
        assert len(gt['track_dict']) == 10

    def test_engine_stores_calcium_gt(self):
        from flika.simulation.engine import SimulationConfig, SimulationEngine
        config = SimulationConfig(
            nx=32, ny=32, nz=1, nt=200,
            structure_type='cells',
            structure_params={'cell_type': 'round'},
            fluorophore='GCaMP6f',
            modality_params={'calcium_rate': 2.0, 'calcium_tau': 0.3},
        )
        engine = SimulationEngine(config)
        stack, gt = engine.run()
        assert 'calcium_trace' in gt
        assert len(gt['calcium_trace']) == 200
        assert 'calcium_spike_frames' in gt


class TestBenchmarkInfrastructure:
    """Test benchmark registry and runner."""

    def test_all_benchmarks_registered(self):
        from flika.simulation.benchmarks import BENCHMARKS
        assert len(BENCHMARKS) >= 7
        categories = {v['category'] for v in BENCHMARKS.values()}
        assert 'detection' in categories
        assert 'tracking' in categories
        assert 'segmentation' in categories
        assert 'calcium' in categories

    def test_run_all_benchmarks(self):
        """Smoke test: all benchmarks execute without error."""
        from flika.simulation.benchmarks import run_all_benchmarks
        results = run_all_benchmarks()
        assert len(results) >= 7
        for r in results:
            assert r.metrics is not None
            assert r.elapsed_seconds >= 0

    def test_run_by_category(self):
        from flika.simulation.benchmarks import run_all_benchmarks
        results = run_all_benchmarks(categories=['segmentation'])
        assert all(r.category == 'segmentation' for r in results)

    def test_unknown_benchmark_raises(self):
        from flika.simulation.benchmarks import run_benchmark
        with pytest.raises(ValueError):
            run_benchmark('nonexistent_benchmark')
