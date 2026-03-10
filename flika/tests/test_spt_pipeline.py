"""Integration tests for SPT pipeline."""
import pytest
import numpy as np


class TestSPTBatchPipeline:
    def test_params_serialization(self):
        """SPTParams should round-trip through dict serialization."""
        from flika.spt.batch.batch_pipeline import SPTParams

        p = SPTParams(detector='utrack', linker='greedy', max_distance=5)
        d = p.to_dict()
        p2 = SPTParams.from_dict(d)
        assert p2.detector == 'utrack'
        assert p2.max_distance == 5

    def test_params_defaults(self):
        """SPTParams defaults should be sensible."""
        from flika.spt.batch.batch_pipeline import SPTParams

        p = SPTParams()
        assert p.detector == 'utrack'
        assert p.linker == 'greedy'
        assert p.psf_sigma == 1.5
        assert p.pixel_size == 108.0
        assert p.enable_features is True

    def test_params_repr(self):
        """SPTParams repr should be a string."""
        from flika.spt.batch.batch_pipeline import SPTParams

        p = SPTParams()
        r = repr(p)
        assert 'SPTParams' in r
        assert 'detector' in r

    def test_detect_synthetic(self):
        """Detection on synthetic data should find particles."""
        from flika.spt.batch.batch_pipeline import SPTBatchPipeline, SPTParams

        # Create synthetic data: 5 frames, 2 moving particles
        stack = np.random.normal(100, 10, (5, 60, 60)).astype(np.float64)
        for f in range(5):
            x1, y1 = 20 + f, 20
            x2, y2 = 40, 20 + f
            yy, xx = np.mgrid[-3:4, -3:4]
            for cx, cy in [(x1, y1), (x2, y2)]:
                spot = 500 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
                stack[f, cy - 3:cy + 4, cx - 3:cx + 4] += spot

        params = SPTParams(detector='utrack', linker='greedy',
                           max_distance=5, psf_sigma=1.5, alpha=0.1,
                           min_track_length=2)
        pipeline = SPTBatchPipeline(params)
        result = pipeline._detect(stack)
        assert len(result) > 0  # should find particles

    def test_detect_and_link_synthetic(self):
        """Full detect-and-link pipeline on synthetic data."""
        from flika.spt.batch.batch_pipeline import SPTBatchPipeline, SPTParams

        stack = np.random.normal(100, 10, (5, 60, 60)).astype(np.float64)
        for f in range(5):
            x1, y1 = 20 + f, 20
            x2, y2 = 40, 20 + f
            yy, xx = np.mgrid[-3:4, -3:4]
            for cx, cy in [(x1, y1), (x2, y2)]:
                spot = 500 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
                stack[f, cy - 3:cy + 4, cx - 3:cx + 4] += spot

        params = SPTParams(detector='utrack', linker='greedy',
                           max_distance=5, psf_sigma=1.5, alpha=0.1,
                           min_track_length=2)
        pipeline = SPTBatchPipeline(params)
        result = pipeline._detect(stack)

        tracks, stats = pipeline._link(result)
        assert stats['num_tracks'] >= 1

    def test_detect_thunderstorm(self):
        """ThunderSTORM detector should work through the pipeline."""
        from flika.spt.batch.batch_pipeline import SPTBatchPipeline, SPTParams

        stack = np.random.normal(100, 10, (3, 60, 60)).astype(np.float64)
        for f in range(3):
            yy, xx = np.mgrid[-3:4, -3:4]
            spot = 500 * np.exp(-(xx**2 + yy**2) / (2 * 1.5**2))
            stack[f, 30 - 3:30 + 4, 30 - 3:30 + 4] += spot

        params = SPTParams(detector='thunderstorm', threshold=2.0,
                           psf_sigma=1.5)
        pipeline = SPTBatchPipeline(params)
        result = pipeline._detect(stack)
        assert result.ndim == 2

    def test_pipeline_default_params(self):
        """Pipeline should accept default params (None)."""
        from flika.spt.batch.batch_pipeline import SPTBatchPipeline

        pipeline = SPTBatchPipeline(params=None)
        assert pipeline.params.detector == 'utrack'

    def test_compute_features(self):
        """Feature computation through the pipeline should work."""
        from flika.spt.batch.batch_pipeline import SPTBatchPipeline, SPTParams

        params = SPTParams()
        pipeline = SPTBatchPipeline(params)

        # Create localizations with known tracks
        locs = np.zeros((15, 3))
        for i in range(15):
            locs[i] = [i % 5, 10 + i * 0.5, 20 + i * 0.3]
        tracks = [list(range(5)), list(range(5, 10)), list(range(10, 15))]

        df = pipeline._compute_features(locs, tracks)
        assert len(df) == 3
        assert 'track_id' in df.columns


class TestExpertConfigs:
    def test_list_configs(self):
        """list_configs should return all known profile names."""
        from flika.spt.batch.expert_configs import list_configs

        configs = list_configs()
        assert 'fast_membrane_proteins' in configs
        assert 'vesicle_trafficking' in configs
        assert 'slow_confined_proteins' in configs
        assert 'single_molecule' in configs
        assert 'viral_particles' in configs
        assert 'motor_proteins' in configs

    def test_get_config(self):
        """get_config should return an SPTParams with correct values."""
        from flika.spt.batch.expert_configs import get_config

        params = get_config('fast_membrane_proteins')
        assert params.max_distance == 3.0
        assert params.max_gap == 36

    def test_get_config_returns_copy(self):
        """get_config should return a copy, not the original."""
        from flika.spt.batch.expert_configs import get_config

        p1 = get_config('fast_membrane_proteins')
        p2 = get_config('fast_membrane_proteins')
        p1.max_distance = 999
        assert p2.max_distance == 3.0  # should not be affected

    def test_get_config_unknown_raises(self):
        """Unknown config name should raise KeyError."""
        from flika.spt.batch.expert_configs import get_config

        with pytest.raises(KeyError, match="Unknown expert config"):
            get_config('nonexistent_config')

    def test_describe_config(self):
        """describe_config should return a multi-line string."""
        from flika.spt.batch.expert_configs import describe_config

        desc = describe_config('vesicle_trafficking')
        assert 'vesicle_trafficking' in desc
        assert 'max_distance' in desc

    def test_get_config_for_experiment(self):
        """get_config_for_experiment should return a name and params."""
        from flika.spt.batch.expert_configs import get_config_for_experiment

        name, params = get_config_for_experiment(particle_type='vesicle')
        assert name == 'vesicle_trafficking'
        assert params.linker == 'utrack_lap'


class TestClassifier:
    def test_classifier_features(self):
        """SPTClassifier should have expected class attributes."""
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        assert 'radius_gyration' in clf.FEATURES
        assert 'net_displacement' in clf.FEATURES
        assert clf.LABELS[1] == 'Mobile'
        assert clf.LABELS[2] == 'Confined'
        assert clf.LABELS[3] == 'Trapped'
        assert not clf.is_trained

    def test_classifier_label_name(self):
        """label_name should return human-readable label strings."""
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        assert clf.label_name(1) == 'Mobile'
        assert clf.label_name(2) == 'Confined'
        assert clf.label_name(3) == 'Trapped'
        assert 'Unknown' in clf.label_name(99)

    def test_predict_untrained_raises(self):
        """Predicting with an untrained classifier should raise RuntimeError."""
        from flika.spt.classification.svm_classifier import SPTClassifier
        import pandas as pd

        clf = SPTClassifier()
        df = pd.DataFrame({
            'net_displacement': [1.0],
            'straightness': [0.5],
            'asymmetry': [0.1],
            'radius_gyration': [2.0],
            'kurtosis': [0.0],
            'fractal_dimension': [1.2],
        })
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(df)

    def test_save_untrained_raises(self):
        """Saving an untrained classifier should raise RuntimeError."""
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.save('/tmp/test_model.pkl')
