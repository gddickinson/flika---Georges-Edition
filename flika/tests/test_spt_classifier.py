"""Comprehensive tests for SPT SVM classifier.

Tests the full train/predict/save/load lifecycle, feature extraction,
error handling, and edge cases.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os


def _make_feature_df(n_samples=60, n_classes=3, seed=42):
    """Generate synthetic feature data with known labels."""
    rng = np.random.RandomState(seed)
    dfs = []
    labels = []
    for cls in range(1, n_classes + 1):
        per_class = n_samples // n_classes
        data = {
            'net_displacement': rng.normal(cls * 5, 1.0, per_class),
            'straightness': rng.uniform(0.1 * cls, 0.3 * cls, per_class),
            'asymmetry': rng.uniform(0.0, 0.5, per_class),
            'radius_gyration': rng.normal(cls * 2, 0.5, per_class),
            'kurtosis': rng.normal(0, 1, per_class),
            'fractal_dimension': rng.uniform(1.0, 2.0, per_class),
        }
        dfs.append(pd.DataFrame(data))
        labels.extend([cls] * per_class)
    return pd.concat(dfs, ignore_index=True), np.array(labels)


class TestSVMClassifierTrainPredict:
    def test_train_returns_metrics(self):
        """Training should return a dict with accuracy and metrics."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        metrics = clf.train(df, labels, test_size=0.2, random_state=42)

        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 'per_class_accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'best_params' in metrics
        assert 'n_components' in metrics
        assert metrics['n_train'] > 0
        assert metrics['n_test'] > 0
        assert clf.is_trained

    def test_predict_after_train(self):
        """Predict should return integer labels after training."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        clf.train(df, labels)

        # Predict on a subset
        preds = clf.predict(df.iloc[:5])
        assert len(preds) == 5
        assert all(p in [1, 2, 3] for p in preds)

    def test_predict_empty_features(self):
        """Predict on empty DataFrame should return empty array."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        clf.train(df, labels)

        empty_df = pd.DataFrame(columns=clf.FEATURES)
        preds = clf.predict(empty_df)
        assert len(preds) == 0

    def test_train_mismatched_lengths_raises(self):
        """Training with mismatched features/labels should raise ValueError."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, _ = _make_feature_df(n_samples=60)
        labels = np.array([1, 2, 3])  # wrong length
        with pytest.raises(ValueError, match="rows"):
            clf.train(df, labels)

    def test_predict_missing_columns_raises(self):
        """Predict with missing feature columns should raise ValueError."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        clf.train(df, labels)

        bad_df = pd.DataFrame({'net_displacement': [1.0], 'straightness': [0.5]})
        with pytest.raises(ValueError, match="Missing"):
            clf.predict(bad_df)


class TestSVMClassifierSaveLoad:
    def test_save_load_roundtrip(self):
        """Saved model should be loadable and produce same predictions."""
        sklearn = pytest.importorskip('sklearn')
        pytest.importorskip('joblib')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        clf.train(df, labels)

        test_data = df.iloc[:5]
        preds_before = clf.predict(test_data)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            clf.save(path)
            assert os.path.exists(path)

            clf2 = SPTClassifier()
            assert not clf2.is_trained
            clf2.load(path)
            assert clf2.is_trained

            preds_after = clf2.predict(test_data)
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            os.unlink(path)

    def test_load_nonexistent_raises(self):
        """Loading from nonexistent path should raise FileNotFoundError."""
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load('/nonexistent/model.pkl')


class TestSVMClassifierFeatureExtraction:
    def test_extract_features_handles_nan(self):
        """Feature extraction should handle NaN by replacing with median."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        clf.train(df, labels)

        # Insert NaN and Inf
        test_df = df.iloc[:3].copy()
        test_df.iloc[0, 0] = np.nan
        test_df.iloc[1, 1] = np.inf

        # Should not crash
        preds = clf.predict(test_df)
        assert len(preds) == 3

    def test_predict_proba_untrained_raises(self):
        """predict_proba on untrained classifier should raise RuntimeError."""
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, _ = _make_feature_df(n_samples=60)
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict_proba(df)

    def test_predict_proba_returns_none_or_array(self):
        """predict_proba should return None or array (SVM trained without proba)."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60)
        clf.train(df, labels)

        result = clf.predict_proba(df.iloc[:5])
        # Without probability=True in SVM, it returns None
        assert result is None


class TestSVMClassifierEdgeCases:
    def test_two_class_training(self):
        """Training with only 2 classes should work."""
        sklearn = pytest.importorskip('sklearn')
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        df, labels = _make_feature_df(n_samples=60, n_classes=3)
        # Keep only classes 1 and 2
        mask = labels <= 2
        clf.train(df[mask], labels[mask])
        assert clf.is_trained

    def test_label_name_unknown(self):
        """label_name for unknown integer should return Unknown(...)."""
        from flika.spt.classification.svm_classifier import SPTClassifier

        clf = SPTClassifier()
        assert clf.label_name(99) == 'Unknown(99)'
        assert clf.label_name(0) == 'Unknown(0)'
