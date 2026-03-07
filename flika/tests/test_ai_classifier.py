"""Tests for AI pixel classification and particle localization modules."""
import numpy as np
import pytest


# ---- FeatureConfig tests ----

def test_feature_config_defaults():
    from flika.ai.features import FeatureConfig
    cfg = FeatureConfig()
    assert cfg.include_intensity is True
    assert cfg.include_gaussian is True
    assert len(cfg.gaussian_sigmas) == 3


def test_feature_config_custom():
    from flika.ai.features import FeatureConfig
    cfg = FeatureConfig(include_gabor=False, include_extras=False)
    assert cfg.include_gabor is False
    assert cfg.include_extras is False


# ---- FeatureExtractor tests ----

def test_feature_extractor_shape():
    from flika.ai.features import FeatureExtractor
    fe = FeatureExtractor()
    img = np.random.rand(64, 64).astype(np.float32)
    features = fe.extract(img)
    assert features.ndim == 3
    assert features.shape[:2] == (64, 64)
    assert features.shape[2] == fe.n_features()


def test_feature_extractor_names_match():
    from flika.ai.features import FeatureExtractor
    fe = FeatureExtractor()
    names = fe.feature_names()
    assert len(names) == fe.n_features()
    assert 'intensity' in names
    assert 'sobel' in names


def test_feature_extractor_no_nans():
    from flika.ai.features import FeatureExtractor
    fe = FeatureExtractor()
    # Constant image — should not produce NaNs
    img = np.ones((32, 32), dtype=np.float32) * 5.0
    features = fe.extract(img)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_feature_extractor_subset():
    from flika.ai.features import FeatureExtractor, FeatureConfig
    cfg = FeatureConfig(
        include_intensity=True,
        include_gaussian=False,
        include_edges=False,
        include_lbp=False,
        include_hessian=False,
        include_gabor=False,
        include_extras=False,
    )
    fe = FeatureExtractor(cfg)
    assert fe.n_features() == 1
    img = np.random.rand(16, 16).astype(np.float32)
    features = fe.extract(img)
    assert features.shape == (16, 16, 1)


def test_feature_extractor_rejects_3d():
    from flika.ai.features import FeatureExtractor
    fe = FeatureExtractor()
    with pytest.raises(ValueError, match="2D"):
        fe.extract(np.zeros((10, 10, 10)))


# ---- ClassifierConfig tests ----

def test_classifier_config_defaults():
    from flika.ai.classifier_backends import ClassifierConfig
    cfg = ClassifierConfig()
    assert cfg.backend == 'random_forest'
    assert cfg.n_estimators == 100


# ---- RandomForestBackend tests ----

def test_rf_backend_train_predict():
    from flika.ai.classifier_backends import RandomForestBackend, ClassifierConfig

    cfg = ClassifierConfig(n_estimators=10, max_depth=5)
    backend = RandomForestBackend(cfg)
    assert not backend.is_trained()

    # Fake features: 2 classes, easily separable
    rng = np.random.default_rng(42)
    n = 200
    features = np.vstack([
        rng.normal(0, 1, (n, 5)),
        rng.normal(5, 1, (n, 5)),
    ]).astype(np.float32)
    labels = np.array([1]*n + [2]*n, dtype=np.int32)

    backend.train(features, labels, n_classes=2)
    assert backend.is_trained()

    pred_labels, pred_probs = backend.predict(features)
    assert pred_labels.shape == (2*n,)
    assert pred_probs.shape == (2*n, 2)
    # Should be mostly correct on training data
    accuracy = (pred_labels == labels).mean()
    assert accuracy > 0.9


def test_rf_backend_save_load(tmp_path):
    from flika.ai.classifier_backends import RandomForestBackend, ClassifierConfig

    cfg = ClassifierConfig(n_estimators=5, max_depth=3)
    backend = RandomForestBackend(cfg)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(50, 3)).astype(np.float32)
    labels = np.array([1]*25 + [2]*25, dtype=np.int32)
    backend.train(features, labels, 2)

    path = str(tmp_path / "model.joblib")
    backend.save(path)

    backend2 = RandomForestBackend(cfg)
    backend2.load(path)
    assert backend2.is_trained()

    l1, _ = backend.predict(features)
    l2, _ = backend2.predict(features)
    np.testing.assert_array_equal(l1, l2)


# ---- PSFSimulator tests ----

def test_psf_simulator_frame():
    from flika.ai.psf_simulator import PSFSimulator, PSFConfig
    cfg = PSFConfig(image_size=64, n_particles=10, psf_sigma=1.5)
    sim = PSFSimulator(cfg)
    image, positions = sim.generate_frame(rng=np.random.default_rng(42))
    assert image.shape == (64, 64)
    assert image.dtype == np.float32
    assert positions.shape == (10, 2)


def test_psf_simulator_stack():
    from flika.ai.psf_simulator import PSFSimulator, PSFConfig
    cfg = PSFConfig(image_size=32, n_particles=5, n_frames=3)
    sim = PSFSimulator(cfg)
    stack, all_pos = sim.generate_stack(rng=np.random.default_rng(0))
    assert stack.shape == (3, 32, 32)
    assert len(all_pos) == 3
    for pos in all_pos:
        assert pos.shape == (5, 2)


def test_psf_density_map():
    from flika.ai.psf_simulator import PSFSimulator, PSFConfig
    cfg = PSFConfig(image_size=64, n_particles=5, psf_sigma=2.0)
    sim = PSFSimulator(cfg)
    positions = np.array([[20, 30], [40, 50]], dtype=np.float64)
    density = sim.positions_to_density_map(positions, (64, 64))
    assert density.shape == (64, 64)
    assert density.dtype == np.float32
    assert density.max() <= 1.0
    assert density.max() > 0.0


def test_coordinate_extraction_roundtrip():
    from flika.ai.psf_simulator import PSFSimulator, PSFConfig
    cfg = PSFConfig(image_size=128, psf_sigma=2.0)
    sim = PSFSimulator(cfg)

    # Place particles at known positions
    true_positions = np.array([[30.0, 40.0], [80.0, 90.0], [50.0, 60.0]])
    density = sim.positions_to_density_map(true_positions, (128, 128))

    detected = PSFSimulator.extract_coordinates(density, threshold=0.1, min_distance=5)
    assert len(detected) == 3

    # Check each true position has a detection nearby (within 3 pixels)
    for ty, tx in true_positions:
        dists = np.sqrt((detected[:, 0] - ty)**2 + (detected[:, 1] - tx)**2)
        assert dists.min() < 3.0, f"No detection near ({ty}, {tx})"


def test_coordinate_extraction_empty():
    from flika.ai.psf_simulator import PSFSimulator
    # All zeros — no peaks
    density = np.zeros((32, 32), dtype=np.float32)
    coords = PSFSimulator.extract_coordinates(density, threshold=0.1)
    assert coords.shape == (0, 2)


# ---- LocalizerConfig tests ----

def test_localizer_config_defaults():
    from flika.ai.localizer_backends import LocalizerConfig
    cfg = LocalizerConfig()
    assert cfg.backend == 'deepstorm'
    assert cfg.epochs == 100


# ---- create_backend factory tests ----

def test_classifier_create_backend_rf():
    from flika.ai.classifier_backends import create_backend, ClassifierConfig
    cfg = ClassifierConfig(backend='random_forest')
    backend = create_backend(cfg)
    assert backend.name() == 'Random Forest'


def test_classifier_create_backend_invalid():
    from flika.ai.classifier_backends import create_backend, ClassifierConfig
    cfg = ClassifierConfig(backend='nonexistent')
    with pytest.raises(ValueError):
        create_backend(cfg)


def test_localizer_create_backend_invalid():
    from flika.ai.localizer_backends import create_backend, LocalizerConfig
    cfg = LocalizerConfig(backend='nonexistent')
    with pytest.raises(ValueError):
        create_backend(cfg)


# ---- Noise types test ----

def test_psf_noise_types():
    from flika.ai.psf_simulator import PSFSimulator, PSFConfig
    rng = np.random.default_rng(42)
    for noise in ('poisson', 'gaussian', 'mixed'):
        cfg = PSFConfig(image_size=32, n_particles=3, noise_type=noise)
        sim = PSFSimulator(cfg)
        image, _ = sim.generate_frame(rng=rng)
        assert image.shape == (32, 32)
        assert np.isfinite(image).all()
