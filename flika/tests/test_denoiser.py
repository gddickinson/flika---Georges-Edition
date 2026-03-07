"""Tests for AI denoiser backend abstraction and configuration."""
import numpy as np
import pytest


# ---- Backend detection ----

def test_detect_backend_returns_valid_string():
    from flika.ai.denoiser import detect_denoiser_backend, _cached_backend
    import flika.ai.denoiser as mod
    # Clear cache for a clean test
    mod._cached_backend = None
    result = detect_denoiser_backend()
    assert result in ('careamics', 'n2v', 'csbdeep', 'none')
    # Second call should return cached value
    result2 = detect_denoiser_backend()
    assert result2 == result
    # Reset cache
    mod._cached_backend = None


def test_detect_backend_caching():
    import flika.ai.denoiser as mod
    mod._cached_backend = None
    r1 = mod.detect_denoiser_backend()
    # Force a different value to verify caching
    mod._cached_backend = 'test_cached'
    r2 = mod.detect_denoiser_backend()
    assert r2 == 'test_cached'
    mod._cached_backend = None


# ---- DenoiserConfig ----

def test_denoiser_config_defaults():
    from flika.ai.denoiser import DenoiserConfig
    cfg = DenoiserConfig()
    assert cfg.mode == 'n2v'
    assert cfg.backend == 'auto'
    assert cfg.epochs == 50
    assert cfg.batch_size == 8
    assert cfg.patch_size == 64
    assert cfg.learning_rate == 1e-4
    assert cfg.val_percentage == 0.1
    assert cfg.device == 'Auto'
    assert cfg.experiment_name == 'flika_denoiser'
    assert cfg.checkpoint_path == ''


def test_denoiser_config_custom():
    from flika.ai.denoiser import DenoiserConfig
    cfg = DenoiserConfig(mode='care', epochs=100, batch_size=16,
                          learning_rate=3e-4, experiment_name='my_model')
    assert cfg.mode == 'care'
    assert cfg.epochs == 100
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 3e-4
    assert cfg.experiment_name == 'my_model'


def test_resolved_backend_auto():
    from flika.ai.denoiser import DenoiserConfig
    cfg = DenoiserConfig(backend='auto')
    result = cfg.resolved_backend()
    assert result in ('careamics', 'n2v', 'csbdeep', 'none')


def test_resolved_backend_explicit():
    from flika.ai.denoiser import DenoiserConfig
    cfg = DenoiserConfig(backend='careamics')
    assert cfg.resolved_backend() == 'careamics'


def test_resolved_model_dir_default(tmp_path, monkeypatch):
    import os
    from flika.ai.denoiser import DenoiserConfig
    cfg = DenoiserConfig(model_dir='')
    result = cfg.resolved_model_dir()
    assert result.endswith(os.path.join('.FLIKA', 'models'))


def test_resolved_model_dir_custom(tmp_path):
    from flika.ai.denoiser import DenoiserConfig
    cfg = DenoiserConfig(model_dir=str(tmp_path))
    assert cfg.resolved_model_dir() == str(tmp_path)


# ---- Axes string helper ----

def test_axes_string_2d():
    from flika.ai.denoiser import _build_axes_string
    assert _build_axes_string(2) == 'YX'
    assert _build_axes_string(2, is_movie=False) == 'YX'


def test_axes_string_3d_movie():
    from flika.ai.denoiser import _build_axes_string
    assert _build_axes_string(3, is_movie=True) == 'SYX'


def test_axes_string_3d_volume():
    from flika.ai.denoiser import _build_axes_string
    assert _build_axes_string(3, is_movie=False) == 'ZYX'


def test_axes_string_4d():
    from flika.ai.denoiser import _build_axes_string
    assert _build_axes_string(4) == 'SZYX'


def test_axes_string_invalid():
    from flika.ai.denoiser import _build_axes_string
    with pytest.raises(ValueError, match="Unsupported ndim=5"):
        _build_axes_string(5)


# ---- TrainingWorker error handling ----

def test_training_worker_no_backend(qtbot):
    """TrainingWorker emits error when no backend is available."""
    import flika.ai.denoiser as mod
    from flika.ai.denoiser import DenoiserConfig, TrainingWorker

    # Force no-backend
    old_cache = mod._cached_backend
    mod._cached_backend = 'none'
    try:
        cfg = DenoiserConfig(backend='auto')
        dummy_data = np.random.rand(10, 64, 64).astype(np.float32)
        worker = TrainingWorker(cfg, dummy_data)

        with qtbot.waitSignal(worker.sig_error, timeout=5000) as blocker:
            worker.start()

        assert "No denoising backend found" in blocker.args[0]
    finally:
        mod._cached_backend = old_cache


def test_prediction_worker_no_backend(qtbot):
    """PredictionWorker emits error when no backend is available."""
    import flika.ai.denoiser as mod
    from flika.ai.denoiser import DenoiserConfig, PredictionWorker

    old_cache = mod._cached_backend
    mod._cached_backend = 'none'
    try:
        cfg = DenoiserConfig(backend='auto')
        dummy_data = np.random.rand(64, 64).astype(np.float32)
        worker = PredictionWorker(cfg, dummy_data)

        with qtbot.waitSignal(worker.sig_error, timeout=5000) as blocker:
            worker.start()

        assert "No denoising backend found" in blocker.args[0]
    finally:
        mod._cached_backend = old_cache
