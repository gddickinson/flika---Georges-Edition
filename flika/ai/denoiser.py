"""AI denoising backend abstraction for flika.

Supports three backends in priority order:
1. CAREamics (PyTorch, actively maintained)
2. n2v (TensorFlow)
3. csbdeep/CARE (TensorFlow)

Heavy imports are deferred so this module can be imported without any
AI packages installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from qtpy.QtCore import QThread, Signal

from ..logger import logger

_cached_backend: Optional[str] = None


def detect_denoiser_backend() -> str:
    """Return the best available denoising backend.

    Returns one of ``'careamics'``, ``'n2v'``, ``'csbdeep'``, or ``'none'``.
    Result is cached for the session.
    """
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend

    # Priority 1: CAREamics (PyTorch)
    try:
        import careamics  # noqa: F401
        _cached_backend = 'careamics'
        return _cached_backend
    except ImportError:
        pass

    # Priority 2: n2v (TensorFlow)
    try:
        import n2v  # noqa: F401
        _cached_backend = 'n2v'
        return _cached_backend
    except ImportError:
        pass

    # Priority 3: csbdeep (TensorFlow)
    try:
        import csbdeep  # noqa: F401
        _cached_backend = 'csbdeep'
        return _cached_backend
    except ImportError:
        pass

    _cached_backend = 'none'
    return _cached_backend


def _build_axes_string(ndim: int, is_movie: bool = True) -> str:
    """Convert array dimensionality to an axes string for N2V/CARE.

    Parameters
    ----------
    ndim : int
        Number of dimensions in the array.
    is_movie : bool
        If True, treat the first axis of 3-D data as time/samples (``'SYX'``).
        If False, treat it as Z (``'ZYX'``).

    Returns
    -------
    str
        Axes string like ``'YX'``, ``'SYX'``, ``'ZYX'``, or ``'SZYX'``.
    """
    if ndim == 2:
        return 'YX'
    elif ndim == 3:
        return 'SYX' if is_movie else 'ZYX'
    elif ndim == 4:
        return 'SZYX'
    else:
        raise ValueError(f"Unsupported ndim={ndim}; expected 2, 3, or 4")


def _patch_keras3_weights():
    """Monkey-patch Keras 3.x weight-saving for n2v/csbdeep compatibility.

    Keras 3.x requires ``.weights.h5`` extension for weight files.  n2v and
    csbdeep hardcode plain ``.h5`` filenames in ModelCheckpoint callbacks,
    ``save_weights()`` calls, and ``load_weights()`` calls.  This patches all
    three to rewrite the filepath automatically.

    Safe to call multiple times; only patches once.
    """
    try:
        import keras
        from tensorflow.keras.callbacks import ModelCheckpoint
    except ImportError:
        return

    def _fix_h5(filepath):
        if (isinstance(filepath, str)
                and filepath.endswith('.h5')
                and not filepath.endswith('.weights.h5')):
            return filepath[:-3] + '.weights.h5'
        return filepath

    # Patch ModelCheckpoint.__init__
    if not getattr(ModelCheckpoint, '_flika_patched', False):
        _orig_mc_init = ModelCheckpoint.__init__

        def _patched_mc_init(self, filepath, *args, **kwargs):
            if kwargs.get('save_weights_only', False):
                filepath = _fix_h5(filepath)
            _orig_mc_init(self, filepath, *args, **kwargs)

        ModelCheckpoint.__init__ = _patched_mc_init
        ModelCheckpoint._flika_patched = True

    # Patch Model.save_weights and Model.load_weights
    Model = keras.Model
    if not getattr(Model.save_weights, '_flika_patched', False):
        _orig_save = Model.save_weights

        def _patched_save(self, filepath, *args, **kwargs):
            return _orig_save(self, _fix_h5(filepath), *args, **kwargs)

        _patched_save._flika_patched = True
        Model.save_weights = _patched_save

    if not getattr(Model.load_weights, '_flika_patched', False):
        _orig_load = Model.load_weights

        def _patched_load(self, filepath, *args, **kwargs):
            return _orig_load(self, _fix_h5(filepath), *args, **kwargs)

        _patched_load._flika_patched = True
        Model.load_weights = _patched_load


def _patch_numpy2_compat():
    """Restore ``np.product`` removed in NumPy 2.0.

    n2v uses ``np.product`` which was removed in NumPy 2.0 (replaced by
    ``np.prod``).  This shim restores it so n2v can run unmodified.

    Safe to call multiple times.
    """
    import numpy as _np
    if not hasattr(_np, 'product'):
        _np.product = _np.prod


def _patch_csbdeep_tensorboard():
    """Disable CARETensorBoardImage for Keras 3.x compatibility.

    csbdeep's ``CARETensorBoardImage`` is deeply incompatible with Keras 3.x
    (read-only ``Callback.model``, changed TF summary APIs, etc.).  Since
    flika has its own progress callback, we replace it with a harmless no-op.

    Safe to call multiple times; only patches once.
    """
    try:
        from csbdeep.utils.tf import CARETensorBoardImage
    except ImportError:
        return

    if getattr(CARETensorBoardImage, '_flika_patched', False):
        return

    import tensorflow as tf

    def _noop_init(self, *args, **kwargs):
        super(CARETensorBoardImage, self).__init__()

    CARETensorBoardImage.__init__ = _noop_init
    CARETensorBoardImage.on_epoch_end = lambda self, *a, **kw: None
    CARETensorBoardImage._flika_patched = True


@dataclass
class DenoiserConfig:
    """Configuration for denoiser training and prediction."""
    mode: str = 'n2v'           # 'n2v' or 'care'
    backend: str = 'auto'       # 'auto', 'careamics', 'n2v', 'csbdeep'
    epochs: int = 50
    batch_size: int = 8
    patch_size: int = 64
    learning_rate: float = 1e-4
    val_percentage: float = 0.1
    device: str = 'Auto'
    model_dir: str = ''         # defaults to ~/.FLIKA/models/
    experiment_name: str = 'flika_denoiser'
    checkpoint_path: str = ''   # for predict-only from pre-trained

    def resolved_backend(self) -> str:
        """Return the actual backend to use (resolve 'auto')."""
        if self.backend != 'auto':
            return self.backend
        return detect_denoiser_backend()

    def resolved_model_dir(self) -> str:
        """Return model directory, defaulting to ~/.FLIKA/models/."""
        if self.model_dir:
            return self.model_dir
        default = os.path.join(os.path.expanduser('~'), '.FLIKA', 'models')
        os.makedirs(default, exist_ok=True)
        return default


class TrainingWorker(QThread):
    """Background thread for denoiser training."""

    sig_epoch_done = Signal(int, float, float)   # epoch, train_loss, val_loss
    sig_progress = Signal(int)                   # percent 0-100
    sig_status = Signal(str)                     # status message
    sig_finished = Signal(str)                   # model path
    sig_error = Signal(str)                      # error message

    def __init__(self, config: DenoiserConfig, train_data, val_data=None,
                 clean_data=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.clean_data = clean_data  # for CARE mode
        self._stop_requested = False

    def request_stop(self):
        """Request training to stop after the current epoch."""
        self._stop_requested = True

    def run(self):
        try:
            backend = self.config.resolved_backend()
            if backend == 'none':
                self.sig_error.emit(
                    "No denoising backend found.\n\n"
                    "Install one of:\n"
                    "  pip install careamics        (recommended, PyTorch)\n"
                    "  pip install n2v tensorflow   (TensorFlow)\n"
                    "  pip install csbdeep tensorflow")
                return

            self.sig_status.emit(f"Starting training with {backend} backend...")

            if backend == 'careamics':
                self._run_careamics()
            elif backend == 'n2v':
                self._run_n2v()
            elif backend == 'csbdeep':
                self._run_csbdeep()
            else:
                self.sig_error.emit(f"Unknown backend: {backend}")
        except Exception as e:
            logger.exception("Training failed")
            self.sig_error.emit(str(e))

    def _run_careamics(self):
        """Train using CAREamics (PyTorch)."""
        import numpy as np
        from careamics import CAREamist

        cfg = self.config
        axes = _build_axes_string(self.train_data.ndim, is_movie=True)

        if cfg.mode == 'n2v':
            from careamics import create_n2v_configuration
            careamics_cfg = create_n2v_configuration(
                experiment_name=cfg.experiment_name,
                data_type='array',
                axes=axes,
                patch_size=[cfg.patch_size] * 2,
                batch_size=cfg.batch_size,
                num_epochs=cfg.epochs,
            )
        else:
            from careamics import create_care_configuration
            careamics_cfg = create_care_configuration(
                experiment_name=cfg.experiment_name,
                data_type='array',
                axes=axes,
                patch_size=[cfg.patch_size] * 2,
                batch_size=cfg.batch_size,
                num_epochs=cfg.epochs,
            )

        # Create callback to emit progress signals
        try:
            from lightning.pytorch.callbacks import Callback as LightningCallback
        except ImportError:
            from pytorch_lightning.callbacks import Callback as LightningCallback

        worker = self

        class FlikaCallback(LightningCallback):
            def on_train_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch
                train_loss = float(trainer.callback_metrics.get('train_loss', 0))
                val_loss = float(trainer.callback_metrics.get('val_loss', 0))
                worker.sig_epoch_done.emit(epoch, train_loss, val_loss)
                pct = int(100 * (epoch + 1) / cfg.epochs)
                worker.sig_progress.emit(pct)
                if worker._stop_requested:
                    trainer.should_stop = True

        model_dir = cfg.resolved_model_dir()
        careamist = CAREamist(careamics_cfg, callbacks=[FlikaCallback()])

        # Set device
        from ..utils.accel import get_torch_device
        device = get_torch_device(cfg.device)
        if device is not None and str(device) != 'cpu':
            careamics_cfg.training_config = getattr(careamics_cfg, 'training_config', None)

        data = self.train_data.astype(np.float32)
        if cfg.mode == 'care' and self.clean_data is not None:
            careamist.train(
                train_source=data,
                train_target=self.clean_data.astype(np.float32),
            )
        else:
            careamist.train(train_source=data)

        # Save model
        save_path = os.path.join(model_dir, cfg.experiment_name)
        try:
            careamist.export_to_bmz(
                path=save_path + '.zip',
                name=cfg.experiment_name,
                authors=[{"name": "flika"}],
                input_array=data[0:1] if data.ndim >= 3 else data[np.newaxis],
            )
            self.sig_finished.emit(save_path + '.zip')
        except Exception:
            # Fallback: just report the checkpoint directory
            ckpt_dir = os.path.join(model_dir, 'checkpoints')
            self.sig_finished.emit(ckpt_dir if os.path.exists(ckpt_dir) else model_dir)

    def _run_n2v(self):
        """Train using n2v (TensorFlow)."""
        import numpy as np

        self._setup_tf_memory()

        from n2v.models import N2VConfig, N2V
        from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

        cfg = self.config
        axes = _build_axes_string(self.train_data.ndim, is_movie=True)

        self.sig_status.emit("Generating patches...")
        datagen = N2V_DataGenerator()

        # n2v expects list of arrays with channel axis
        data = self.train_data.astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis, ..., np.newaxis]  # add S and C
            patch_axes = 'SYXC'
        elif data.ndim == 3:
            data = data[..., np.newaxis]  # add C
            patch_axes = 'SYXC'
        else:
            data = data[..., np.newaxis]
            patch_axes = 'SZYXC'

        patches = datagen.generate_patches_from_list(
            [data],
            shape=(cfg.patch_size, cfg.patch_size),
        )

        n_val = max(1, int(len(patches) * cfg.val_percentage))
        train_patches = patches[:-n_val]
        val_patches = patches[-n_val:]

        self.sig_status.emit("Configuring N2V model...")
        n2v_config = N2VConfig(
            X=train_patches,
            unet_kern_size=3,
            train_steps_per_epoch=max(1, len(train_patches) // cfg.batch_size),
            train_epochs=cfg.epochs,
            train_loss='mse',
            batch_norm=True,
            train_batch_size=cfg.batch_size,
            n2v_perc_pix=0.198,
            n2v_patch_shape=(cfg.patch_size, cfg.patch_size),
            n2v_manipulator='uniform_withCP',
            train_learning_rate=cfg.learning_rate,
        )

        model_dir = cfg.resolved_model_dir()
        model = N2V(n2v_config, cfg.experiment_name, basedir=model_dir)

        # Keras callback for progress
        import tensorflow as tf
        worker = self

        class FlikaKerasCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self_cb, epoch, logs=None):
                logs = logs or {}
                train_loss = float(logs.get('loss', 0))
                val_loss = float(logs.get('val_loss', 0))
                worker.sig_epoch_done.emit(epoch, train_loss, val_loss)
                pct = int(100 * (epoch + 1) / cfg.epochs)
                worker.sig_progress.emit(pct)
                if worker._stop_requested:
                    model.keras_model.stop_training = True

        self.sig_status.emit("Training N2V...")
        # N2V.train() doesn't accept callbacks; prepare the model first
        # then append our callback to model.callbacks before training.
        _patch_numpy2_compat()
        _patch_keras3_weights()
        _patch_csbdeep_tensorboard()
        model.prepare_for_training(metrics=())
        model.callbacks.append(FlikaKerasCallback())
        model.train(train_patches, val_patches)

        save_path = os.path.join(model_dir, cfg.experiment_name)
        self.sig_finished.emit(save_path)

    def _run_csbdeep(self):
        """Train using csbdeep/CARE (TensorFlow)."""
        import numpy as np

        self._setup_tf_memory()

        from csbdeep.models import Config as CareConfig, CARE

        cfg = self.config

        if self.clean_data is None:
            self.sig_error.emit(
                "CARE requires paired noisy/clean data.\n"
                "Select clean data or use N2V for self-supervised denoising.")
            return

        data = self.train_data.astype(np.float32)
        target = self.clean_data.astype(np.float32)

        # Add channel axis if needed
        if data.ndim == 2:
            data = data[np.newaxis, ..., np.newaxis]
            target = target[np.newaxis, ..., np.newaxis]
            axes = 'SYXC'
        elif data.ndim == 3:
            data = data[..., np.newaxis]
            target = target[..., np.newaxis]
            axes = 'SYXC'
        else:
            data = data[..., np.newaxis]
            target = target[..., np.newaxis]
            axes = 'SZYXC'

        n_val = max(1, int(len(data) * cfg.val_percentage))
        X_train, X_val = data[:-n_val], data[-n_val:]
        Y_train, Y_val = target[:-n_val], target[-n_val:]

        self.sig_status.emit("Configuring CARE model...")
        care_config = CareConfig(
            axes,
            n_channel_in=1,
            n_channel_out=1,
            train_epochs=cfg.epochs,
            train_steps_per_epoch=max(1, len(X_train) // cfg.batch_size),
            train_batch_size=cfg.batch_size,
            train_learning_rate=cfg.learning_rate,
        )

        model_dir = cfg.resolved_model_dir()
        model = CARE(care_config, cfg.experiment_name, basedir=model_dir)

        import tensorflow as tf
        worker = self

        class FlikaKerasCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self_cb, epoch, logs=None):
                logs = logs or {}
                train_loss = float(logs.get('loss', 0))
                val_loss = float(logs.get('val_loss', 0))
                worker.sig_epoch_done.emit(epoch, train_loss, val_loss)
                pct = int(100 * (epoch + 1) / cfg.epochs)
                worker.sig_progress.emit(pct)
                if worker._stop_requested:
                    model.keras_model.stop_training = True

        self.sig_status.emit("Training CARE...")
        # CARE.train() doesn't accept callbacks; prepare first then append.
        _patch_keras3_weights()
        _patch_csbdeep_tensorboard()
        model.prepare_for_training()
        model.callbacks.append(FlikaKerasCallback())
        model.train(X_train, Y_train,
                     validation_data=(X_val, Y_val))

        save_path = os.path.join(model_dir, cfg.experiment_name)
        self.sig_finished.emit(save_path)

    @staticmethod
    def _setup_tf_memory():
        """Enable TensorFlow GPU memory growth to avoid OOM."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


class PredictionWorker(QThread):
    """Background thread for denoiser inference."""

    sig_result = Signal(object)    # denoised numpy array
    sig_progress = Signal(int)     # percent 0-100
    sig_status = Signal(str)       # status message
    sig_error = Signal(str)        # error message

    def __init__(self, config: DenoiserConfig, data, parent=None):
        super().__init__(parent)
        self.config = config
        self.data = data

    def run(self):
        try:
            backend = self.config.resolved_backend()
            if backend == 'none':
                self.sig_error.emit(
                    "No denoising backend found.\n\n"
                    "Install one of:\n"
                    "  pip install careamics        (recommended, PyTorch)\n"
                    "  pip install n2v tensorflow   (TensorFlow)\n"
                    "  pip install csbdeep tensorflow")
                return

            self.sig_status.emit(f"Running prediction with {backend} backend...")

            if backend == 'careamics':
                self._predict_careamics()
            elif backend == 'n2v':
                self._predict_n2v()
            elif backend == 'csbdeep':
                self._predict_csbdeep()
        except Exception as e:
            logger.exception("Prediction failed")
            self.sig_error.emit(str(e))

    def _predict_careamics(self):
        import numpy as np
        from careamics import CAREamist

        cfg = self.config
        axes = _build_axes_string(self.data.ndim, is_movie=True)
        data = self.data.astype(np.float32)

        if cfg.checkpoint_path:
            careamist = CAREamist(cfg.checkpoint_path)
        else:
            model_dir = cfg.resolved_model_dir()
            ckpt = os.path.join(model_dir, cfg.experiment_name + '.zip')
            if not os.path.exists(ckpt):
                ckpt = os.path.join(model_dir, cfg.experiment_name)
            careamist = CAREamist(ckpt)

        self.sig_status.emit("Predicting...")
        self.sig_progress.emit(10)

        # Predict frame-by-frame for 3D+ to show progress
        if data.ndim >= 3:
            result_frames = []
            n_frames = data.shape[0]
            for i in range(n_frames):
                frame = data[i]
                pred = careamist.predict(source=frame, axes='YX' if frame.ndim == 2 else 'ZYX')
                if isinstance(pred, list):
                    pred = pred[0]
                result_frames.append(np.squeeze(pred))
                self.sig_progress.emit(int(10 + 90 * (i + 1) / n_frames))
            result = np.stack(result_frames, axis=0)
        else:
            result = careamist.predict(source=data, axes=axes)
            if isinstance(result, list):
                result = result[0]
            result = np.squeeze(result)
            self.sig_progress.emit(100)

        self.sig_result.emit(result)

    def _predict_n2v(self):
        import numpy as np

        TrainingWorker._setup_tf_memory()

        from n2v.models import N2V

        cfg = self.config
        data = self.data.astype(np.float32)

        if cfg.checkpoint_path:
            model_name = os.path.basename(cfg.checkpoint_path)
            basedir = os.path.dirname(cfg.checkpoint_path)
        else:
            model_name = cfg.experiment_name
            basedir = cfg.resolved_model_dir()

        model = N2V(config=None, name=model_name, basedir=basedir)

        self.sig_status.emit("Predicting with N2V...")
        self.sig_progress.emit(10)

        axes = _build_axes_string(data.ndim, is_movie=True)
        # Predict per-frame for movies
        if data.ndim >= 3:
            result_frames = []
            n_frames = data.shape[0]
            for i in range(n_frames):
                frame = data[i]
                pred = model.predict(frame, axes='YX' if frame.ndim == 2 else 'ZYX')
                result_frames.append(pred)
                self.sig_progress.emit(int(10 + 90 * (i + 1) / n_frames))
            result = np.stack(result_frames, axis=0)
        else:
            result = model.predict(data, axes='YX')
            self.sig_progress.emit(100)

        self.sig_result.emit(result)

    def _predict_csbdeep(self):
        import numpy as np

        TrainingWorker._setup_tf_memory()

        from csbdeep.models import CARE

        cfg = self.config
        data = self.data.astype(np.float32)

        if cfg.checkpoint_path:
            model_name = os.path.basename(cfg.checkpoint_path)
            basedir = os.path.dirname(cfg.checkpoint_path)
        else:
            model_name = cfg.experiment_name
            basedir = cfg.resolved_model_dir()

        model = CARE(config=None, name=model_name, basedir=basedir)

        self.sig_status.emit("Predicting with CARE...")
        self.sig_progress.emit(10)

        if data.ndim >= 3:
            result_frames = []
            n_frames = data.shape[0]
            for i in range(n_frames):
                frame = data[i]
                axes_str = 'YX' if frame.ndim == 2 else 'ZYX'
                pred = model.predict(frame, axes_str)
                result_frames.append(pred)
                self.sig_progress.emit(int(10 + 90 * (i + 1) / n_frames))
            result = np.stack(result_frames, axis=0)
        else:
            result = model.predict(data, 'YX')
            self.sig_progress.emit(100)

        self.sig_result.emit(result)
