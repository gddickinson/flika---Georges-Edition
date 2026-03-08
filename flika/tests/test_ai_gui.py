"""GUI tests for the flika AI module.

Tests that AI dialogs can be instantiated, their widgets are wired
correctly, and basic interactions work (class management, mode switching,
overlay creation, annotation drawing, etc.).

Requires a running FlikaApplication (not in _STANDALONE_MODULES).
"""
import numpy as np
import pytest

from .. import global_vars as g
from ..window import Window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_window():
    """Create a small test window and make it current."""
    img = np.random.rand(5, 64, 64).astype(np.float32)
    win = Window(img, name='ai_test')
    yield win
    try:
        win.close()
    except Exception:
        pass


@pytest.fixture
def sample_2d_window():
    """Create a 2D test window."""
    img = np.random.rand(64, 64).astype(np.float32)
    win = Window(img, name='ai_test_2d')
    yield win
    try:
        win.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Segmentation singletons (CellposeSegmenter, StarDistSegmenter)
# ---------------------------------------------------------------------------

class TestCellposeSegmenterGUI:

    def test_gui_creates_dialog(self, sample_window):
        from ..ai.segmentation import cellpose_segment
        cellpose_segment.gui()
        items = cellpose_segment.items
        names = [i['name'] for i in items]
        assert 'diameter' in names
        assert 'model_type' in names
        assert 'device' in names
        assert 'use_sam' in names
        cellpose_segment.ui.close()

    def test_model_type_options(self, sample_window):
        from ..ai.segmentation import cellpose_segment
        cellpose_segment.gui()
        model_item = [i for i in cellpose_segment.items if i['name'] == 'model_type'][0]
        combo = model_item['object']
        models = [combo.itemText(i) for i in range(combo.count())]
        assert 'cyto3' in models
        assert 'nuclei' in models
        cellpose_segment.ui.close()


class TestStarDistSegmenterGUI:

    def test_gui_creates_dialog(self, sample_window):
        from ..ai.segmentation import stardist_segment
        stardist_segment.gui()
        items = stardist_segment.items
        names = [i['name'] for i in items]
        assert 'model_name' in names
        assert 'device' in names
        stardist_segment.ui.close()

    def test_model_name_options(self, sample_window):
        from ..ai.segmentation import stardist_segment
        stardist_segment.gui()
        model_item = [i for i in stardist_segment.items if i['name'] == 'model_name'][0]
        combo = model_item['object']
        models = [combo.itemText(i) for i in range(combo.count())]
        assert '2D_versatile_fluo' in models
        stardist_segment.ui.close()


# ---------------------------------------------------------------------------
# Pixel Classifier Dialog
# ---------------------------------------------------------------------------

class TestPixelClassifierDialog:

    def test_dialog_opens(self, sample_window):
        from ..ai.classifier_dialog import PixelClassifierDialog
        dlg = PixelClassifierDialog(parent=g.m)
        assert dlg.windowTitle() == 'Pixel Classifier'
        dlg.close()

    def test_default_classes(self, sample_window):
        from ..ai.classifier_dialog import PixelClassifierDialog
        dlg = PixelClassifierDialog(parent=g.m)
        assert len(dlg._class_list) >= 2
        assert dlg.class_list_widget.count() >= 2
        dlg.close()

    def test_add_class(self, sample_window):
        from ..ai.classifier_dialog import PixelClassifierDialog
        dlg = PixelClassifierDialog(parent=g.m)
        initial_count = len(dlg._class_list)
        dlg._add_class()
        assert len(dlg._class_list) == initial_count + 1
        dlg.close()

    def test_backend_combo(self, sample_window):
        from ..ai.classifier_dialog import PixelClassifierDialog
        dlg = PixelClassifierDialog(parent=g.m)
        backends = [dlg.backend_combo.itemText(i) for i in range(dlg.backend_combo.count())]
        assert 'Random Forest' in backends
        assert 'CNN' in backends
        dlg.close()

    def test_image_view_exists(self, sample_window):
        from ..ai.classifier_dialog import PixelClassifierDialog
        dlg = PixelClassifierDialog(parent=g.m)
        assert dlg.image_view is not None
        dlg.close()


class TestPaintOverlay:

    def test_overlay_creation(self):
        from ..ai.classifier_dialog import PaintOverlay
        overlay = PaintOverlay(shape=(64, 64), n_frames=5)
        assert overlay.label_masks.shape == (5, 64, 64)
        assert overlay.label_masks.dtype == np.int32

    def test_brush_size(self):
        from ..ai.classifier_dialog import PaintOverlay
        overlay = PaintOverlay(shape=(32, 32))
        overlay.brush_size = 10
        assert overlay.brush_size == 10

    def test_active_class(self):
        from ..ai.classifier_dialog import PaintOverlay
        overlay = PaintOverlay(shape=(32, 32))
        overlay.active_class = 3
        assert overlay.active_class == 3

    def test_label_masks_shape(self):
        from ..ai.classifier_dialog import PaintOverlay
        overlay = PaintOverlay(shape=(32, 32), n_frames=3)
        assert overlay.label_masks.shape == (3, 32, 32)
        # Can write to mask directly
        overlay.label_masks[0, 10:20, 10:20] = 1
        assert overlay.label_masks[0, 15, 15] == 1


# ---------------------------------------------------------------------------
# Object Detection Dialog
# ---------------------------------------------------------------------------

class TestObjectDetectionDialog:

    def test_dialog_opens(self, sample_window):
        from ..ai.detection_dialog import ObjectDetectionDialog
        dlg = ObjectDetectionDialog(parent=g.m)
        assert 'Object Detection' in dlg.windowTitle()
        assert dlg.tabs.count() == 3
        dlg.close()

    def test_tab_names(self, sample_window):
        from ..ai.detection_dialog import ObjectDetectionDialog
        dlg = ObjectDetectionDialog(parent=g.m)
        tab_names = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
        assert 'Predict' in tab_names
        assert 'Annotate' in tab_names
        assert 'Train' in tab_names
        dlg.close()

    def test_predict_tab_widgets(self, sample_window):
        from ..ai.detection_dialog import ObjectDetectionDialog
        dlg = ObjectDetectionDialog(parent=g.m)
        assert hasattr(dlg, '_conf_slider')
        assert hasattr(dlg, '_model_combo')
        dlg.close()

    def test_annotate_tab_has_overlay_support(self, sample_window):
        from ..ai.detection_dialog import ObjectDetectionDialog
        dlg = ObjectDetectionDialog(parent=g.m)
        # _overlay is created when an image is loaded into annotate tab
        assert hasattr(dlg, '_overlay')
        dlg.close()


# ---------------------------------------------------------------------------
# Box Annotation Overlay
# ---------------------------------------------------------------------------

class TestBoxAnnotationOverlay:

    def test_overlay_creation(self):
        from ..ai.annotation_overlay import BoxAnnotationOverlay, InteractionMode
        from ..ai.annotations import AnnotationSet, AnnotationClass
        classes = [AnnotationClass(id=0, name='cell', color=(255, 0, 0))]
        aset = AnnotationSet(image_width=100, image_height=100, classes=classes)
        overlay = BoxAnnotationOverlay(aset)
        assert overlay is not None
        assert overlay.mode == InteractionMode.DRAW

    def test_mode_switching(self):
        from ..ai.annotation_overlay import BoxAnnotationOverlay, InteractionMode
        from ..ai.annotations import AnnotationSet, AnnotationClass
        classes = [AnnotationClass(id=0, name='cell', color=(255, 0, 0))]
        aset = AnnotationSet(image_width=100, image_height=100, classes=classes)
        overlay = BoxAnnotationOverlay(aset)
        overlay.mode = InteractionMode.SELECT
        assert overlay.mode == InteractionMode.SELECT
        overlay.mode = InteractionMode.DELETE
        assert overlay.mode == InteractionMode.DELETE

    def test_set_frame(self):
        from ..ai.annotation_overlay import BoxAnnotationOverlay, InteractionMode
        from ..ai.annotations import AnnotationSet, AnnotationClass
        classes = [AnnotationClass(id=0, name='cell', color=(255, 0, 0))]
        aset = AnnotationSet(image_width=100, image_height=100, n_frames=5, classes=classes)
        overlay = BoxAnnotationOverlay(aset)
        overlay.set_frame(3)
        assert overlay._current_frame == 3


# ---------------------------------------------------------------------------
# Particle Localizer Dialog
# ---------------------------------------------------------------------------

class TestParticleLocalizerDialog:

    def test_dialog_opens(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        assert dlg.windowTitle() == 'Particle Localizer'
        dlg.close()

    def test_mode_options(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        modes = [dlg.mode_combo.itemText(i) for i in range(dlg.mode_combo.count())]
        assert 'Train from Simulation' in modes
        assert 'Train from Data' in modes
        assert 'Load Pre-trained' in modes
        dlg.close()

    def test_device_options(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        devices = [dlg.device_combo.itemText(i) for i in range(dlg.device_combo.count())]
        assert 'Auto' in devices
        assert 'CPU' in devices
        dlg.close()

    def test_psf_group_visible_toggles_with_mode(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        dlg.show()
        # Simulation mode: psf_group visible
        dlg.mode_combo.setCurrentIndex(0)
        assert dlg.psf_group.isVisible()
        # Load Pre-trained: psf_group hidden
        dlg.mode_combo.setCurrentIndex(2)
        assert not dlg.psf_group.isVisible()
        dlg.close()

    def test_train_disabled_in_load_mode(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        dlg.mode_combo.setCurrentIndex(2)  # Load Pre-trained
        assert not dlg.train_btn.isEnabled()
        dlg.close()

    def test_window_combos_populated(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        assert dlg.predict_window_combo.count() >= 1
        dlg.close()

    def test_build_config(self, sample_window):
        from ..ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        config = dlg._build_config()
        assert config.epochs == dlg.epochs_spin.value()
        assert config.psf_sigma == dlg.psf_sigma_spin.value()
        assert config.n_particles == dlg.particles_spin.value()
        dlg.close()


# ---------------------------------------------------------------------------
# Denoiser Dialog
# ---------------------------------------------------------------------------

class TestDenoiserDialog:

    def test_dialog_opens(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        assert dlg.windowTitle() == 'AI Denoiser'
        dlg.close()

    def test_mode_options(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        modes = [dlg.mode_combo.itemText(i) for i in range(dlg.mode_combo.count())]
        assert 'N2V (Self-Supervised)' in modes
        assert 'CARE (Supervised)' in modes
        assert 'Load Pre-trained' in modes
        dlg.close()

    def test_n2v_mode_disables_clean(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        dlg.mode_combo.setCurrentIndex(0)  # N2V
        assert not dlg.clean_combo.isEnabled()
        dlg.close()

    def test_care_mode_enables_clean(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        dlg.mode_combo.setCurrentIndex(1)  # CARE
        assert dlg.clean_combo.isEnabled()
        dlg.close()

    def test_pretrained_mode_disables_train(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        dlg.mode_combo.setCurrentIndex(2)  # Load Pre-trained
        assert not dlg.train_btn.isEnabled()
        dlg.close()

    def test_build_config(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        config = dlg._build_config()
        assert config.epochs == dlg.epochs_spin.value()
        assert config.batch_size == dlg.batch_spin.value()
        dlg.close()

    def test_backend_status_shown(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        status = dlg.status_label.text()
        # Should show either the backend name or a "not installed" message
        assert 'Backend' in status or 'backend' in status or 'install' in status.lower()
        dlg.close()

    def test_window_combos_populated(self, sample_window):
        from ..ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        assert dlg.noisy_combo.count() >= 1
        dlg.close()


# ---------------------------------------------------------------------------
# SAM Dialog
# ---------------------------------------------------------------------------

class TestSAMDialog:

    def test_dialog_opens(self, sample_window):
        from ..ai.sam_dialog import SAMSegmentationDialog
        dlg = SAMSegmentationDialog(parent=g.m)
        assert dlg.windowTitle() == 'SAM Interactive Segmentation'
        dlg.close()

    def test_model_options(self, sample_window):
        from ..ai.sam_dialog import SAMSegmentationDialog
        dlg = SAMSegmentationDialog(parent=g.m)
        model_count = dlg.model_combo.count()
        assert model_count >= 3  # vit_b, vit_l, vit_h
        dlg.close()

    def test_mode_options(self, sample_window):
        from ..ai.sam_dialog import SAMSegmentationDialog
        dlg = SAMSegmentationDialog(parent=g.m)
        modes = [dlg.mode_combo.itemText(i) for i in range(dlg.mode_combo.count())]
        assert 'Point' in modes
        assert 'Box' in modes
        dlg.close()

    def test_box_roi_hidden_in_point_mode(self, sample_window):
        from ..ai.sam_dialog import SAMSegmentationDialog
        dlg = SAMSegmentationDialog(parent=g.m)
        dlg.mode_combo.setCurrentText('Point')
        assert not dlg._box_roi.isVisible()
        dlg.close()

    def test_box_roi_shown_in_box_mode(self, sample_window):
        from ..ai.sam_dialog import SAMSegmentationDialog
        dlg = SAMSegmentationDialog(parent=g.m)
        dlg.mode_combo.setCurrentText('Box')
        assert dlg._box_roi.isVisible()
        dlg.close()

    def test_clear_prompts(self, sample_window):
        from ..ai.sam_dialog import SAMSegmentationDialog, PromptPoint
        dlg = SAMSegmentationDialog(parent=g.m)
        dlg._points.append(PromptPoint(x=10, y=20, is_positive=True))
        dlg._clear_prompts()
        assert len(dlg._points) == 0
        assert dlg._mask is None
        dlg.close()

    def test_checkpoint_helpers(self):
        from ..ai.sam_dialog import _checkpoint_path, _is_checkpoint_downloaded
        path = _checkpoint_path('vit_b')
        assert 'sam_vit_b' in path
        # Should not crash even if model not downloaded
        result = _is_checkpoint_downloaded('vit_b')
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Model Zoo Browser
# ---------------------------------------------------------------------------

class TestModelZooBrowser:

    def test_dialog_opens(self, sample_window):
        from ..ai.model_zoo import ModelZooBrowser
        dlg = ModelZooBrowser(parent=g.m)
        assert dlg.windowTitle() == 'BioImage.IO Model Zoo'
        dlg.close()

    def test_table_columns(self, sample_window):
        from ..ai.model_zoo import ModelZooBrowser
        dlg = ModelZooBrowser(parent=g.m)
        assert dlg.table.columnCount() == 5
        headers = [dlg.table.horizontalHeaderItem(i).text()
                    for i in range(dlg.table.columnCount())]
        assert 'Name' in headers
        assert 'Description' in headers
        dlg.close()

    def test_task_filter_options(self, sample_window):
        from ..ai.model_zoo import ModelZooBrowser
        dlg = ModelZooBrowser(parent=g.m)
        tasks = [dlg.task_combo.itemText(i) for i in range(dlg.task_combo.count())]
        assert 'All Tasks' in tasks
        assert 'segmentation' in tasks
        assert 'denoising' in tasks
        dlg.close()

    def test_buttons_exist(self, sample_window):
        from ..ai.model_zoo import ModelZooBrowser
        dlg = ModelZooBrowser(parent=g.m)
        assert dlg.download_btn is not None
        assert dlg.run_btn is not None
        dlg.close()

    def test_model_cache_helpers(self):
        from ..ai.model_zoo import _model_cache_dir, _is_model_cached
        cache_dir = _model_cache_dir('test/model')
        assert 'test_model' in cache_dir
        assert not _is_model_cached('nonexistent/model')


# ---------------------------------------------------------------------------
# AI menu integration
# ---------------------------------------------------------------------------

class TestAIMenuIntegration:
    """Test that the AI menu actions open dialogs without errors."""

    def test_ai_classify_gui(self, sample_window):
        from ..ai.segmentation import ai_classify
        ai_classify.gui()
        # Dialog should have been appended to g.dialogs
        assert len(g.dialogs) > 0
        dlg = g.dialogs[-1]
        assert dlg.windowTitle() == 'Pixel Classifier'
        dlg.close()

    def test_ai_localize_gui(self, sample_window):
        from ..ai.segmentation import ai_localize
        ai_localize.gui()
        assert len(g.dialogs) > 0
        dlg = g.dialogs[-1]
        assert dlg.windowTitle() == 'Particle Localizer'
        dlg.close()

    def test_ai_denoise_gui(self, sample_window):
        from ..ai.segmentation import ai_denoise
        ai_denoise.gui()
        assert len(g.dialogs) > 0
        dlg = g.dialogs[-1]
        assert dlg.windowTitle() == 'AI Denoiser'
        dlg.close()

    def test_ai_detect_gui(self, sample_window):
        from ..ai.segmentation import ai_detect
        ai_detect.gui()
        assert len(g.dialogs) > 0
        dlg = g.dialogs[-1]
        assert 'Object Detection' in dlg.windowTitle()
        dlg.close()

    def test_ai_sam_gui(self, sample_window):
        from ..ai.segmentation import ai_sam
        ai_sam.gui()
        assert len(g.dialogs) > 0
        dlg = g.dialogs[-1]
        assert dlg.windowTitle() == 'SAM Interactive Segmentation'
        dlg.close()

    def test_ai_model_zoo_gui(self, sample_window):
        from ..ai.segmentation import ai_model_zoo
        ai_model_zoo.gui()
        assert len(g.dialogs) > 0
        dlg = g.dialogs[-1]
        assert dlg.windowTitle() == 'BioImage.IO Model Zoo'
        dlg.close()


# ---------------------------------------------------------------------------
# PromptPoint dataclass
# ---------------------------------------------------------------------------

class TestPromptPoint:

    def test_defaults(self):
        from ..ai.sam_dialog import PromptPoint
        p = PromptPoint(x=10, y=20)
        assert p.is_positive is True

    def test_negative_prompt(self):
        from ..ai.sam_dialog import PromptPoint
        p = PromptPoint(x=10, y=20, is_positive=False)
        assert p.is_positive is False


# ---------------------------------------------------------------------------
# Detection workers (instantiation only - no inference)
# ---------------------------------------------------------------------------

class TestDetectionWorkers:

    def test_predict_worker_creation(self):
        from ..ai.detection_dialog import DetectionPredictWorker
        from ..ai.detection_backend import DetectionConfig, UltralyticsBackend
        backend = UltralyticsBackend()
        config = DetectionConfig()
        img = np.random.rand(64, 64, 3).astype(np.uint8)
        worker = DetectionPredictWorker(backend, img, config)
        assert worker is not None
        assert not worker.isRunning()

    def test_train_worker_creation(self):
        from ..ai.detection_dialog import DetectionTrainWorker
        from ..ai.detection_backend import DetectionConfig, UltralyticsBackend
        backend = UltralyticsBackend()
        config = DetectionConfig()
        worker = DetectionTrainWorker(backend, '/tmp/fake.yaml', config)
        assert worker is not None
        assert not worker.isRunning()


# ---------------------------------------------------------------------------
# Localizer workers (instantiation only)
# ---------------------------------------------------------------------------

class TestLocalizerWorkers:

    def test_simulation_worker_creation(self):
        from ..ai.localizer_dialog import SimulationWorker
        from ..ai.psf_simulator import PSFConfig
        cfg = PSFConfig(image_size=32, n_particles=5, n_frames=2)
        worker = SimulationWorker(cfg)
        assert worker is not None

    def test_training_worker_creation(self):
        from ..ai.localizer_dialog import LocalizerTrainingWorker
        # Can create with a mock backend
        class MockBackend:
            def name(self): return 'mock'
            def train(self, *a, **kw): pass
        images = np.zeros((10, 32, 32), dtype=np.float32)
        density = np.zeros((10, 32, 32), dtype=np.float32)
        from ..ai.localizer_backends import LocalizerConfig
        config = LocalizerConfig(epochs=1)
        worker = LocalizerTrainingWorker(MockBackend(), images, density, config)
        assert worker is not None

    def test_prediction_worker_creation(self):
        from ..ai.localizer_dialog import LocalizerPredictionWorker
        class MockBackend:
            def name(self): return 'mock'
            def predict(self, img): return np.zeros_like(img, dtype=np.float32)
        data = np.zeros((32, 32), dtype=np.float32)
        worker = LocalizerPredictionWorker(MockBackend(), data, threshold=0.2,
                                            min_distance=3)
        assert worker is not None
