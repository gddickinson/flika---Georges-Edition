"""Non-GUI tests for the flika AI module.

Tests pure Python / NumPy components that do not require a Qt application:
- AnnotationSet (CRUD, YOLO/COCO/VOC export/import)
- DetectionConfig and prepare_image_for_detection()
- NMS (non-maximum suppression)
- FeatureExtractor (feature computation on synthetic images)
- PSFSimulator (frame/stack generation, density maps, coordinate extraction)
- ClassifierConfig and RandomForestBackend (train/predict round-trip)
- LocalizerConfig defaults
- Denoiser utility functions
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Annotation data model
# ---------------------------------------------------------------------------

class TestBoundingBox:

    def test_defaults(self):
        from flika.ai.annotations import BoundingBox
        bb = BoundingBox(x=10, y=20, width=30, height=40, class_id=0, class_name='cell')
        assert bb.confidence == 1.0
        assert bb.frame == 0

    def test_custom_fields(self):
        from flika.ai.annotations import BoundingBox
        bb = BoundingBox(x=5, y=6, width=7, height=8, class_id=2,
                         class_name='nucleus', confidence=0.85, frame=3)
        assert bb.confidence == 0.85
        assert bb.frame == 3


class TestAnnotationClass:

    def test_creation(self):
        from flika.ai.annotations import AnnotationClass
        ac = AnnotationClass(id=0, name='cell', color=(255, 0, 0))
        assert ac.name == 'cell'
        assert ac.color == (255, 0, 0)


class TestAnnotationSet:

    def _make_set(self):
        from flika.ai.annotations import AnnotationSet, AnnotationClass, BoundingBox
        classes = [
            AnnotationClass(id=0, name='cell', color=(255, 0, 0)),
            AnnotationClass(id=1, name='bg', color=(0, 255, 0)),
        ]
        aset = AnnotationSet(image_width=640, image_height=480, n_frames=3,
                              classes=classes)
        aset.add_box(BoundingBox(x=10, y=20, width=30, height=40,
                                  class_id=0, class_name='cell', frame=0))
        aset.add_box(BoundingBox(x=100, y=200, width=50, height=60,
                                  class_id=1, class_name='bg', frame=1))
        aset.add_box(BoundingBox(x=50, y=50, width=25, height=25,
                                  class_id=0, class_name='cell', frame=0,
                                  confidence=0.9))
        return aset

    def test_add_and_get_boxes(self):
        aset = self._make_set()
        assert len(aset.boxes) == 3
        assert len(aset.get_frame_boxes(0)) == 2
        assert len(aset.get_frame_boxes(1)) == 1
        assert len(aset.get_frame_boxes(2)) == 0

    def test_remove_box(self):
        aset = self._make_set()
        box = aset.boxes[0]
        aset.remove_box(box)
        assert len(aset.boxes) == 2

    def test_remove_nonexistent_box(self):
        from flika.ai.annotations import BoundingBox
        aset = self._make_set()
        fake = BoundingBox(x=999, y=999, width=1, height=1, class_id=0, class_name='x')
        aset.remove_box(fake)  # should not raise
        assert len(aset.boxes) == 3

    def test_update_box(self):
        aset = self._make_set()
        box = aset.boxes[0]
        aset.update_box(box, x=99, confidence=0.5)
        assert box.x == 99
        assert box.confidence == 0.5

    def test_clear_all_boxes(self):
        aset = self._make_set()
        aset.clear_boxes()
        assert len(aset.boxes) == 0

    def test_clear_frame_boxes(self):
        aset = self._make_set()
        aset.clear_boxes(frame=0)
        assert len(aset.boxes) == 1
        assert aset.boxes[0].frame == 1

    def test_add_class(self):
        aset = self._make_set()
        cls = aset.add_class('nucleus', (0, 0, 255))
        assert cls.id == 2
        assert cls.name == 'nucleus'
        assert len(aset.classes) == 3

    def test_remove_class_cascades(self):
        aset = self._make_set()
        aset.remove_class(0)  # remove 'cell'
        assert all(c.id != 0 for c in aset.classes)
        assert all(b.class_id != 0 for b in aset.boxes)

    def test_rename_class(self):
        aset = self._make_set()
        aset.rename_class(0, 'neuron')
        cls = aset.get_class_by_id(0)
        assert cls.name == 'neuron'
        cell_boxes = [b for b in aset.boxes if b.class_id == 0]
        assert all(b.class_name == 'neuron' for b in cell_boxes)

    def test_get_class_map(self):
        aset = self._make_set()
        m = aset.get_class_map()
        assert m == {0: 'cell', 1: 'bg'}

    def test_get_class_by_id_missing(self):
        aset = self._make_set()
        assert aset.get_class_by_id(99) is None


class TestAnnotationYOLO:

    def _make_set(self):
        from flika.ai.annotations import AnnotationSet, AnnotationClass, BoundingBox
        classes = [AnnotationClass(id=0, name='cell', color=(255, 0, 0))]
        aset = AnnotationSet(image_width=100, image_height=100, n_frames=1,
                              classes=classes)
        aset.add_box(BoundingBox(x=10, y=20, width=30, height=40,
                                  class_id=0, class_name='cell'))
        return aset

    def test_yolo_export(self):
        aset = self._make_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = aset.to_yolo_format(tmpdir)
            assert os.path.isfile(yaml_path)
            label_file = os.path.join(tmpdir, 'labels', 'frame_000000.txt')
            assert os.path.isfile(label_file)
            content = Path(label_file).read_text().strip()
            parts = content.split()
            assert parts[0] == '0'  # class index
            # center x, center y, width, height (normalized)
            cx = float(parts[1])
            cy = float(parts[2])
            nw = float(parts[3])
            nh = float(parts[4])
            assert 0 < cx < 1
            assert 0 < cy < 1
            np.testing.assert_allclose(nw, 0.3, atol=0.01)
            np.testing.assert_allclose(nh, 0.4, atol=0.01)

    def test_yolo_roundtrip(self):
        pytest.importorskip('yaml')
        aset = self._make_set()
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = aset.to_yolo_format(tmpdir)
            from flika.ai.annotations import AnnotationSet
            loaded = AnnotationSet.from_yolo_format(
                os.path.join(tmpdir, 'labels'), yaml_path,
                image_width=100, image_height=100)
            assert len(loaded.classes) == 1
            assert loaded.classes[0].name == 'cell'
            assert len(loaded.boxes) == 1
            b = loaded.boxes[0]
            np.testing.assert_allclose(b.x, 10, atol=1)
            np.testing.assert_allclose(b.y, 20, atol=1)
            np.testing.assert_allclose(b.width, 30, atol=1)
            np.testing.assert_allclose(b.height, 40, atol=1)


class TestAnnotationCOCO:

    def _make_set(self):
        from flika.ai.annotations import AnnotationSet, AnnotationClass, BoundingBox
        classes = [
            AnnotationClass(id=0, name='cell', color=(255, 0, 0)),
            AnnotationClass(id=1, name='bg', color=(0, 255, 0)),
        ]
        aset = AnnotationSet(image_width=200, image_height=150, n_frames=2,
                              classes=classes)
        aset.add_box(BoundingBox(x=10, y=20, width=30, height=40,
                                  class_id=0, class_name='cell', frame=0))
        aset.add_box(BoundingBox(x=50, y=60, width=20, height=20,
                                  class_id=1, class_name='bg', frame=1,
                                  confidence=0.7))
        return aset

    def test_coco_export_structure(self):
        aset = self._make_set()
        coco = aset.to_coco_json()
        assert 'images' in coco
        assert 'annotations' in coco
        assert 'categories' in coco
        assert len(coco['images']) == 2
        assert len(coco['annotations']) == 2
        assert len(coco['categories']) == 2

    def test_coco_roundtrip(self):
        from flika.ai.annotations import AnnotationSet
        aset = self._make_set()
        coco = aset.to_coco_json()
        loaded = AnnotationSet.from_coco_json(coco)
        assert loaded.image_width == 200
        assert loaded.image_height == 150
        assert len(loaded.classes) == 2
        assert len(loaded.boxes) == 2
        # Verify the first box
        b0 = loaded.boxes[0]
        assert b0.x == 10
        assert b0.width == 30

    def test_coco_json_serializable(self):
        aset = self._make_set()
        coco = aset.to_coco_json()
        s = json.dumps(coco)  # should not raise
        assert len(s) > 0


class TestAnnotationVOC:

    def test_voc_xml_output(self):
        from flika.ai.annotations import AnnotationSet, AnnotationClass, BoundingBox
        from xml.etree import ElementTree as ET

        classes = [AnnotationClass(id=0, name='cell', color=(255, 0, 0))]
        aset = AnnotationSet(image_width=100, image_height=100, n_frames=1,
                              classes=classes)
        aset.add_box(BoundingBox(x=10, y=20, width=30, height=40,
                                  class_id=0, class_name='cell'))
        xml_str = aset.to_voc_xml('test.png', frame=0)
        root = ET.fromstring(xml_str)
        assert root.tag == 'annotation'
        assert root.find('filename').text == 'test.png'
        size = root.find('size')
        assert size.find('width').text == '100'
        obj = root.find('object')
        assert obj.find('name').text == 'cell'
        bndbox = obj.find('bndbox')
        assert bndbox.find('xmin').text == '10'
        assert bndbox.find('ymin').text == '20'
        assert bndbox.find('xmax').text == '40'  # 10+30
        assert bndbox.find('ymax').text == '60'  # 20+40


# ---------------------------------------------------------------------------
# Detection backend utilities
# ---------------------------------------------------------------------------

class TestDetectionConfig:

    def test_defaults(self):
        from flika.ai.detection_backend import DetectionConfig
        cfg = DetectionConfig()
        assert cfg.model_size == 'n'
        assert cfg.confidence == 0.25
        assert cfg.iou_threshold == 0.45
        assert cfg.image_size == 640
        assert cfg.use_tiling is False


class TestPrepareImage:

    def test_grayscale_float64(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.random.rand(64, 64)
        result = prepare_image_for_detection(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_grayscale_uint16(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
        result = prepare_image_for_detection(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_rgb_uint8_passthrough(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = prepare_image_for_detection(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_rgba_drops_alpha(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
        result = prepare_image_for_detection(img)
        assert result.shape == (64, 64, 3)

    def test_single_channel_3d(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.random.rand(64, 64, 1)
        result = prepare_image_for_detection(img)
        assert result.shape == (64, 64, 3)

    def test_empty_image(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.empty((0, 0))
        result = prepare_image_for_detection(img)
        assert result.dtype == np.uint8
        assert result.shape[-1] == 3

    def test_constant_image(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.full((32, 32), 42.0)
        result = prepare_image_for_detection(img)
        # p2 == p98 → zeros
        assert result.max() == 0

    def test_contrast_stretch(self):
        from flika.ai.detection_backend import prepare_image_for_detection
        img = np.zeros((100, 100), dtype=np.float64)
        img[40:60, 40:60] = 1000.0
        result = prepare_image_for_detection(img)
        # Bright region should be brighter than dark region
        bright = result[50, 50, 0]
        dark = result[0, 0, 0]
        assert bright > dark


class TestNMS:

    def test_empty(self):
        from flika.ai.detection_backend import UltralyticsBackend
        result = UltralyticsBackend._nms_boxes([], 0.5)
        assert result == []

    def test_no_overlap(self):
        from flika.ai.detection_backend import UltralyticsBackend
        from flika.ai.annotations import BoundingBox
        boxes = [
            BoundingBox(x=0, y=0, width=10, height=10, class_id=0,
                        class_name='a', confidence=0.9),
            BoundingBox(x=100, y=100, width=10, height=10, class_id=0,
                        class_name='a', confidence=0.8),
        ]
        result = UltralyticsBackend._nms_boxes(boxes, 0.5)
        assert len(result) == 2

    def test_full_overlap_suppresses(self):
        from flika.ai.detection_backend import UltralyticsBackend
        from flika.ai.annotations import BoundingBox
        # Two identical boxes — lower confidence should be suppressed
        boxes = [
            BoundingBox(x=10, y=10, width=20, height=20, class_id=0,
                        class_name='a', confidence=0.9),
            BoundingBox(x=10, y=10, width=20, height=20, class_id=0,
                        class_name='a', confidence=0.5),
        ]
        result = UltralyticsBackend._nms_boxes(boxes, 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_partial_overlap_kept(self):
        from flika.ai.detection_backend import UltralyticsBackend
        from flika.ai.annotations import BoundingBox
        # Two boxes with ~25% overlap (below 0.5 threshold)
        boxes = [
            BoundingBox(x=0, y=0, width=20, height=20, class_id=0,
                        class_name='a', confidence=0.9),
            BoundingBox(x=15, y=15, width=20, height=20, class_id=0,
                        class_name='a', confidence=0.8),
        ]
        result = UltralyticsBackend._nms_boxes(boxes, 0.5)
        assert len(result) == 2

    def test_ordering_by_confidence(self):
        from flika.ai.detection_backend import UltralyticsBackend
        from flika.ai.annotations import BoundingBox
        boxes = [
            BoundingBox(x=10, y=10, width=20, height=20, class_id=0,
                        class_name='a', confidence=0.3),
            BoundingBox(x=10, y=10, width=20, height=20, class_id=0,
                        class_name='a', confidence=0.95),
        ]
        result = UltralyticsBackend._nms_boxes(boxes, 0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.95


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class TestFeatureConfig:

    def test_defaults(self):
        from flika.ai.features import FeatureConfig
        cfg = FeatureConfig()
        assert cfg.gaussian_sigmas == (1.0, 2.0, 4.0)
        assert cfg.gabor_orientations == 4
        assert cfg.include_intensity is True


class TestFeatureExtractor:

    def test_feature_count(self):
        from flika.ai.features import FeatureExtractor
        fe = FeatureExtractor()
        names = fe.feature_names()
        assert fe.n_features() == len(names)
        # Default: 1+9+4+1+6+12+4 = 37
        assert fe.n_features() == 37

    def test_extract_shape(self):
        from flika.ai.features import FeatureExtractor
        fe = FeatureExtractor()
        img = np.random.rand(32, 32).astype(np.float32)
        features = fe.extract(img)
        assert features.shape == (32, 32, 37)
        assert features.dtype == np.float32

    def test_extract_no_nan(self):
        from flika.ai.features import FeatureExtractor
        fe = FeatureExtractor()
        img = np.random.rand(32, 32).astype(np.float32)
        features = fe.extract(img)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_extract_constant_image(self):
        from flika.ai.features import FeatureExtractor
        fe = FeatureExtractor()
        img = np.ones((16, 16), dtype=np.float32) * 42.0
        features = fe.extract(img)
        assert features.shape == (16, 16, 37)
        assert not np.isnan(features).any()

    def test_intensity_only(self):
        from flika.ai.features import FeatureExtractor, FeatureConfig
        cfg = FeatureConfig(
            include_intensity=True, include_gaussian=False,
            include_edges=False, include_lbp=False,
            include_hessian=False, include_gabor=False,
            include_extras=False,
        )
        fe = FeatureExtractor(cfg)
        assert fe.n_features() == 1
        assert fe.feature_names() == ['intensity']
        img = np.random.rand(16, 16)
        features = fe.extract(img)
        assert features.shape == (16, 16, 1)

    def test_rejects_3d_input(self):
        from flika.ai.features import FeatureExtractor
        fe = FeatureExtractor()
        img = np.random.rand(10, 16, 16)
        with pytest.raises(ValueError, match="2D"):
            fe.extract(img)

    def test_edge_features_on_gradient(self):
        from flika.ai.features import FeatureExtractor, FeatureConfig
        cfg = FeatureConfig(
            include_intensity=False, include_gaussian=False,
            include_edges=True, include_lbp=False,
            include_hessian=False, include_gabor=False,
            include_extras=False,
        )
        fe = FeatureExtractor(cfg)
        # Horizontal gradient → strong edge response
        img = np.tile(np.linspace(0, 1, 64), (64, 1)).astype(np.float32)
        features = fe.extract(img)
        # All 4 edge detectors should have nonzero response
        for i in range(4):
            assert features[:, :, i].max() > 0


# ---------------------------------------------------------------------------
# Gabor kernel
# ---------------------------------------------------------------------------

class TestGaborKernel:

    def test_kernel_shape_and_symmetry(self):
        from flika.ai.features import _make_gabor_kernel
        kernel = _make_gabor_kernel(frequency=0.1, theta=0.0)
        assert kernel.ndim == 2
        assert kernel.dtype == np.float32
        # Kernel should be roughly symmetric in size
        h, w = kernel.shape
        assert h >= 3
        assert w >= 3


# ---------------------------------------------------------------------------
# PSF Simulator
# ---------------------------------------------------------------------------

class TestPSFConfig:

    def test_defaults(self):
        from flika.ai.psf_simulator import PSFConfig
        cfg = PSFConfig()
        assert cfg.image_size == 128
        assert cfg.n_particles == 50
        assert cfg.psf_sigma == 1.5
        assert cfg.noise_type == 'poisson'


class TestPSFSimulator:

    def test_generate_frame(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        cfg = PSFConfig(image_size=64, n_particles=10, psf_sigma=1.5)
        sim = PSFSimulator(cfg)
        rng = np.random.default_rng(42)
        image, positions = sim.generate_frame(rng)
        assert image.shape == (64, 64)
        assert image.dtype == np.float32
        assert positions.shape == (10, 2)
        assert image.max() > image.min()

    def test_generate_stack(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        cfg = PSFConfig(image_size=32, n_particles=5, n_frames=4)
        sim = PSFSimulator(cfg)
        rng = np.random.default_rng(42)
        stack, all_pos = sim.generate_stack(rng)
        assert stack.shape == (4, 32, 32)
        assert len(all_pos) == 4
        for p in all_pos:
            assert p.shape == (5, 2)

    def test_density_map(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        cfg = PSFConfig(image_size=64, n_particles=5, psf_sigma=2.0)
        sim = PSFSimulator(cfg)
        rng = np.random.default_rng(42)
        _, positions = sim.generate_frame(rng)
        density = sim.positions_to_density_map(positions, (64, 64))
        assert density.shape == (64, 64)
        assert density.dtype == np.float32
        assert density.max() <= 1.0
        assert density.max() > 0.0

    def test_extract_coordinates(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        cfg = PSFConfig(image_size=128, n_particles=3, psf_sigma=2.0)
        sim = PSFSimulator(cfg)
        rng = np.random.default_rng(42)
        _, positions = sim.generate_frame(rng)
        density = sim.positions_to_density_map(positions, (128, 128), sigma=2.0)
        coords = PSFSimulator.extract_coordinates(density, threshold=0.1,
                                                    min_distance=5)
        # Should recover approximately the right number of particles
        assert coords.shape[1] == 2
        assert len(coords) >= 1  # at least some should be found

    def test_noise_types(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        rng = np.random.default_rng(42)
        for noise in ('poisson', 'gaussian', 'mixed'):
            cfg = PSFConfig(image_size=32, n_particles=3, noise_type=noise)
            sim = PSFSimulator(cfg)
            image, _ = sim.generate_frame(rng)
            assert image.shape == (32, 32)

    def test_reproducibility(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        cfg = PSFConfig(image_size=32, n_particles=5)
        sim = PSFSimulator(cfg)
        img1, pos1 = sim.generate_frame(np.random.default_rng(123))
        img2, pos2 = sim.generate_frame(np.random.default_rng(123))
        np.testing.assert_array_equal(pos1, pos2)


# ---------------------------------------------------------------------------
# Classifier backends
# ---------------------------------------------------------------------------

class TestClassifierConfig:

    def test_defaults(self):
        from flika.ai.classifier_backends import ClassifierConfig
        cfg = ClassifierConfig()
        assert cfg.backend == 'random_forest'
        assert cfg.n_estimators == 100
        assert cfg.max_depth == 10


class TestRandomForestBackend:

    @pytest.fixture
    def sklearn_available(self):
        return pytest.importorskip('sklearn')

    def test_train_predict_roundtrip(self, sklearn_available):
        from flika.ai.classifier_backends import RandomForestBackend, ClassifierConfig
        cfg = ClassifierConfig(n_estimators=10, max_depth=5)
        backend = RandomForestBackend(cfg)
        assert not backend.is_trained()

        rng = np.random.default_rng(42)
        n_samples = 200
        n_features = 5
        features = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        # Class 1: positive mean, Class 2: negative mean
        labels = np.ones(n_samples, dtype=np.int32)
        labels[features[:, 0] < 0] = 2

        backend.train(features, labels, n_classes=2)
        assert backend.is_trained()

        pred_labels, probs = backend.predict(features)
        assert pred_labels.shape == (n_samples,)
        assert probs.shape == (n_samples, 2)
        # Should get reasonable accuracy on the training data
        accuracy = np.mean(pred_labels == labels)
        assert accuracy > 0.8

    def test_save_load(self, sklearn_available, tmp_path):
        pytest.importorskip('joblib')
        from flika.ai.classifier_backends import RandomForestBackend, ClassifierConfig
        cfg = ClassifierConfig(n_estimators=5, max_depth=3)
        backend = RandomForestBackend(cfg)

        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 3)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(np.int32) + 1
        backend.train(features, labels, n_classes=2)

        model_path = str(tmp_path / 'test_rf.joblib')
        backend.save(model_path)
        assert os.path.isfile(model_path)

        backend2 = RandomForestBackend(cfg)
        backend2.load(model_path)
        assert backend2.is_trained()

        pred1, _ = backend.predict(features)
        pred2, _ = backend2.predict(features)
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_before_train_raises(self, sklearn_available):
        from flika.ai.classifier_backends import RandomForestBackend
        backend = RandomForestBackend()
        with pytest.raises(RuntimeError, match="not trained"):
            backend.predict(np.zeros((10, 5)))

    def test_create_backend_factory(self, sklearn_available):
        from flika.ai.classifier_backends import create_backend, ClassifierConfig
        cfg = ClassifierConfig(backend='random_forest')
        backend = create_backend(cfg)
        assert backend.name() == 'Random Forest'

    def test_create_backend_unknown(self):
        from flika.ai.classifier_backends import create_backend, ClassifierConfig
        cfg = ClassifierConfig(backend='nonexistent')
        with pytest.raises(ValueError, match="Unknown"):
            create_backend(cfg)


# ---------------------------------------------------------------------------
# Localizer config
# ---------------------------------------------------------------------------

class TestLocalizerConfig:

    def test_defaults(self):
        from flika.ai.localizer_backends import LocalizerConfig
        cfg = LocalizerConfig()
        assert cfg.backend == 'deepstorm'
        assert cfg.epochs == 100
        assert cfg.detection_threshold == 0.2
        assert cfg.psf_sigma == 1.5


# ---------------------------------------------------------------------------
# Denoiser utilities
# ---------------------------------------------------------------------------

class TestDenoiserUtils:

    def test_build_axes_string_2d(self):
        from flika.ai.denoiser import _build_axes_string
        assert _build_axes_string(2) == 'YX'

    def test_build_axes_string_3d_movie(self):
        from flika.ai.denoiser import _build_axes_string
        assert _build_axes_string(3, is_movie=True) == 'SYX'

    def test_build_axes_string_3d_volume(self):
        from flika.ai.denoiser import _build_axes_string
        assert _build_axes_string(3, is_movie=False) == 'ZYX'

    def test_build_axes_string_4d(self):
        from flika.ai.denoiser import _build_axes_string
        assert _build_axes_string(4) == 'SZYX'

    def test_build_axes_string_invalid(self):
        from flika.ai.denoiser import _build_axes_string
        with pytest.raises(ValueError, match="Unsupported ndim"):
            _build_axes_string(5)

    def test_detect_backend_returns_string(self):
        from flika.ai.denoiser import detect_denoiser_backend
        result = detect_denoiser_backend()
        assert result in ('careamics', 'n2v', 'csbdeep', 'none')


# ---------------------------------------------------------------------------
# Integration: FeatureExtractor + RandomForest pipeline
# ---------------------------------------------------------------------------

class TestFeatureClassifierPipeline:

    def test_end_to_end(self):
        sklearn = pytest.importorskip('sklearn')
        from flika.ai.features import FeatureExtractor, FeatureConfig
        from flika.ai.classifier_backends import RandomForestBackend, ClassifierConfig

        # Create a simple image with two regions
        img = np.zeros((64, 64), dtype=np.float32)
        img[:32, :] = 1.0  # top half bright
        img[32:, :] = 0.0  # bottom half dark

        # Extract features (intensity-only for speed)
        cfg = FeatureConfig(
            include_intensity=True, include_gaussian=True,
            include_edges=False, include_lbp=False,
            include_hessian=False, include_gabor=False,
            include_extras=False,
        )
        fe = FeatureExtractor(cfg)
        features = fe.extract(img)  # (64, 64, N)
        h, w, n_feat = features.shape

        # Sample labeled pixels: class 1 = bright, class 2 = dark
        rng = np.random.default_rng(42)
        n_train = 100
        bright_idx = rng.choice(32 * 64, n_train // 2, replace=False)
        dark_idx = rng.choice(32 * 64, n_train // 2, replace=False) + 32 * 64

        all_flat = features.reshape(-1, n_feat)
        train_idx = np.concatenate([bright_idx, dark_idx])
        train_features = all_flat[train_idx]
        train_labels = np.array([1] * (n_train // 2) + [2] * (n_train // 2),
                                 dtype=np.int32)

        # Train
        backend = RandomForestBackend(ClassifierConfig(n_estimators=10, max_depth=5))
        backend.train(train_features, train_labels, n_classes=2)

        # Predict on all pixels
        pred_labels, probs = backend.predict(all_flat)
        pred_map = pred_labels.reshape(h, w)

        # Top half should mostly be class 1, bottom half class 2
        top_accuracy = np.mean(pred_map[:32, :] == 1)
        bottom_accuracy = np.mean(pred_map[32:, :] == 2)
        assert top_accuracy > 0.8
        assert bottom_accuracy > 0.8


# ---------------------------------------------------------------------------
# Integration: PSFSimulator + density map round-trip
# ---------------------------------------------------------------------------

class TestPSFDensityRoundtrip:

    def test_positions_recoverable(self):
        from flika.ai.psf_simulator import PSFSimulator, PSFConfig
        cfg = PSFConfig(image_size=128, n_particles=3, psf_sigma=2.0,
                         background_mean=0, noise_type='gaussian',
                         read_noise_std=0, background_std=0)
        sim = PSFSimulator(cfg)
        rng = np.random.default_rng(42)
        _, positions = sim.generate_frame(rng)

        density = sim.positions_to_density_map(positions, (128, 128), sigma=2.0)
        recovered = PSFSimulator.extract_coordinates(density, threshold=0.1,
                                                      min_distance=5)
        # We should recover approximately the same number of particles
        assert abs(len(recovered) - len(positions)) <= 1

        # Each recovered position should be near a true position
        if len(recovered) > 0 and len(positions) > 0:
            from scipy.spatial.distance import cdist
            dists = cdist(recovered, positions)
            min_dists = dists.min(axis=1)
            assert np.all(min_dists < 5.0)  # within 5 pixels
