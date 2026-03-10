"""AI-based segmentation wrappers for flika.

Each class wraps a well-known deep-learning segmentation tool as a
:class:`BaseProcess` so it appears in flika's menu system and records
commands like any other filter.

All heavy imports are deferred so these classes can be imported without
requiring the AI packages to be installed.
"""
from __future__ import annotations

import os

import numpy as np

from flika.logger import logger
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
import flika.global_vars as g


class CellposeSegmenter(BaseProcess):
    """cellpose_segment(diameter=30, model_type='cyto3', device='Auto', use_sam=False, keepSourceWindow=False)

    Wraps `cellpose <https://github.com/MouseLand/cellpose>`_ for cell
    segmentation.  Returns a label image.

    Parameters:
        diameter (float): Expected cell diameter in pixels (0 = auto).
        model_type (str): Cellpose model name (e.g. 'cyto3', 'nuclei').
        device (str): Compute device — 'Auto', 'CPU', 'CUDA', or 'MPS'.
        use_sam (bool): If True, refine Cellpose masks with SAM box prompts.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        diameter = SliderLabel(0)
        diameter.setRange(0, 200)
        diameter.setValue(30)
        model_type = ComboBox()
        model_type.addItems(['cyto3', 'cyto2', 'cyto', 'nuclei'])
        device = ComboBox()
        device.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        self.items.append({'name': 'diameter', 'string': 'Diameter', 'object': diameter})
        self.items.append({'name': 'model_type', 'string': 'Model', 'object': model_type})
        self.items.append({'name': 'device', 'string': 'Device', 'object': device})
        use_sam = CheckBox()
        self.items.append({'name': 'use_sam', 'string': 'Refine with SAM', 'object': use_sam})
        super().gui()

    def __call__(self, diameter=30, model_type='cyto3', device='Auto', use_sam=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        try:
            from cellpose import models
        except ImportError:
            g.alert("cellpose is not installed.  Install with:  pip install cellpose")
            return None

        from flika.utils.accel import get_torch_device
        torch_device = get_torch_device(device)

        logger.info("Running Cellpose segmentation (model=%s, diameter=%s, device=%s)",
                     model_type, diameter, torch_device)
        # Cellpose 4.x uses CellposeModel; fall back to Cellpose for older versions
        ModelClass = getattr(models, 'CellposeModel', None) or models.Cellpose
        use_gpu = torch_device is not None and str(torch_device) != 'cpu'
        model_kwargs = {'model_type': model_type}
        if use_gpu:
            model_kwargs['gpu'] = True
            model_kwargs['device'] = torch_device
        model = ModelClass(**model_kwargs)

        if self.tif.ndim == 2:
            result = model.eval(self.tif, diameter=diameter or None)
            masks = result[0]
        elif self.tif.ndim == 3:
            masks = np.zeros_like(self.tif, dtype=np.int32)
            for i in range(self.tif.shape[0]):
                result = model.eval(self.tif[i], diameter=diameter or None)
                masks[i] = result[0]
        else:
            g.alert("Cellpose segmentation requires 2-D or 3-D data.")
            return None

        if use_sam:
            masks = self._refine_with_sam(masks, device)

        self.newtif = masks.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - Cellpose'
        return self.end()

    def _refine_with_sam(self, masks, device):
        """Refine Cellpose masks using SAM box prompts."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            logger.warning("segment_anything not installed; skipping SAM refinement")
            return masks

        sam_path = os.path.join(os.path.expanduser('~'), '.FLIKA', 'models', 'sam_vit_b_01ec64.pth')
        if not os.path.exists(sam_path):
            logger.warning("SAM checkpoint not found at %s; skipping refinement", sam_path)
            return masks

        from flika.utils.accel import get_torch_device
        torch_device = get_torch_device(device)
        dev_str = str(torch_device) if torch_device is not None else 'cpu'

        sam = sam_model_registry['vit_b'](checkpoint=sam_path)
        sam.to(dev_str)
        predictor = SamPredictor(sam)

        def _refine_2d(image, mask_2d):
            img = image
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.dtype != np.uint8:
                mn, mx = img.min(), img.max()
                if mx > mn:
                    img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            predictor.set_image(img)

            refined = np.zeros_like(mask_2d)
            for label_id in np.unique(mask_2d):
                if label_id == 0:
                    continue
                ys, xs = np.where(mask_2d == label_id)
                box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                pred_masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                best = pred_masks[np.argmax(scores)]
                refined[best] = label_id
            return refined

        if masks.ndim == 2:
            return _refine_2d(self.tif, masks)
        elif masks.ndim == 3:
            for i in range(masks.shape[0]):
                masks[i] = _refine_2d(self.tif[i], masks[i])
            return masks
        return masks


class StarDistSegmenter(BaseProcess):
    """stardist_segment(model_name='2D_versatile_fluo', device='Auto', keepSourceWindow=False)

    Wraps `StarDist <https://github.com/stardist/stardist>`_ for
    star-convex polygon segmentation.

    Parameters:
        model_name (str): Pre-trained model name.
        device (str): Compute device — 'Auto', 'CPU', 'CUDA', or 'MPS'.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        model_name = ComboBox()
        model_name.addItems(['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'])
        device = ComboBox()
        device.addItems(['Auto', 'CPU', 'CUDA', 'MPS'])
        self.items.append({'name': 'model_name', 'string': 'Model', 'object': model_name})
        self.items.append({'name': 'device', 'string': 'Device', 'object': device})
        super().gui()

    def __call__(self, model_name='2D_versatile_fluo', device='Auto', keepSourceWindow=False):
        self.start(keepSourceWindow)
        try:
            from stardist.models import StarDist2D
        except ImportError:
            g.alert("stardist is not installed.  Install with:  pip install stardist")
            return None

        logger.info("Running StarDist segmentation (model=%s, device=%s)", model_name, device)

        # StarDist uses TensorFlow — control GPU via env var
        if device == 'CPU':
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        model = StarDist2D.from_pretrained(model_name)

        if self.tif.ndim == 2:
            labels, _ = model.predict_instances(self.tif)
        elif self.tif.ndim == 3:
            labels = np.zeros_like(self.tif, dtype=np.int32)
            for i in range(self.tif.shape[0]):
                lab, _ = model.predict_instances(self.tif[i])
                labels[i] = lab
        else:
            g.alert("StarDist segmentation requires 2-D or 3-D data.")
            return None

        self.newtif = labels.astype(g.settings['internal_data_type'])
        self.newname = self.oldname + ' - StarDist'
        return self.end()


class AIDenoiser:
    """AI Denoiser — opens the full training/prediction dialog.

    Supports N2V (self-supervised) and CARE (supervised) denoising via
    CAREamics (PyTorch) or n2v/csbdeep (TensorFlow) backends.
    """

    def gui(self):
        from flika.ai.training_dialog import DenoiserDialog
        dlg = DenoiserDialog(parent=g.m)
        dlg.show()
        g.dialogs.append(dlg)


class AIPixelClassifier:
    """AI Pixel Classifier — opens the interactive classification dialog."""

    def gui(self):
        from flika.ai.classifier_dialog import PixelClassifierDialog
        dlg = PixelClassifierDialog(parent=g.m)
        dlg.show()
        g.dialogs.append(dlg)


class AIParticleLocalizer:
    """AI Particle Localizer — opens the DeepSTORM localization dialog."""

    def gui(self):
        from flika.ai.localizer_dialog import ParticleLocalizerDialog
        dlg = ParticleLocalizerDialog(parent=g.m)
        dlg.show()
        g.dialogs.append(dlg)


class BioImageIORunner:
    """BioImage.IO Model Zoo — opens the browser dialog."""

    def gui(self):
        from flika.ai.model_zoo import ModelZooBrowser
        dlg = ModelZooBrowser(parent=g.m)
        dlg.show()
        g.dialogs.append(dlg)


class AISAMSegmenter:
    """SAM Interactive Segmentation — opens the point/box prompt dialog."""

    def gui(self):
        from flika.ai.sam_dialog import SAMSegmentationDialog
        dlg = SAMSegmentationDialog(parent=g.m)
        dlg.show()
        g.dialogs.append(dlg)


class AIObjectDetector:
    """AI Object Detection — opens detection/annotation/training dialog.

    Supports pretrained YOLO models, custom training, and fine-tuning
    via the Ultralytics backend.
    """

    def gui(self):
        from flika.ai.detection_dialog import ObjectDetectionDialog
        dlg = ObjectDetectionDialog(parent=g.m)
        dlg.show()
        g.dialogs.append(dlg)

    def predict(self, model='yolov8n.pt', confidence=0.25, iou=0.45,
                device='Auto', create_rois=True):
        """Non-GUI prediction for scripting/macros."""
        from flika.ai.detection_backend import DetectionConfig, UltralyticsBackend
        if g.win is None:
            g.alert('No window selected.')
            return []
        backend = UltralyticsBackend()
        config = DetectionConfig(model_path=model, confidence=confidence,
                                 iou_threshold=iou, device=device)
        boxes = backend.predict(g.win.image, config)
        if create_rois:
            from flika.ai.annotations import AnnotationSet, AnnotationClass
            aset = AnnotationSet(
                image_width=g.win.image.shape[-1],
                image_height=g.win.image.shape[-2],
            )
            for b in boxes:
                aset.add_box(b)
            aset.to_rois(g.win)
        return boxes


# Singleton instances
cellpose_segment = CellposeSegmenter()
stardist_segment = StarDistSegmenter()
ai_denoise = AIDenoiser()
ai_classify = AIPixelClassifier()
ai_localize = AIParticleLocalizer()
ai_model_zoo = BioImageIORunner()
ai_sam = AISAMSegmenter()
ai_detect = AIObjectDetector()
