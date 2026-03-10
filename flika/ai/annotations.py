"""Reusable annotation data model for bounding boxes.

Pure Python data model (no Qt dependency). Used by the object detection
system and available for any future AI feature that needs spatial annotations.

Supports YOLO, COCO JSON, and Pascal VOC XML import/export formats.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np


@dataclass
class BoundingBox:
    """A single bounding box annotation."""
    x: float
    y: float
    width: float
    height: float
    class_id: int
    class_name: str
    confidence: float = 1.0  # 1.0 for manual, <1 for predictions
    frame: int = 0           # for stacks


@dataclass
class AnnotationClass:
    """A named class with display color."""
    id: int
    name: str
    color: Tuple[int, int, int]


class AnnotationSet:
    """Collection of bounding box annotations for an image or stack.

    Parameters
    ----------
    image_width : int
        Width of the image in pixels.
    image_height : int
        Height of the image in pixels.
    n_frames : int
        Number of frames (1 for single images).
    classes : list of AnnotationClass, optional
        Pre-defined classes.
    """

    def __init__(self, image_width: int, image_height: int, n_frames: int = 1,
                 classes: Optional[List[AnnotationClass]] = None):
        self.image_width = image_width
        self.image_height = image_height
        self.n_frames = n_frames
        self.classes: List[AnnotationClass] = list(classes) if classes else []
        self.boxes: List[BoundingBox] = []

    # -- Accessors -----------------------------------------------------------

    def get_frame_boxes(self, frame: int) -> List[BoundingBox]:
        """Return boxes belonging to *frame*."""
        return [b for b in self.boxes if b.frame == frame]

    def add_box(self, box: BoundingBox) -> None:
        self.boxes.append(box)

    def remove_box(self, box: BoundingBox) -> None:
        try:
            self.boxes.remove(box)
        except ValueError:
            pass

    def update_box(self, box: BoundingBox, **kwargs) -> None:
        """Update attributes of *box* in-place."""
        for k, v in kwargs.items():
            if hasattr(box, k):
                setattr(box, k, v)

    def clear_boxes(self, frame: Optional[int] = None) -> None:
        """Remove all boxes, or only those in *frame*."""
        if frame is None:
            self.boxes.clear()
        else:
            self.boxes = [b for b in self.boxes if b.frame != frame]

    # -- Class management ----------------------------------------------------

    def add_class(self, name: str, color: Tuple[int, int, int]) -> AnnotationClass:
        """Add a new class and return it."""
        new_id = max((c.id for c in self.classes), default=-1) + 1
        cls = AnnotationClass(id=new_id, name=name, color=color)
        self.classes.append(cls)
        return cls

    def remove_class(self, class_id: int) -> None:
        self.classes = [c for c in self.classes if c.id != class_id]
        self.boxes = [b for b in self.boxes if b.class_id != class_id]

    def rename_class(self, class_id: int, new_name: str) -> None:
        for c in self.classes:
            if c.id == class_id:
                c.name = new_name
        for b in self.boxes:
            if b.class_id == class_id:
                b.class_name = new_name

    def get_class_by_id(self, class_id: int) -> Optional[AnnotationClass]:
        for c in self.classes:
            if c.id == class_id:
                return c
        return None

    def get_class_map(self) -> Dict[int, str]:
        """Return {class_id: class_name} mapping."""
        return {c.id: c.name for c in self.classes}

    # -- YOLO format ---------------------------------------------------------

    def to_yolo_format(self, output_dir: str, image_paths: Optional[List[str]] = None) -> str:
        """Export annotations in YOLO format.

        Creates ``labels/`` directory with one ``.txt`` per frame/image,
        and a ``data.yaml`` describing the dataset.

        Parameters
        ----------
        output_dir : str
            Root directory for the dataset.
        image_paths : list of str, optional
            Paths to the source images (one per frame).

        Returns
        -------
        str
            Path to the generated ``data.yaml``.
        """
        output = Path(output_dir)
        label_dir = output / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)

        # Remap class ids to consecutive 0-based indices
        id_to_idx = {c.id: i for i, c in enumerate(self.classes)}

        for frame in range(self.n_frames):
            frame_boxes = self.get_frame_boxes(frame)
            fname = f'frame_{frame:06d}.txt'
            lines = []
            for b in frame_boxes:
                if b.class_id not in id_to_idx:
                    continue
                cx = (b.x + b.width / 2) / self.image_width
                cy = (b.y + b.height / 2) / self.image_height
                nw = b.width / self.image_width
                nh = b.height / self.image_height
                idx = id_to_idx[b.class_id]
                lines.append(f'{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
            (label_dir / fname).write_text('\n'.join(lines) + ('\n' if lines else ''))

        # data.yaml
        names = [c.name for c in self.classes]
        yaml_content = (
            f"path: {output.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: {len(self.classes)}\n"
            f"names: {names}\n"
        )
        yaml_path = output / 'data.yaml'
        yaml_path.write_text(yaml_content)

        # Copy/symlink images dir if paths provided
        if image_paths:
            img_dir = output / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(image_paths):
                dst = img_dir / f'frame_{i:06d}{Path(src).suffix}'
                if not dst.exists():
                    try:
                        os.symlink(os.path.abspath(src), str(dst))
                    except OSError:
                        import shutil
                        shutil.copy2(src, str(dst))

        return str(yaml_path)

    @classmethod
    def from_yolo_format(cls, label_dir: str, data_yaml: str,
                         image_width: int = 640, image_height: int = 640) -> 'AnnotationSet':
        """Import annotations from YOLO format."""
        import yaml as _yaml

        yaml_path = Path(data_yaml)
        with open(yaml_path) as f:
            data = _yaml.safe_load(f)

        names = data.get('names', [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]

        _DEFAULT_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0),
        ]
        classes = []
        for i, name in enumerate(names):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            classes.append(AnnotationClass(id=i, name=name, color=color))

        label_path = Path(label_dir)
        txt_files = sorted(label_path.glob('*.txt'))

        aset = cls(image_width=image_width, image_height=image_height,
                   n_frames=max(len(txt_files), 1), classes=classes)

        for frame, txt in enumerate(txt_files):
            for line in txt.read_text().strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                idx = int(parts[0])
                cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x = (cx - nw / 2) * image_width
                y = (cy - nh / 2) * image_height
                w = nw * image_width
                h = nh * image_height
                name = names[idx] if idx < len(names) else f'class_{idx}'
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                aset.add_box(BoundingBox(
                    x=x, y=y, width=w, height=h,
                    class_id=idx, class_name=name,
                    confidence=conf, frame=frame,
                ))
        return aset

    # -- COCO JSON format ----------------------------------------------------

    def to_coco_json(self, image_paths: Optional[List[str]] = None) -> dict:
        """Export annotations as a COCO-format dict."""
        images = []
        for frame in range(self.n_frames):
            fname = image_paths[frame] if image_paths and frame < len(image_paths) else f'frame_{frame:06d}.png'
            images.append({
                'id': frame,
                'file_name': os.path.basename(fname),
                'width': self.image_width,
                'height': self.image_height,
            })

        id_to_idx = {c.id: i for i, c in enumerate(self.classes)}
        categories = [{'id': i, 'name': c.name} for i, c in enumerate(self.classes)]

        annotations = []
        for ann_id, b in enumerate(self.boxes):
            if b.class_id not in id_to_idx:
                continue
            annotations.append({
                'id': ann_id,
                'image_id': b.frame,
                'category_id': id_to_idx[b.class_id],
                'bbox': [b.x, b.y, b.width, b.height],
                'area': b.width * b.height,
                'iscrowd': 0,
                'score': b.confidence,
            })

        return {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }

    @classmethod
    def from_coco_json(cls, coco_dict: dict) -> 'AnnotationSet':
        """Import annotations from a COCO-format dict."""
        images = coco_dict.get('images', [])
        categories = coco_dict.get('categories', [])
        anns = coco_dict.get('annotations', [])

        if images:
            w = images[0].get('width', 640)
            h = images[0].get('height', 640)
        else:
            w, h = 640, 640

        _DEFAULT_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        ]
        classes = []
        cat_id_map = {}
        for i, cat in enumerate(categories):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            classes.append(AnnotationClass(id=i, name=cat['name'], color=color))
            cat_id_map[cat['id']] = i

        aset = cls(image_width=w, image_height=h,
                   n_frames=max(len(images), 1), classes=classes)

        for ann in anns:
            cat_idx = cat_id_map.get(ann['category_id'], 0)
            bbox = ann['bbox']  # [x, y, w, h]
            cat_name = classes[cat_idx].name if cat_idx < len(classes) else f'class_{cat_idx}'
            aset.add_box(BoundingBox(
                x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3],
                class_id=cat_idx, class_name=cat_name,
                confidence=ann.get('score', 1.0),
                frame=ann.get('image_id', 0),
            ))
        return aset

    # -- Pascal VOC XML format -----------------------------------------------

    def to_voc_xml(self, image_path: str, frame: int = 0) -> str:
        """Export a single frame's annotations as Pascal VOC XML string."""
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = os.path.basename(image_path)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(self.image_width)
        ET.SubElement(size, 'height').text = str(self.image_height)
        ET.SubElement(size, 'depth').text = '3'

        for b in self.get_frame_boxes(frame):
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = b.class_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(b.x))
            ET.SubElement(bndbox, 'ymin').text = str(int(b.y))
            ET.SubElement(bndbox, 'xmax').text = str(int(b.x + b.width))
            ET.SubElement(bndbox, 'ymax').text = str(int(b.y + b.height))
            ET.SubElement(obj, 'confidence').text = f'{b.confidence:.4f}'

        ET.indent(root)
        return ET.tostring(root, encoding='unicode')

    # -- ROI interop ---------------------------------------------------------

    def to_rois(self, window, frame: Optional[int] = None):
        """Create flika rectangle ROIs from annotations.

        Parameters
        ----------
        window : flika.window.Window
            Target window.
        frame : int, optional
            Only create ROIs for this frame.  If None, creates for all frames.

        Returns
        -------
        list of ROI objects
        """
        from flika.roi import makeROI
        from qtpy import QtGui

        boxes = self.get_frame_boxes(frame) if frame is not None else self.boxes
        rois = []
        for b in boxes:
            cls = self.get_class_by_id(b.class_id)
            color = QtGui.QColor(*(cls.color if cls else (255, 255, 0)))
            roi = makeROI('rectangle',
                          [[b.y, b.x], [b.height, b.width]],
                          window=window, color=color)
            if roi is not None:
                roi.name = f"{b.class_name} ({b.confidence:.2f})"
                rois.append(roi)
        return rois

    @classmethod
    def from_rois(cls, rois, class_map: Dict[str, int],
                  image_width: int = 640, image_height: int = 640,
                  frame: int = 0) -> 'AnnotationSet':
        """Create an AnnotationSet from existing flika rectangle ROIs.

        Parameters
        ----------
        rois : list of ROI objects
            Flika ROIs (only rectangles are used).
        class_map : dict
            Mapping from class name to class id.
        """
        _DEFAULT_COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        classes = []
        for name, cid in class_map.items():
            color = _DEFAULT_COLORS[cid % len(_DEFAULT_COLORS)]
            classes.append(AnnotationClass(id=cid, name=name, color=color))

        aset = cls(image_width=image_width, image_height=image_height,
                   n_frames=1, classes=classes)

        for roi in rois:
            roi_type = type(roi).__name__
            if 'rectangle' not in roi_type.lower() and 'rect' not in roi_type.lower():
                continue
            try:
                pos = roi.pos()
                size = roi.size()
                x, y = pos.x(), pos.y()
                w, h = size.x(), size.y()
            except AttributeError:
                continue

            # Determine class from roi name or default to first class
            class_name = getattr(roi, 'name', '') or ''
            class_id = 0
            for cname, cid in class_map.items():
                if cname.lower() in class_name.lower():
                    class_id = cid
                    class_name = cname
                    break
            else:
                if classes:
                    class_id = classes[0].id
                    class_name = classes[0].name

            aset.add_box(BoundingBox(
                x=y, y=x, width=h, height=w,
                class_id=class_id, class_name=class_name,
                frame=frame,
            ))
        return aset
