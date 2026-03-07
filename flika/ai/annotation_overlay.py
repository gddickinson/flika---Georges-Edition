"""Interactive bounding box annotation overlay for pyqtgraph ImageView.

Provides drawing, selection, moving, and resizing of bounding boxes on
an image, with per-frame filtering for stacks.  Follows the pattern of
``PaintOverlay`` in ``classifier_dialog.py`` but for bounding boxes.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Signal
import pyqtgraph as pg

from .annotations import AnnotationClass, AnnotationSet, BoundingBox


class InteractionMode(Enum):
    DRAW = auto()
    SELECT = auto()
    DELETE = auto()


_HANDLE_SIZE = 6


class _BoxItem(QtWidgets.QGraphicsRectItem):
    """Visual representation of a single BoundingBox."""

    def __init__(self, box: BoundingBox, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self.box = box
        self._color = color
        self._selected = False
        self._update_from_box()
        self.setAcceptHoverEvents(True)

    def _update_from_box(self):
        self.setRect(self.box.x, self.box.y, self.box.width, self.box.height)
        self._refresh_pen()

    def _refresh_pen(self):
        pen = QtGui.QPen(self._color)
        pen.setWidth(2 if self._selected else 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QtGui.QColor(self._color)
        fill.setAlpha(40 if self._selected else 20)
        self.setBrush(fill)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._refresh_pen()

    def set_color(self, color: QtGui.QColor):
        self._color = color
        self._refresh_pen()

    def sync_to_box(self):
        """Write current rect geometry back to the BoundingBox dataclass."""
        r = self.rect()
        self.box.x = r.x()
        self.box.y = r.y()
        self.box.width = r.width()
        self.box.height = r.height()


class _LabelItem(QtWidgets.QGraphicsSimpleTextItem):
    """Class label text shown above a box."""

    def __init__(self, text: str, color: QtGui.QColor, parent=None):
        super().__init__(text, parent)
        self.setBrush(color)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        self.setFont(font)
        flags = self.flags()
        self.setFlags(flags | QtWidgets.QGraphicsItem.ItemIgnoresTransformations)


class BoxAnnotationOverlay(pg.GraphicsObject):
    """Overlay for drawing / editing bounding boxes on a pg.ImageView.

    Signals
    -------
    sigBoxCreated(BoundingBox)
    sigBoxRemoved(BoundingBox)
    sigBoxChanged(BoundingBox)
    sigSelectionChanged(BoundingBox or None)
    """

    sigBoxCreated = Signal(object)
    sigBoxRemoved = Signal(object)
    sigBoxChanged = Signal(object)
    sigSelectionChanged = Signal(object)

    def __init__(self, annotation_set: AnnotationSet, parent=None):
        super().__init__(parent)
        self.annotation_set = annotation_set
        self._mode = InteractionMode.DRAW
        self._current_frame = 0
        self._active_class_id = 0

        # Visual items keyed by BoundingBox id
        self._box_items: dict[int, _BoxItem] = {}
        self._label_items: dict[int, _LabelItem] = {}

        # Drawing state
        self._drawing = False
        self._draw_start: Optional[QtCore.QPointF] = None
        self._draw_rect_item: Optional[QtWidgets.QGraphicsRectItem] = None

        # Selection / move / resize state
        self._selected_box: Optional[BoundingBox] = None
        self._dragging = False
        self._drag_offset = QtCore.QPointF()
        self._resizing = False
        self._resize_handle = ''  # 'tl','tr','bl','br'

        self.setZValue(20)
        self._rebuild_items()

    # -- Public API ----------------------------------------------------------

    @property
    def mode(self) -> InteractionMode:
        return self._mode

    @mode.setter
    def mode(self, m: InteractionMode):
        self._mode = m
        self._cancel_draw()
        self.select_box(None)

    def set_frame(self, idx: int):
        """Show annotations for frame *idx*."""
        idx = max(0, min(idx, self.annotation_set.n_frames - 1))
        if idx != self._current_frame:
            self._current_frame = idx
            self._rebuild_items()

    def set_active_class(self, class_id: int):
        self._active_class_id = class_id

    def select_box(self, box: Optional[BoundingBox]):
        if self._selected_box is not None:
            item = self._box_items.get(id(self._selected_box))
            if item:
                item.set_selected(False)
        self._selected_box = box
        if box is not None:
            item = self._box_items.get(id(box))
            if item:
                item.set_selected(True)
        self.sigSelectionChanged.emit(box)

    def refresh(self):
        """Rebuild all visual items from the annotation set."""
        self._rebuild_items()

    def delete_selected(self):
        if self._selected_box is not None:
            box = self._selected_box
            self.select_box(None)
            self.annotation_set.remove_box(box)
            self._remove_item(box)
            self.sigBoxRemoved.emit(box)

    # -- Internal ------------------------------------------------------------

    def _get_color(self, class_id: int) -> QtGui.QColor:
        cls = self.annotation_set.get_class_by_id(class_id)
        if cls:
            return QtGui.QColor(*cls.color)
        return QtGui.QColor(255, 255, 0)

    def _rebuild_items(self):
        """Remove all items and recreate for current frame."""
        scene = self.scene()
        for bid, item in self._box_items.items():
            if scene and item.scene() is scene:
                scene.removeItem(item)
        for bid, item in self._label_items.items():
            if scene and item.scene() is scene:
                scene.removeItem(item)
        self._box_items.clear()
        self._label_items.clear()

        boxes = self.annotation_set.get_frame_boxes(self._current_frame)
        for box in boxes:
            self._add_item(box)

    def _add_item(self, box: BoundingBox):
        scene = self.scene()
        if scene is None:
            return
        color = self._get_color(box.class_id)
        item = _BoxItem(box, color)
        scene.addItem(item)
        self._box_items[id(box)] = item

        label_text = f'{box.class_name}'
        if box.confidence < 1.0:
            label_text += f' {box.confidence:.2f}'
        label = _LabelItem(label_text, color)
        label.setPos(box.x, box.y - 2)
        scene.addItem(label)
        self._label_items[id(box)] = label

    def _remove_item(self, box: BoundingBox):
        scene = self.scene()
        bid = id(box)
        item = self._box_items.pop(bid, None)
        if item and scene and item.scene() is scene:
            scene.removeItem(item)
        label = self._label_items.pop(bid, None)
        if label and scene and label.scene() is scene:
            scene.removeItem(label)

    def _update_item(self, box: BoundingBox):
        bid = id(box)
        item = self._box_items.get(bid)
        if item:
            item._update_from_box()
        label = self._label_items.get(bid)
        if label:
            label.setPos(box.x, box.y - 2)
            label_text = f'{box.class_name}'
            if box.confidence < 1.0:
                label_text += f' {box.confidence:.2f}'
            label.setText(label_text)

    def _cancel_draw(self):
        if self._draw_rect_item:
            scene = self.scene()
            if scene and self._draw_rect_item.scene() is scene:
                scene.removeItem(self._draw_rect_item)
            self._draw_rect_item = None
        self._drawing = False
        self._draw_start = None

    def _box_at(self, pos: QtCore.QPointF) -> Optional[BoundingBox]:
        """Find the topmost box under *pos* in the current frame."""
        px, py = pos.x(), pos.y()
        for box in reversed(self.annotation_set.get_frame_boxes(self._current_frame)):
            if (box.x <= px <= box.x + box.width and
                    box.y <= py <= box.y + box.height):
                return box
        return None

    # -- Qt event handling ---------------------------------------------------

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.annotation_set.image_width,
                             self.annotation_set.image_height)

    def paint(self, p, *args):
        pass  # child items do the painting

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        pos = ev.pos()

        if self._mode == InteractionMode.DRAW:
            self._drawing = True
            self._draw_start = pos
            pen = QtGui.QPen(self._get_color(self._active_class_id))
            pen.setWidth(1)
            pen.setCosmetic(True)
            self._draw_rect_item = QtWidgets.QGraphicsRectItem(self)
            self._draw_rect_item.setPen(pen)
            fill = QtGui.QColor(self._get_color(self._active_class_id))
            fill.setAlpha(30)
            self._draw_rect_item.setBrush(fill)
            ev.accept()

        elif self._mode == InteractionMode.SELECT:
            box = self._box_at(pos)
            self.select_box(box)
            if box is not None:
                item = self._box_items.get(id(box))
                if item:
                    self._dragging = True
                    self._drag_offset = pos - QtCore.QPointF(box.x, box.y)
            ev.accept()

        elif self._mode == InteractionMode.DELETE:
            box = self._box_at(pos)
            if box is not None:
                self.annotation_set.remove_box(box)
                self._remove_item(box)
                self.sigBoxRemoved.emit(box)
            ev.accept()

    def mouseMoveEvent(self, ev):
        pos = ev.pos()

        if self._drawing and self._draw_start is not None:
            x0, y0 = self._draw_start.x(), self._draw_start.y()
            x1, y1 = pos.x(), pos.y()
            rect = QtCore.QRectF(min(x0, x1), min(y0, y1),
                                 abs(x1 - x0), abs(y1 - y0))
            if self._draw_rect_item:
                self._draw_rect_item.setRect(rect)
            ev.accept()

        elif self._dragging and self._selected_box is not None:
            new_pos = pos - self._drag_offset
            self._selected_box.x = new_pos.x()
            self._selected_box.y = new_pos.y()
            self._update_item(self._selected_box)
            ev.accept()

        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if self._drawing and self._draw_start is not None:
            pos = ev.pos()
            x0, y0 = self._draw_start.x(), self._draw_start.y()
            x1, y1 = pos.x(), pos.y()
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w > 2 and h > 2:  # minimum box size
                cls = self.annotation_set.get_class_by_id(self._active_class_id)
                class_name = cls.name if cls else f'class_{self._active_class_id}'
                box = BoundingBox(
                    x=min(x0, x1), y=min(y0, y1),
                    width=w, height=h,
                    class_id=self._active_class_id,
                    class_name=class_name,
                    confidence=1.0,
                    frame=self._current_frame,
                )
                self.annotation_set.add_box(box)
                self._add_item(box)
                self.sigBoxCreated.emit(box)
            self._cancel_draw()
            ev.accept()

        elif self._dragging:
            self._dragging = False
            if self._selected_box:
                self.sigBoxChanged.emit(self._selected_box)
            ev.accept()

        else:
            ev.ignore()

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.delete_selected()
            ev.accept()
        else:
            ev.ignore()
