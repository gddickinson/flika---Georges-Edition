# -*- coding: utf-8 -*-
"""Interactive binary mask painting tool.

Provides a dockable control panel and mouse/keyboard-driven drawing and
erasing on a mask window.
"""
import numpy as np
from skimage.draw import disk, line
from qtpy import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess_noPriorWindow, WindowSelector, SliderLabel
from ..logger import logger

__all__ = ['mask_editor']


class MaskEditorPanel(QtWidgets.QDockWidget):
    """Dockable control panel for the mask editor."""

    _instance = None

    @classmethod
    def instance(cls, editor, parent=None):
        if cls._instance is None or cls._instance._destroyed:
            cls._instance = cls(editor, parent)
        else:
            cls._instance._editor = editor
        return cls._instance

    def __init__(self, editor, parent=None):
        super().__init__("Mask Editor", parent)
        self._destroyed = False
        self._editor = editor
        self.setObjectName("MaskEditorDock")
        self.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # Window info label
        self.info_label = QtWidgets.QLabel("No window selected")
        layout.addWidget(self.info_label)

        # Brush size slider
        brush_layout = QtWidgets.QHBoxLayout()
        brush_layout.addWidget(QtWidgets.QLabel("Brush size:"))
        self.brush_slider = SliderLabel(0)
        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(5)
        brush_layout.addWidget(self.brush_slider)
        layout.addLayout(brush_layout)

        # Fill / Clear buttons
        fill_clear_layout = QtWidgets.QHBoxLayout()
        self.fill_btn = QtWidgets.QPushButton("Fill (255)")
        self.clear_btn = QtWidgets.QPushButton("Clear (0)")
        fill_clear_layout.addWidget(self.fill_btn)
        fill_clear_layout.addWidget(self.clear_btn)
        layout.addLayout(fill_clear_layout)

        # Undo / Redo buttons
        undo_redo_layout = QtWidgets.QHBoxLayout()
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.redo_btn = QtWidgets.QPushButton("Redo")
        undo_redo_layout.addWidget(self.undo_btn)
        undo_redo_layout.addWidget(self.redo_btn)
        layout.addLayout(undo_redo_layout)

        # Copy buttons
        copy_layout = QtWidgets.QHBoxLayout()
        self.copy_next_btn = QtWidgets.QPushButton("Copy to Next")
        self.copy_prev_btn = QtWidgets.QPushButton("Copy to Prev")
        self.copy_all_btn = QtWidgets.QPushButton("Copy to All")
        copy_layout.addWidget(self.copy_next_btn)
        copy_layout.addWidget(self.copy_prev_btn)
        copy_layout.addWidget(self.copy_all_btn)
        layout.addLayout(copy_layout)

        # Save button
        self.save_btn = QtWidgets.QPushButton("Save as TIFF")
        layout.addWidget(self.save_btn)

        # Instructions
        instructions = QtWidgets.QLabel("D=Draw, E=Erase, Arrows=Navigate")
        instructions.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(instructions)

        layout.addStretch()
        self.setWidget(container)

        # Connect buttons
        self.fill_btn.clicked.connect(lambda: self._editor.fill_frame())
        self.clear_btn.clicked.connect(lambda: self._editor.clear_frame())
        self.undo_btn.clicked.connect(lambda: self._editor.undo())
        self.redo_btn.clicked.connect(lambda: self._editor.redo())
        self.copy_next_btn.clicked.connect(lambda: self._editor.copy_to_next())
        self.copy_prev_btn.clicked.connect(lambda: self._editor.copy_to_prev())
        self.copy_all_btn.clicked.connect(lambda: self._editor.copy_to_all())
        self.save_btn.clicked.connect(lambda: self._editor.save_mask())

    def update_info(self, window):
        """Update the info label with window details."""
        if window is not None:
            name = window.name
            shape = window.image.shape
            self.info_label.setText(f"{name}  {shape}")
        else:
            self.info_label.setText("No window selected")

    def closeEvent(self, event):
        self._destroyed = True
        self._editor._disconnect_signals()
        MaskEditorPanel._instance = None
        event.accept()


class Mask_Editor(BaseProcess_noPriorWindow):
    """mask_editor(mask_window, keepSourceWindow=True)

    Opens an interactive binary mask painting tool. Hold D to draw (set
    pixels to 255) or E to erase (set pixels to 0) while moving the mouse
    over the selected mask window. Use arrow keys to navigate frames.

    Parameters
    ----------
    mask_window : Window
        The window containing the mask image to edit.
    keepSourceWindow : bool
        Unused; kept for BaseProcess compatibility.
    """

    def __init__(self):
        super().__init__()
        self._drawing = False
        self._erasing = False
        self._mask_window = None
        self._panel = None
        self._undo_stack = []
        self._redo_stack = []
        self._stroke_saved = False
        self._connected = False

    def gui(self):
        self.gui_reset()
        mask_window = WindowSelector()
        self.items.append({'name': 'mask_window', 'string': 'Mask Window', 'object': mask_window})
        super().gui()

    def __call__(self, mask_window=None, keepSourceWindow=True):
        if mask_window is None:
            if g.win is None:
                g.alert("No window selected for mask editing.")
                return
            mask_window = g.win

        self._mask_window = mask_window
        self._drawing = False
        self._erasing = False
        self._undo_stack = []
        self._redo_stack = []
        self._stroke_saved = False

        # Create or reuse the panel
        self._panel = MaskEditorPanel.instance(self, g.m)
        self._panel.update_info(mask_window)
        g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._panel)
        self._panel.show()
        self._panel.raise_()

        self._connect_signals(mask_window)
        g.status_msg('Mask Editor opened. D=Draw, E=Erase.')

    def _connect_signals(self, window):
        """Connect mouse and keyboard signals to the mask window."""
        self._disconnect_signals()
        window.imageview.scene.sigMouseMoved.connect(self.mouse_moved)
        window.installEventFilter(self)
        self._connected = True

    def _disconnect_signals(self):
        """Disconnect all editor signals from the mask window."""
        if self._connected and self._mask_window is not None:
            try:
                self._mask_window.imageview.scene.sigMouseMoved.disconnect(self.mouse_moved)
            except (TypeError, RuntimeError):
                pass
            try:
                self._mask_window.removeEventFilter(self)
            except (TypeError, RuntimeError):
                pass
            self._connected = False

    def eventFilter(self, obj, event):
        """Filter key press/release events from the mask window."""
        if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
            self.key_pressed(event)
            return True
        elif event.type() == QtCore.QEvent.KeyRelease and not event.isAutoRepeat():
            self.key_released(event)
            return True
        return False

    def key_pressed(self, event):
        """Handle key press events for drawing, erasing, and navigation."""
        key = event.key()
        if key == QtCore.Qt.Key_D:
            self._drawing = True
            self._erasing = False
            self._stroke_saved = False
            g.status_msg('Mask Editor: DRAW mode')
        elif key == QtCore.Qt.Key_E:
            self._erasing = True
            self._drawing = False
            self._stroke_saved = False
            g.status_msg('Mask Editor: ERASE mode')
        elif key == QtCore.Qt.Key_Left:
            self._navigate(-1)
        elif key == QtCore.Qt.Key_Right:
            self._navigate(1)

    def key_released(self, event):
        """Handle key release to stop drawing or erasing."""
        key = event.key()
        if key == QtCore.Qt.Key_D:
            self._drawing = False
            self._stroke_saved = False
            g.status_msg('Mask Editor ready.')
        elif key == QtCore.Qt.Key_E:
            self._erasing = False
            self._stroke_saved = False
            g.status_msg('Mask Editor ready.')

    def _navigate(self, delta):
        """Navigate frames by *delta* steps."""
        w = self._mask_window
        if w is None or w.image.ndim < 3:
            return
        new_idx = int(w.currentIndex) + delta
        if 0 <= new_idx < w.image.shape[0]:
            w.imageview.setCurrentIndex(new_idx)

    def mouse_moved(self, pos):
        """Handle mouse movement — draw or erase at the cursor position."""
        if not (self._drawing or self._erasing):
            return
        w = self._mask_window
        if w is None:
            return
        point = w.imageview.getImageItem().mapFromScene(pos)
        x = int(point.x())
        y = int(point.y())
        img = w.image
        # Determine frame shape depending on dimensionality
        if img.ndim == 2:
            rows, cols = img.shape
        elif img.ndim >= 3:
            rows, cols = img.shape[-2], img.shape[-1]
        else:
            return
        if x < 0 or y < 0 or x >= rows or y >= cols:
            return
        self.draw_point(x, y)

    def draw_point(self, x, y):
        """Draw or erase a disk at (*x*, *y*) using the current brush size."""
        w = self._mask_window
        if w is None:
            return

        # Save undo state once per stroke
        if not self._stroke_saved:
            self.save_undo_state()
            self._stroke_saved = True

        radius = max(1, self._panel.brush_slider.value() // 2)
        img = w.image

        if img.ndim == 2:
            frame = img
        elif img.ndim >= 3:
            idx = int(w.currentIndex)
            frame = img[idx]
        else:
            return

        rr, cc = disk((x, y), radius, shape=frame.shape)
        if self._drawing:
            frame[rr, cc] = 255
        elif self._erasing:
            frame[rr, cc] = 0

        self.update_image()

    def update_image(self):
        """Refresh the image display while preserving the current view range."""
        w = self._mask_window
        if w is None:
            return
        view = w.imageview.getView()
        view_range = view.viewRange()
        current_idx = int(w.currentIndex) if w.image.ndim >= 3 else 0
        w.imageview.setImage(w.image, autoLevels=False)
        if w.image.ndim >= 3:
            w.imageview.setCurrentIndex(current_idx)
        view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    def save_undo_state(self):
        """Push the current frame data onto the undo stack (max 20)."""
        w = self._mask_window
        if w is None:
            return
        if w.image.ndim == 2:
            frame_idx = 0
            frame_copy = w.image.copy()
        else:
            frame_idx = int(w.currentIndex)
            frame_copy = w.image[frame_idx].copy()
        self._undo_stack.append((frame_idx, frame_copy))
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        """Undo the last paint stroke."""
        w = self._mask_window
        if w is None or not self._undo_stack:
            return
        frame_idx, frame_data = self._undo_stack.pop()
        # Save current state to redo
        if w.image.ndim == 2:
            self._redo_stack.append((0, w.image.copy()))
            w.image[:] = frame_data
        else:
            self._redo_stack.append((frame_idx, w.image[frame_idx].copy()))
            w.image[frame_idx] = frame_data
        self.update_image()

    def redo(self):
        """Redo the last undone paint stroke."""
        w = self._mask_window
        if w is None or not self._redo_stack:
            return
        frame_idx, frame_data = self._redo_stack.pop()
        # Save current state to undo
        if w.image.ndim == 2:
            self._undo_stack.append((0, w.image.copy()))
            w.image[:] = frame_data
        else:
            self._undo_stack.append((frame_idx, w.image[frame_idx].copy()))
            w.image[frame_idx] = frame_data
        self.update_image()

    def copy_to_next(self):
        """Copy the current frame to the next frame."""
        w = self._mask_window
        if w is None or w.image.ndim < 3:
            return
        idx = int(w.currentIndex)
        if idx + 1 < w.image.shape[0]:
            self.save_undo_state()
            w.image[idx + 1] = w.image[idx].copy()
            self._navigate(1)
            self.update_image()

    def copy_to_prev(self):
        """Copy the current frame to the previous frame."""
        w = self._mask_window
        if w is None or w.image.ndim < 3:
            return
        idx = int(w.currentIndex)
        if idx - 1 >= 0:
            self.save_undo_state()
            w.image[idx - 1] = w.image[idx].copy()
            self._navigate(-1)
            self.update_image()

    def copy_to_all(self):
        """Copy the current frame to all other frames."""
        w = self._mask_window
        if w is None or w.image.ndim < 3:
            return
        idx = int(w.currentIndex)
        self.save_undo_state()
        frame = w.image[idx].copy()
        for i in range(w.image.shape[0]):
            if i != idx:
                w.image[i] = frame.copy()
        self.update_image()

    def fill_frame(self):
        """Set all pixels in the current frame to 255."""
        w = self._mask_window
        if w is None:
            return
        self.save_undo_state()
        if w.image.ndim == 2:
            w.image[:] = 255
        else:
            idx = int(w.currentIndex)
            w.image[idx][:] = 255
        self.update_image()

    def clear_frame(self):
        """Set all pixels in the current frame to 0."""
        w = self._mask_window
        if w is None:
            return
        self.save_undo_state()
        if w.image.ndim == 2:
            w.image[:] = 0
        else:
            idx = int(w.currentIndex)
            w.image[idx][:] = 0
        self.update_image()

    def save_mask(self):
        """Save the mask image as a TIFF file."""
        w = self._mask_window
        if w is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            g.m, "Save Mask as TIFF", "", "TIFF Files (*.tif *.tiff)")
        if not path:
            return
        try:
            import tifffile
            tifffile.imwrite(path, w.image)
            g.status_msg(f'Mask saved to {path}')
            logger.info(f'Mask saved to {path}')
        except Exception as e:
            g.alert(f"Failed to save mask: {e}")
            logger.error(f"Failed to save mask: {e}")


mask_editor = Mask_Editor()
