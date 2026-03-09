"""Overlay Manager panel and baking system for flika.

Provides a dockable panel to manage all overlays on a window (grids,
counting markers, text annotations, scale bars, timestamps, ROIs)
and a system to 'bake' (burn) overlays into the image data so they
persist when copied to other windows or exported.
"""
from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .. import global_vars as g
from ..logger import logger
from .grid_overlay import GridOverlay, TextAnnotation, _make_pen, _GRID_COLORS
from ..utils.singleton import DockSingleton


# ---------------------------------------------------------------------------
# Baking Engine
# ---------------------------------------------------------------------------

def bake_overlays(window, include_rois: bool = True,
                  include_grid: bool = True,
                  include_scale_bar: bool = True,
                  include_timestamp: bool = True,
                  include_annotations: bool = True,
                  include_counting: bool = True,
                  create_new_window: bool = True):
    """Burn visible overlays into the image pixel data.

    Renders the current view (with all visible graphics items) to an
    RGB(A) array and either replaces the window's image or creates a
    new window.

    Parameters
    ----------
    window : flika.window.Window
        Source window.
    include_rois : bool
        Include ROIs.
    include_grid : bool
        Include grid overlay.
    include_scale_bar : bool
        Include scale bar.
    include_timestamp : bool
        Include timestamp label.
    include_annotations : bool
        Include text annotations.
    include_counting : bool
        Include counting markers.
    create_new_window : bool
        If True, create a new window; otherwise overwrite the current image.

    Returns
    -------
    numpy.ndarray or Window
        The baked image array, or the new Window if create_new_window.
    """
    from ..window import Window

    image = window.image
    is_stack = image.ndim >= 3 and not window.metadata.get('is_rgb', False)

    if is_stack:
        n_frames = image.shape[0]
        result_frames = []
        current_idx = window.currentIndex

        for i in range(n_frames):
            window.setIndex(i)
            # Update timestamp if present
            if hasattr(window, 'updateTimeStampLabel') and window._ts_props:
                window.updateTimeStampLabel(i)
            frame_rgb = _render_frame(window, include_rois, include_grid,
                                      include_scale_bar, include_timestamp,
                                      include_annotations, include_counting)
            result_frames.append(frame_rgb)

        window.setIndex(current_idx)
        result = np.stack(result_frames, axis=0)
    else:
        result = _render_frame(window, include_rois, include_grid,
                               include_scale_bar, include_timestamp,
                               include_annotations, include_counting)

    if create_new_window:
        new_win = Window(result, name=window.name + ' - Baked')
        new_win.metadata['is_rgb'] = True
        return new_win
    else:
        window.imageview.setImage(result)
        window.image = result
        window.metadata['is_rgb'] = True
        return result


def _render_frame(window, include_rois, include_grid, include_scale_bar,
                  include_timestamp, include_annotations, include_counting):
    """Render the current frame with overlays to an RGB array."""
    view = window.imageview.view
    scene = view.scene()

    # Temporarily hide items we don't want
    hidden_items = []

    def _set_visible(attr_name, include):
        """Hide an item if include is False, track for restoration."""
        items = []
        if hasattr(window, attr_name):
            item = getattr(window, attr_name)
            if item is not None:
                if isinstance(item, list):
                    items = item
                else:
                    items = [item]
        for item in items:
            if not include and item.isVisible():
                item.setVisible(False)
                hidden_items.append(item)

    if not include_scale_bar:
        _set_visible('_sb_bar', False)
        _set_visible('_sb_text', False)
    if not include_timestamp:
        _set_visible('timeStampLabel', False)

    # Hide ROIs if not wanted
    if not include_rois:
        for roi in window.rois:
            if roi.isVisible():
                roi.setVisible(False)
                hidden_items.append(roi)

    # Grid overlay items
    if not include_grid and hasattr(window, '_grid_overlay'):
        grid = window._grid_overlay
        for item in grid._items:
            if item.isVisible():
                item.setVisible(False)
                hidden_items.append(item)

    # Text annotations
    if not include_annotations and hasattr(window, '_text_annotations'):
        for ann in window._text_annotations:
            if ann._item and ann._item.isVisible():
                ann._item.setVisible(False)
                hidden_items.append(ann._item)

    # Render to QImage
    # Get the view rect in scene coordinates
    view_rect = view.viewRect()
    h = int(view_rect.height())
    w = int(view_rect.width())

    # Use a higher resolution render
    target_size = QtCore.QSize(w, h)
    qimage = QtGui.QImage(target_size, QtGui.QImage.Format_ARGB32)
    qimage.fill(QtCore.Qt.black)

    painter = QtGui.QPainter(qimage)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    scene.render(painter, QtCore.QRectF(qimage.rect()),
                 QtCore.QRectF(view_rect))
    painter.end()

    # Restore hidden items
    for item in hidden_items:
        item.setVisible(True)

    # Convert QImage to numpy
    ptr = qimage.bits()
    if hasattr(ptr, 'setsize'):
        ptr.setsize(qimage.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()
    # ARGB32 -> RGB
    rgb = arr[:, :, [2, 1, 0]]  # BGRA -> RGB
    return rgb


def bake_rois_only(window, color=None, line_width: int = 2,
                   fill_opacity: float = 0.0, create_new_window: bool = True):
    """Burn only ROI outlines into the image data.

    This is a pixel-level operation that draws ROI boundaries directly
    onto the numpy array without going through Qt rendering, giving
    cleaner results for individual ROI export.
    """
    from ..window import Window

    image = window.image
    is_rgb = window.metadata.get('is_rgb', False)
    is_stack = image.ndim >= 3 and not is_rgb

    # Work with RGB
    if is_rgb:
        result = image.copy()
    elif image.ndim == 2:
        # Normalize to uint8 and convert to RGB
        result = _to_rgb(image)
    elif is_stack:
        frames = []
        for i in range(image.shape[0]):
            frames.append(_to_rgb(image[i]))
        result = np.stack(frames, axis=0)
    else:
        result = _to_rgb(image)

    # Default color
    if color is None:
        color = (255, 255, 0)  # yellow

    # Draw ROI boundaries
    for roi in window.rois:
        try:
            mask_rows, mask_cols = roi.getMask()
        except Exception:
            continue

        if len(mask_rows) == 0:
            continue

        # Find boundary pixels (pixels in mask adjacent to non-mask pixels)
        from scipy.ndimage import binary_dilation
        h = result.shape[-3] if result.ndim > 3 else (result.shape[-3] if result.ndim == 3 and not is_rgb else result.shape[0])

        if is_stack:
            # Apply to all frames
            for f_idx in range(result.shape[0]):
                _draw_roi_boundary(result[f_idx], mask_rows, mask_cols,
                                   color, line_width)
        else:
            _draw_roi_boundary(result, mask_rows, mask_cols, color, line_width)

    if create_new_window:
        new_win = Window(result, name=window.name + ' - ROIs Baked')
        new_win.metadata['is_rgb'] = True
        return new_win
    else:
        window.imageview.setImage(result)
        window.image = result
        window.metadata['is_rgb'] = True
        return result


def _to_rgb(frame_2d: np.ndarray) -> np.ndarray:
    """Convert a single-channel frame to uint8 RGB."""
    f = frame_2d.astype(np.float64)
    mn, mx = f.min(), f.max()
    if mx > mn:
        f = (f - mn) / (mx - mn) * 255
    else:
        f = np.zeros_like(f)
    gray = f.astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _draw_roi_boundary(rgb_frame, mask_rows, mask_cols, color, line_width):
    """Draw ROI boundary pixels onto an RGB frame."""
    h, w = rgb_frame.shape[:2]
    # Create binary mask
    mask = np.zeros((h, w), dtype=bool)
    valid = (mask_rows >= 0) & (mask_rows < h) & (mask_cols >= 0) & (mask_cols < w)
    mask[mask_rows[valid], mask_cols[valid]] = True

    # Dilate and find boundary
    from scipy.ndimage import binary_dilation
    struct = np.ones((3, 3), dtype=bool)
    for _ in range(max(1, line_width // 2)):
        dilated = binary_dilation(mask, structure=struct)
        boundary = dilated & ~mask
        mask = dilated

    boundary_rows, boundary_cols = np.where(boundary)
    if len(boundary_rows) > 0:
        rgb_frame[boundary_rows, boundary_cols] = color


# ---------------------------------------------------------------------------
# Overlay Manager Panel
# ---------------------------------------------------------------------------

class OverlayManagerPanel(DockSingleton):
    """Dockable panel for managing all overlays on the current window."""

    def __init__(self, parent=None):
        super().__init__('Overlay Manager', parent)
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self._grid: GridOverlay | None = None
        self._annotations: list[TextAnnotation] = []
        self._build_ui()

    def _build_ui(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # -- Grid section --
        grid_group = QtWidgets.QGroupBox('Grid Overlay')
        grid_layout = QtWidgets.QVBoxLayout(grid_group)

        # Grid type
        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel('Type:'))
        self._grid_type = QtWidgets.QComboBox()
        self._grid_type.addItems([
            'None', 'Rectangular', 'Crosshair', 'Rule of Thirds',
            'Dot Grid', 'Polar',
        ])
        type_row.addWidget(self._grid_type)
        grid_layout.addLayout(type_row)

        # Grid parameters
        param_form = QtWidgets.QFormLayout()

        self._grid_spacing_x = QtWidgets.QSpinBox()
        self._grid_spacing_x.setRange(5, 5000)
        self._grid_spacing_x.setValue(50)
        param_form.addRow('Spacing X:', self._grid_spacing_x)

        self._grid_spacing_y = QtWidgets.QSpinBox()
        self._grid_spacing_y.setRange(5, 5000)
        self._grid_spacing_y.setValue(50)
        param_form.addRow('Spacing Y:', self._grid_spacing_y)

        self._grid_divisions_x = QtWidgets.QSpinBox()
        self._grid_divisions_x.setRange(1, 100)
        self._grid_divisions_x.setValue(4)
        param_form.addRow('Divisions X:', self._grid_divisions_x)

        self._grid_divisions_y = QtWidgets.QSpinBox()
        self._grid_divisions_y.setRange(1, 100)
        self._grid_divisions_y.setValue(4)
        param_form.addRow('Divisions Y:', self._grid_divisions_y)

        self._grid_color = QtWidgets.QComboBox()
        self._grid_color.addItems(list(_GRID_COLORS.keys()))
        self._grid_color.setCurrentText('Yellow')
        param_form.addRow('Color:', self._grid_color)

        self._grid_width = QtWidgets.QDoubleSpinBox()
        self._grid_width.setRange(0.5, 10)
        self._grid_width.setValue(1.0)
        self._grid_width.setSingleStep(0.5)
        param_form.addRow('Line Width:', self._grid_width)

        self._grid_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._grid_opacity.setRange(5, 100)
        self._grid_opacity.setValue(40)
        param_form.addRow('Opacity:', self._grid_opacity)

        self._grid_style = QtWidgets.QComboBox()
        self._grid_style.addItems(['solid', 'dash', 'dot', 'dash-dot'])
        param_form.addRow('Line Style:', self._grid_style)

        self._grid_use_divisions = QtWidgets.QCheckBox('Use divisions instead of spacing')
        param_form.addRow('', self._grid_use_divisions)

        # Crosshair extras
        self._grid_circle = QtWidgets.QCheckBox('Show center circle')
        param_form.addRow('', self._grid_circle)

        self._grid_circle_radius = QtWidgets.QSpinBox()
        self._grid_circle_radius.setRange(5, 500)
        self._grid_circle_radius.setValue(20)
        param_form.addRow('Circle Radius:', self._grid_circle_radius)

        # Polar extras
        self._grid_rings = QtWidgets.QSpinBox()
        self._grid_rings.setRange(1, 50)
        self._grid_rings.setValue(5)
        param_form.addRow('Rings:', self._grid_rings)

        self._grid_spokes = QtWidgets.QSpinBox()
        self._grid_spokes.setRange(2, 72)
        self._grid_spokes.setValue(8)
        param_form.addRow('Spokes:', self._grid_spokes)

        # Dot size
        self._grid_dot_size = QtWidgets.QDoubleSpinBox()
        self._grid_dot_size.setRange(1, 20)
        self._grid_dot_size.setValue(3)
        param_form.addRow('Dot Size:', self._grid_dot_size)

        grid_layout.addLayout(param_form)

        btn_apply_grid = QtWidgets.QPushButton('Apply Grid')
        btn_apply_grid.clicked.connect(self._apply_grid)
        grid_layout.addWidget(btn_apply_grid)

        btn_clear_grid = QtWidgets.QPushButton('Clear Grid')
        btn_clear_grid.clicked.connect(self._clear_grid)
        grid_layout.addWidget(btn_clear_grid)

        layout.addWidget(grid_group)

        # -- Text Annotations section --
        ann_group = QtWidgets.QGroupBox('Text Annotations')
        ann_layout = QtWidgets.QVBoxLayout(ann_group)

        ann_row1 = QtWidgets.QHBoxLayout()
        ann_row1.addWidget(QtWidgets.QLabel('Text:'))
        self._ann_text = QtWidgets.QLineEdit()
        self._ann_text.setPlaceholderText('Enter annotation text')
        ann_row1.addWidget(self._ann_text)
        ann_layout.addLayout(ann_row1)

        ann_row2 = QtWidgets.QHBoxLayout()
        ann_row2.addWidget(QtWidgets.QLabel('X:'))
        self._ann_x = QtWidgets.QSpinBox()
        self._ann_x.setRange(0, 99999)
        ann_row2.addWidget(self._ann_x)
        ann_row2.addWidget(QtWidgets.QLabel('Y:'))
        self._ann_y = QtWidgets.QSpinBox()
        self._ann_y.setRange(0, 99999)
        ann_row2.addWidget(self._ann_y)
        ann_layout.addLayout(ann_row2)

        ann_row3 = QtWidgets.QHBoxLayout()
        ann_row3.addWidget(QtWidgets.QLabel('Size:'))
        self._ann_size = QtWidgets.QSpinBox()
        self._ann_size.setRange(4, 120)
        self._ann_size.setValue(14)
        ann_row3.addWidget(self._ann_size)
        ann_row3.addWidget(QtWidgets.QLabel('Color:'))
        self._ann_color = QtWidgets.QComboBox()
        self._ann_color.addItems(list(_GRID_COLORS.keys()))
        self._ann_color.setCurrentText('White')
        ann_row3.addWidget(self._ann_color)
        self._ann_bold = QtWidgets.QCheckBox('Bold')
        ann_row3.addWidget(self._ann_bold)
        ann_layout.addLayout(ann_row3)

        self._ann_frame_only = QtWidgets.QCheckBox('Current frame only')
        ann_layout.addWidget(self._ann_frame_only)

        btn_add_ann = QtWidgets.QPushButton('Add Annotation')
        btn_add_ann.clicked.connect(self._add_annotation)
        ann_layout.addWidget(btn_add_ann)

        btn_clear_ann = QtWidgets.QPushButton('Clear All Annotations')
        btn_clear_ann.clicked.connect(self._clear_annotations)
        ann_layout.addWidget(btn_clear_ann)

        layout.addWidget(ann_group)

        # -- Ruler section --
        ruler_group = QtWidgets.QGroupBox('Measurement Ruler')
        ruler_layout = QtWidgets.QVBoxLayout(ruler_group)

        ruler_info = QtWidgets.QLabel(
            'Draw a line ROI on the image, then click\n'
            '"Add Ruler from ROI" to create a measurement.')
        ruler_info.setWordWrap(True)
        ruler_layout.addWidget(ruler_info)

        btn_ruler = QtWidgets.QPushButton('Add Ruler from Line ROI')
        btn_ruler.clicked.connect(self._add_ruler_from_roi)
        ruler_layout.addWidget(btn_ruler)

        layout.addWidget(ruler_group)

        # -- Baking section --
        bake_group = QtWidgets.QGroupBox('Bake Overlays')
        bake_layout = QtWidgets.QVBoxLayout(bake_group)

        bake_info = QtWidgets.QLabel(
            'Burns selected overlays into the image pixels\n'
            'so they persist in copies and exports.')
        bake_info.setWordWrap(True)
        bake_layout.addWidget(bake_info)

        self._bake_rois = QtWidgets.QCheckBox('ROIs')
        self._bake_rois.setChecked(True)
        bake_layout.addWidget(self._bake_rois)

        self._bake_grid = QtWidgets.QCheckBox('Grid')
        self._bake_grid.setChecked(True)
        bake_layout.addWidget(self._bake_grid)

        self._bake_scalebar = QtWidgets.QCheckBox('Scale Bar')
        self._bake_scalebar.setChecked(True)
        bake_layout.addWidget(self._bake_scalebar)

        self._bake_timestamp = QtWidgets.QCheckBox('Timestamp')
        self._bake_timestamp.setChecked(True)
        bake_layout.addWidget(self._bake_timestamp)

        self._bake_annotations = QtWidgets.QCheckBox('Text Annotations')
        self._bake_annotations.setChecked(True)
        bake_layout.addWidget(self._bake_annotations)

        self._bake_counting = QtWidgets.QCheckBox('Counting Markers')
        self._bake_counting.setChecked(True)
        bake_layout.addWidget(self._bake_counting)

        self._bake_new_window = QtWidgets.QCheckBox('Create new window')
        self._bake_new_window.setChecked(True)
        bake_layout.addWidget(self._bake_new_window)

        btn_bake_all = QtWidgets.QPushButton('Bake All Selected')
        btn_bake_all.clicked.connect(self._bake_all)
        bake_layout.addWidget(btn_bake_all)

        btn_bake_rois = QtWidgets.QPushButton('Bake ROIs Only (pixel-level)')
        btn_bake_rois.clicked.connect(self._bake_rois_only)
        bake_layout.addWidget(btn_bake_rois)

        layout.addWidget(bake_group)

        layout.addStretch()
        self.setWidget(widget)

    # -- Grid actions --

    def _apply_grid(self):
        win = g.win
        if win is None:
            g.alert('No window selected.')
            return

        grid_type = self._grid_type.currentText()
        if grid_type == 'None':
            self._clear_grid()
            return

        if not hasattr(win, '_grid_overlay') or win._grid_overlay is None:
            win._grid_overlay = GridOverlay(win)
        self._grid = win._grid_overlay

        color = self._grid_color.currentText()
        width = self._grid_width.value()
        opacity = self._grid_opacity.value() / 100.0
        style = self._grid_style.currentText()

        if grid_type == 'Rectangular':
            self._grid.draw_rectangular(
                spacing_x=self._grid_spacing_x.value(),
                spacing_y=self._grid_spacing_y.value(),
                color=color, width=width, opacity=opacity, style=style,
                divisions_mode=self._grid_use_divisions.isChecked(),
                divisions_x=self._grid_divisions_x.value(),
                divisions_y=self._grid_divisions_y.value(),
            )
        elif grid_type == 'Crosshair':
            self._grid.draw_crosshair(
                color=color, width=width, opacity=opacity, style=style,
                show_circle=self._grid_circle.isChecked(),
                circle_radius=self._grid_circle_radius.value(),
            )
        elif grid_type == 'Rule of Thirds':
            self._grid.draw_thirds(color=color, width=width,
                                   opacity=opacity, style=style)
        elif grid_type == 'Dot Grid':
            self._grid.draw_dot_grid(
                spacing_x=self._grid_spacing_x.value(),
                spacing_y=self._grid_spacing_y.value(),
                color=color, dot_size=self._grid_dot_size.value(),
                opacity=opacity,
            )
        elif grid_type == 'Polar':
            self._grid.draw_polar(
                n_rings=self._grid_rings.value(),
                n_spokes=self._grid_spokes.value(),
                color=color, width=width, opacity=opacity, style=style,
            )

    def _clear_grid(self):
        win = g.win
        if win and hasattr(win, '_grid_overlay') and win._grid_overlay:
            win._grid_overlay.clear()

    # -- Annotation actions --

    def _add_annotation(self):
        win = g.win
        if win is None:
            g.alert('No window selected.')
            return

        text = self._ann_text.text().strip()
        if not text:
            return

        frame = None
        if self._ann_frame_only.isChecked():
            frame = win.currentIndex if hasattr(win, 'currentIndex') else 0

        ann = TextAnnotation(
            win, text,
            x=self._ann_x.value(), y=self._ann_y.value(),
            font_size=self._ann_size.value(),
            color=self._ann_color.currentText(),
            bold=self._ann_bold.isChecked(),
            frame=frame,
        )

        if not hasattr(win, '_text_annotations'):
            win._text_annotations = []
        win._text_annotations.append(ann)
        self._annotations.append(ann)

    def _clear_annotations(self):
        win = g.win
        if win and hasattr(win, '_text_annotations'):
            for ann in win._text_annotations:
                ann.remove()
            win._text_annotations.clear()
        for ann in self._annotations:
            ann.remove()
        self._annotations.clear()

    # -- Ruler --

    def _add_ruler_from_roi(self):
        win = g.win
        if win is None or win.currentROI is None:
            g.alert('Select a line ROI first.')
            return

        roi = win.currentROI
        roi_type = type(roi).__name__
        if 'line' not in roi_type.lower():
            g.alert('Please select a line ROI.')
            return

        try:
            handles = roi.getHandles()
            if len(handles) >= 2:
                p1 = roi.mapToParent(handles[0].pos())
                p2 = roi.mapToParent(handles[1].pos())
                if not hasattr(win, '_grid_overlay') or win._grid_overlay is None:
                    win._grid_overlay = GridOverlay(win)
                win._grid_overlay.draw_ruler(
                    p1.x(), p1.y(), p2.x(), p2.y(),
                    color=self._grid_color.currentText(),
                    width=self._grid_width.value(),
                    opacity=self._grid_opacity.value() / 100.0,
                )
        except Exception as e:
            g.alert(f'Could not create ruler: {e}')

    # -- Baking --

    def _bake_all(self):
        win = g.win
        if win is None:
            g.alert('No window selected.')
            return

        bake_overlays(
            win,
            include_rois=self._bake_rois.isChecked(),
            include_grid=self._bake_grid.isChecked(),
            include_scale_bar=self._bake_scalebar.isChecked(),
            include_timestamp=self._bake_timestamp.isChecked(),
            include_annotations=self._bake_annotations.isChecked(),
            include_counting=self._bake_counting.isChecked(),
            create_new_window=self._bake_new_window.isChecked(),
        )

    def _bake_rois_only(self):
        win = g.win
        if win is None:
            g.alert('No window selected.')
            return
        if not win.rois:
            g.alert('No ROIs to bake.')
            return

        bake_rois_only(
            win,
            create_new_window=self._bake_new_window.isChecked(),
        )

    # closeEvent is handled by DockSingleton (calls cleanup + clears _instance)
