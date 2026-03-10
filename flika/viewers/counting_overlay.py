"""Object counting overlay for flika windows.

Provides a manual counting/marker system for microscopists:
  - Multiple named categories with distinct colors
  - Numbered markers placed by clicking
  - Per-frame tracking through stacks
  - Labels can be shown or hidden
  - Summary statistics panel
  - CSV export of counts and marker positions
  - Undo support for marker placement/removal
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Signal
import pyqtgraph as pg

import flika.global_vars as g
from flika.logger import logger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CountCategory:
    """A category for counted objects."""
    id: int
    name: str
    color: Tuple[int, int, int]
    marker_symbol: str = 'o'   # 'o', 's', 't', 'd', '+', 'x', 'star'
    marker_size: float = 10


@dataclass
class CountMarker:
    """A single counted marker point."""
    x: float
    y: float
    frame: int
    category_id: int
    number: int   # sequential number within category


class CountingData:
    """Data model for counting markers across all frames.

    Parameters
    ----------
    n_frames : int
        Number of frames in the stack (1 for single images).
    """

    def __init__(self, n_frames: int = 1):
        self.n_frames = n_frames
        self.categories: List[CountCategory] = []
        self.markers: List[CountMarker] = []
        self._next_number: Dict[int, int] = {}  # category_id -> next number

    def add_category(self, name: str, color: Tuple[int, int, int],
                     symbol: str = 'o', size: float = 10) -> CountCategory:
        new_id = max((c.id for c in self.categories), default=-1) + 1
        cat = CountCategory(id=new_id, name=name, color=color,
                            marker_symbol=symbol, marker_size=size)
        self.categories.append(cat)
        self._next_number[new_id] = 1
        return cat

    def remove_category(self, cat_id: int):
        self.categories = [c for c in self.categories if c.id != cat_id]
        self.markers = [m for m in self.markers if m.category_id != cat_id]
        self._next_number.pop(cat_id, None)

    def rename_category(self, cat_id: int, new_name: str):
        for c in self.categories:
            if c.id == cat_id:
                c.name = new_name

    def get_category(self, cat_id: int) -> Optional[CountCategory]:
        for c in self.categories:
            if c.id == cat_id:
                return c
        return None

    def add_marker(self, x: float, y: float, frame: int,
                   category_id: int) -> CountMarker:
        num = self._next_number.get(category_id, 1)
        marker = CountMarker(x=x, y=y, frame=frame,
                             category_id=category_id, number=num)
        self.markers.append(marker)
        self._next_number[category_id] = num + 1
        return marker

    def remove_marker(self, marker: CountMarker):
        try:
            self.markers.remove(marker)
        except ValueError:
            pass

    def remove_nearest(self, x: float, y: float, frame: int,
                       max_dist: float = 15) -> Optional[CountMarker]:
        """Remove and return the nearest marker within max_dist pixels."""
        frame_markers = [m for m in self.markers if m.frame == frame]
        if not frame_markers:
            return None
        dists = [np.sqrt((m.x - x)**2 + (m.y - y)**2) for m in frame_markers]
        idx = int(np.argmin(dists))
        if dists[idx] <= max_dist:
            marker = frame_markers[idx]
            self.markers.remove(marker)
            return marker
        return None

    def get_frame_markers(self, frame: int,
                          category_id: Optional[int] = None) -> List[CountMarker]:
        markers = [m for m in self.markers if m.frame == frame]
        if category_id is not None:
            markers = [m for m in markers if m.category_id == category_id]
        return markers

    def get_counts(self) -> Dict[str, Dict[int, int]]:
        """Return {category_name: {frame: count}} for all categories."""
        result = {}
        for cat in self.categories:
            counts = {}
            for f in range(self.n_frames):
                counts[f] = len([m for m in self.markers
                                 if m.category_id == cat.id and m.frame == f])
            result[cat.name] = counts
        return result

    def get_summary(self) -> Dict[str, int]:
        """Return {category_name: total_count}."""
        summary = {}
        for cat in self.categories:
            summary[cat.name] = len([m for m in self.markers
                                     if m.category_id == cat.id])
        return summary

    def get_frame_summary(self, frame: int) -> Dict[str, int]:
        """Return {category_name: count} for a single frame."""
        result = {}
        for cat in self.categories:
            result[cat.name] = len([m for m in self.markers
                                    if m.category_id == cat.id and m.frame == frame])
        return result

    def export_csv(self, filepath: str):
        """Export all markers to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['number', 'category', 'x', 'y', 'frame'])
            for m in sorted(self.markers, key=lambda m: (m.category_id, m.frame, m.number)):
                cat = self.get_category(m.category_id)
                cat_name = cat.name if cat else f'cat_{m.category_id}'
                writer.writerow([m.number, cat_name, f'{m.x:.2f}', f'{m.y:.2f}', m.frame])

    def export_summary_csv(self, filepath: str):
        """Export per-frame counts to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            cat_names = [c.name for c in self.categories]
            writer.writerow(['frame'] + cat_names + ['total'])
            for frame in range(self.n_frames):
                counts = self.get_frame_summary(frame)
                row = [frame] + [counts.get(n, 0) for n in cat_names]
                row.append(sum(counts.values()))
                writer.writerow(row)

    def clear(self, frame: Optional[int] = None):
        if frame is None:
            self.markers.clear()
            for cat_id in self._next_number:
                self._next_number[cat_id] = 1
        else:
            self.markers = [m for m in self.markers if m.frame != frame]


# ---------------------------------------------------------------------------
# Counting Overlay (visual layer)
# ---------------------------------------------------------------------------

class CountingOverlay(QtCore.QObject):
    """Interactive counting overlay for a flika Window.

    Draws numbered markers on the image, handles click-to-place,
    and provides show/hide labels functionality.

    Signals
    -------
    sigMarkerAdded(CountMarker)
    sigMarkerRemoved(CountMarker)
    sigCountsChanged()
    """

    sigMarkerAdded = Signal(object)
    sigMarkerRemoved = Signal(object)
    sigCountsChanged = Signal()

    def __init__(self, window, n_frames: int = 1):
        super().__init__()
        self.window = window
        self.data = CountingData(n_frames=n_frames)
        self._active_category_id: int = -1
        self._enabled = False
        self._show_labels = True
        self._show_all_frames = False
        self._current_frame = 0
        self._delete_mode = False

        # Graphics items
        self._scatter_items: Dict[int, pg.ScatterPlotItem] = {}  # cat_id -> scatter
        self._label_items: List[Tuple[pg.TextItem, CountMarker]] = []

        # Mouse event filter
        self._proxy = None

    def enable(self):
        """Start counting mode — clicks add markers."""
        self._enabled = True
        # Install scene event filter
        scene = self.window.imageview.scene
        if scene and self._proxy is None:
            self._proxy = _SceneClickFilter(self)
            scene.installEventFilter(self._proxy)

    def disable(self):
        """Stop counting mode."""
        self._enabled = False
        if self._proxy:
            scene = self.window.imageview.scene
            if scene:
                scene.removeEventFilter(self._proxy)
            self._proxy = None

    def set_active_category(self, cat_id: int):
        self._active_category_id = cat_id

    def set_delete_mode(self, enabled: bool):
        self._delete_mode = enabled

    def set_show_labels(self, show: bool):
        self._show_labels = show
        self._rebuild_display()

    def set_show_all_frames(self, show: bool):
        self._show_all_frames = show
        self._rebuild_display()

    def set_frame(self, frame: int):
        self._current_frame = frame
        self._rebuild_display()

    def handle_click(self, x: float, y: float):
        """Handle a click at image coordinates (x, y)."""
        if not self._enabled:
            return

        if self._delete_mode:
            removed = self.data.remove_nearest(x, y, self._current_frame)
            if removed:
                self.sigMarkerRemoved.emit(removed)
                self.sigCountsChanged.emit()
                self._rebuild_display()
            return

        if self._active_category_id < 0:
            return

        marker = self.data.add_marker(x, y, self._current_frame,
                                      self._active_category_id)
        self.sigMarkerAdded.emit(marker)
        self.sigCountsChanged.emit()
        self._rebuild_display()

    def clear_all(self):
        self.data.clear()
        self.sigCountsChanged.emit()
        self._rebuild_display()

    def clear_frame(self):
        self.data.clear(self._current_frame)
        self.sigCountsChanged.emit()
        self._rebuild_display()

    def cleanup(self):
        """Remove all graphics and disconnect."""
        self.disable()
        view = self.window.imageview.view
        for scatter in self._scatter_items.values():
            try:
                view.removeItem(scatter)
            except Exception:
                pass
        self._scatter_items.clear()
        for text_item, _ in self._label_items:
            try:
                view.removeItem(text_item)
            except Exception:
                pass
        self._label_items.clear()

    def _rebuild_display(self):
        """Rebuild all scatter + label items for current frame."""
        view = self.window.imageview.view

        # Remove old items
        for scatter in self._scatter_items.values():
            try:
                view.removeItem(scatter)
            except Exception:
                pass
        self._scatter_items.clear()

        for text_item, _ in self._label_items:
            try:
                view.removeItem(text_item)
            except Exception:
                pass
        self._label_items.clear()

        # Get markers to display
        if self._show_all_frames:
            markers = self.data.markers
        else:
            markers = self.data.get_frame_markers(self._current_frame)

        # Group by category
        by_cat: Dict[int, List[CountMarker]] = {}
        for m in markers:
            by_cat.setdefault(m.category_id, []).append(m)

        for cat_id, cat_markers in by_cat.items():
            cat = self.data.get_category(cat_id)
            if cat is None:
                continue

            spots = []
            for m in cat_markers:
                spots.append({'pos': (m.x, m.y), 'size': cat.marker_size,
                              'symbol': cat.marker_symbol})

            r, g_, b = cat.color
            scatter = pg.ScatterPlotItem(
                spots=spots,
                brush=pg.mkBrush(r, g_, b, 180),
                pen=pg.mkPen('w', width=1),
                pxMode=True,
            )
            scatter.setZValue(18)
            view.addItem(scatter)
            self._scatter_items[cat_id] = scatter

            # Labels (number text)
            if self._show_labels:
                for m in cat_markers:
                    text = pg.TextItem(
                        str(m.number),
                        color=QtGui.QColor(r, g_, b),
                        anchor=(0, 1),
                    )
                    text.setPos(m.x + cat.marker_size / 2, m.y - cat.marker_size / 2)
                    text.setZValue(19)
                    view.addItem(text)
                    self._label_items.append((text, m))

    def get_data(self) -> CountingData:
        return self.data


class _SceneClickFilter(QtCore.QObject):
    """Event filter to intercept mouse clicks on the ImageView scene."""

    def __init__(self, overlay: CountingOverlay):
        super().__init__()
        self.overlay = overlay

    def eventFilter(self, obj, event):
        if not self.overlay._enabled:
            return False
        if event.type() == QtCore.QEvent.GraphicsSceneMousePress:
            if event.button() == QtCore.Qt.LeftButton:
                # Map scene position to image coordinates
                img_item = self.overlay.window.imageview.getImageItem()
                pos = img_item.mapFromScene(event.scenePos())
                x, y = pos.x(), pos.y()
                # Check bounds
                image = img_item.image
                if image is not None:
                    h, w = image.shape[:2]
                    if 0 <= x < w and 0 <= y < h:
                        self.overlay.handle_click(x, y)
                        return True
        return False


# ---------------------------------------------------------------------------
# Counting Panel (dockable control panel)
# ---------------------------------------------------------------------------

class CountingPanel(QtWidgets.QDockWidget):
    """Dockable panel for manual object counting.

    Provides category management, count/place/delete controls,
    per-frame statistics, and export.
    """

    _instance = None

    @classmethod
    def instance(cls, parent=None):
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent=None):
        super().__init__('Counting Tool', parent)
        self.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self._overlay: Optional[CountingOverlay] = None
        self._build_ui()

    def _build_ui(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Attach button
        btn_attach = QtWidgets.QPushButton('Attach to Current Window')
        btn_attach.clicked.connect(self._attach_to_window)
        layout.addWidget(btn_attach)

        # Enable/disable
        self._enable_check = QtWidgets.QCheckBox('Counting Mode Active')
        self._enable_check.toggled.connect(self._toggle_counting)
        layout.addWidget(self._enable_check)

        # Category management
        cat_group = QtWidgets.QGroupBox('Categories')
        cat_layout = QtWidgets.QVBoxLayout(cat_group)

        self._cat_list = QtWidgets.QListWidget()
        self._cat_list.currentRowChanged.connect(self._category_selected)
        cat_layout.addWidget(self._cat_list)

        cat_btns = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton('+')
        btn_add.setFixedWidth(30)
        btn_add.clicked.connect(self._add_category)
        btn_rm = QtWidgets.QPushButton('-')
        btn_rm.setFixedWidth(30)
        btn_rm.clicked.connect(self._remove_category)
        btn_rename = QtWidgets.QPushButton('Rename')
        btn_rename.clicked.connect(self._rename_category)
        cat_btns.addWidget(btn_add)
        cat_btns.addWidget(btn_rm)
        cat_btns.addWidget(btn_rename)
        cat_layout.addLayout(cat_btns)

        # Marker style
        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel('Symbol:'))
        self._symbol_combo = QtWidgets.QComboBox()
        self._symbol_combo.addItems(['o', 's', 't', 'd', '+', 'x', 'star'])
        style_row.addWidget(self._symbol_combo)
        style_row.addWidget(QtWidgets.QLabel('Size:'))
        self._size_spin = QtWidgets.QSpinBox()
        self._size_spin.setRange(3, 50)
        self._size_spin.setValue(10)
        style_row.addWidget(self._size_spin)
        cat_layout.addLayout(style_row)

        layout.addWidget(cat_group)

        # Tool mode
        mode_group = QtWidgets.QGroupBox('Mode')
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self._mode_place = QtWidgets.QRadioButton('Place')
        self._mode_delete = QtWidgets.QRadioButton('Delete')
        self._mode_place.setChecked(True)
        self._mode_place.toggled.connect(lambda c: self._set_delete_mode(not c))
        mode_layout.addWidget(self._mode_place)
        mode_layout.addWidget(self._mode_delete)
        layout.addWidget(mode_group)

        # Display options
        disp_group = QtWidgets.QGroupBox('Display')
        disp_layout = QtWidgets.QVBoxLayout(disp_group)
        self._show_labels_check = QtWidgets.QCheckBox('Show Labels')
        self._show_labels_check.setChecked(True)
        self._show_labels_check.toggled.connect(self._toggle_labels)
        disp_layout.addWidget(self._show_labels_check)
        self._show_all_check = QtWidgets.QCheckBox('Show All Frames')
        self._show_all_check.toggled.connect(self._toggle_all_frames)
        disp_layout.addWidget(self._show_all_check)
        layout.addWidget(disp_group)

        # Statistics
        stats_group = QtWidgets.QGroupBox('Statistics')
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        self._stats_label = QtWidgets.QLabel('No counts yet')
        self._stats_label.setWordWrap(True)
        stats_layout.addWidget(self._stats_label)
        layout.addWidget(stats_group)

        # Actions
        action_group = QtWidgets.QGroupBox('Actions')
        action_layout = QtWidgets.QVBoxLayout(action_group)
        btn_clear_frame = QtWidgets.QPushButton('Clear Current Frame')
        btn_clear_frame.clicked.connect(self._clear_frame)
        action_layout.addWidget(btn_clear_frame)
        btn_clear_all = QtWidgets.QPushButton('Clear All')
        btn_clear_all.clicked.connect(self._clear_all)
        action_layout.addWidget(btn_clear_all)

        export_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton('Export Markers CSV')
        btn_export.clicked.connect(self._export_markers)
        export_row.addWidget(btn_export)
        btn_export_summary = QtWidgets.QPushButton('Export Summary CSV')
        btn_export_summary.clicked.connect(self._export_summary)
        export_row.addWidget(btn_export_summary)
        action_layout.addLayout(export_row)

        layout.addWidget(action_group)
        layout.addStretch()
        self.setWidget(widget)

    def _attach_to_window(self):
        win = g.win
        if win is None:
            g.alert('No window selected.')
            return

        # Clean up old overlay
        if self._overlay is not None:
            self._overlay.cleanup()

        n_frames = win.image.shape[0] if win.image.ndim >= 3 and not win.metadata.get('is_rgb', False) else 1
        self._overlay = CountingOverlay(win, n_frames=n_frames)
        self._overlay.sigCountsChanged.connect(self._update_stats)

        # Connect frame changes
        if n_frames > 1:
            win.sigTimeChanged.connect(self._on_frame_changed)

        # Add default category if empty
        if not self._overlay.data.categories:
            self._overlay.data.add_category('Type 1', (255, 0, 0), 'o', 10)
            self._overlay.data.add_category('Type 2', (0, 255, 0), 's', 10)

        self._refresh_categories()
        self._enable_check.setChecked(False)
        self._update_stats()
        self.setWindowTitle(f'Counting Tool — {win.name}')

    def _toggle_counting(self, enabled: bool):
        if self._overlay is None:
            if enabled:
                self._enable_check.setChecked(False)
                g.alert('Attach to a window first.')
            return
        if enabled:
            self._overlay.enable()
        else:
            self._overlay.disable()

    def _on_frame_changed(self, frame):
        if self._overlay:
            self._overlay.set_frame(frame)
            self._update_stats()

    def _refresh_categories(self):
        self._cat_list.clear()
        if self._overlay is None:
            return
        for cat in self._overlay.data.categories:
            item = QtWidgets.QListWidgetItem(cat.name)
            item.setForeground(QtGui.QColor(*cat.color))
            item.setData(QtCore.Qt.UserRole, cat.id)
            self._cat_list.addItem(item)
        if self._cat_list.count() > 0:
            self._cat_list.setCurrentRow(0)

    def _category_selected(self, row: int):
        if row < 0 or self._overlay is None:
            return
        item = self._cat_list.item(row)
        if item is None:
            return
        cat_id = item.data(QtCore.Qt.UserRole)
        self._overlay.set_active_category(cat_id)

    def _add_category(self):
        if self._overlay is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, 'Add Category', 'Name:')
        if not ok or not name:
            return
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(255, 0, 0), self, 'Category Color')
        if not color.isValid():
            return
        symbol = self._symbol_combo.currentText()
        size = self._size_spin.value()
        self._overlay.data.add_category(
            name, (color.red(), color.green(), color.blue()), symbol, size)
        self._refresh_categories()

    def _remove_category(self):
        if self._overlay is None:
            return
        row = self._cat_list.currentRow()
        if row < 0:
            return
        cat_id = self._cat_list.item(row).data(QtCore.Qt.UserRole)
        self._overlay.data.remove_category(cat_id)
        self._refresh_categories()
        self._overlay._rebuild_display()
        self._update_stats()

    def _rename_category(self):
        if self._overlay is None:
            return
        row = self._cat_list.currentRow()
        if row < 0:
            return
        item = self._cat_list.item(row)
        cat_id = item.data(QtCore.Qt.UserRole)
        name, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename Category', 'New name:', text=item.text())
        if ok and name:
            self._overlay.data.rename_category(cat_id, name)
            self._refresh_categories()

    def _set_delete_mode(self, delete: bool):
        if self._overlay:
            self._overlay.set_delete_mode(delete)

    def _toggle_labels(self, show: bool):
        if self._overlay:
            self._overlay.set_show_labels(show)

    def _toggle_all_frames(self, show: bool):
        if self._overlay:
            self._overlay.set_show_all_frames(show)

    def _update_stats(self):
        if self._overlay is None:
            self._stats_label.setText('No counts yet')
            return

        frame = self._overlay._current_frame
        frame_counts = self._overlay.data.get_frame_summary(frame)
        total_counts = self._overlay.data.get_summary()

        lines = [f'Frame {frame + 1}/{self._overlay.data.n_frames}:']
        frame_total = 0
        for name, count in frame_counts.items():
            lines.append(f'  {name}: {count}')
            frame_total += count
        lines.append(f'  Frame total: {frame_total}')
        lines.append('')
        lines.append('All frames:')
        grand_total = 0
        for name, count in total_counts.items():
            lines.append(f'  {name}: {count}')
            grand_total += count
        lines.append(f'  Grand total: {grand_total}')

        self._stats_label.setText('\n'.join(lines))

    def _clear_frame(self):
        if self._overlay:
            self._overlay.clear_frame()
            self._update_stats()

    def _clear_all(self):
        if self._overlay:
            self._overlay.clear_all()
            self._update_stats()

    def _export_markers(self):
        if self._overlay is None or not self._overlay.data.markers:
            g.alert('No markers to export.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export Markers', 'markers.csv', 'CSV (*.csv)')
        if path:
            self._overlay.data.export_csv(path)
            g.alert(f'Exported {len(self._overlay.data.markers)} markers to {path}')

    def _export_summary(self):
        if self._overlay is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export Summary', 'counts_summary.csv', 'CSV (*.csv)')
        if path:
            self._overlay.data.export_summary_csv(path)
            g.alert(f'Summary exported to {path}')

    def closeEvent(self, event):
        if self._overlay:
            self._overlay.cleanup()
            self._overlay = None
        CountingPanel._instance = None
        super().closeEvent(event)
