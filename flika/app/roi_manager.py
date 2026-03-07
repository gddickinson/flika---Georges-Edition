# -*- coding: utf-8 -*-
"""ROI Manager — dockable panel listing all ROIs in the current window."""
from qtpy import QtCore, QtGui, QtWidgets
import numpy as np
from .. import global_vars as g


class ROIManager(QtWidgets.QDockWidget):
    """Dockable panel that lists all ROIs in the active window.

    Provides bulk operations (plot all, unplot all, delete selected, export CSV)
    and per-ROI editing (rename, change color).
    """

    _instance = None  # singleton

    @classmethod
    def instance(cls, parent=None):
        if cls._instance is None or cls._instance._destroyed:
            cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent=None):
        super().__init__("ROI Manager", parent)
        self._destroyed = False
        self.setObjectName("ROIManagerDock")
        self.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # Tree widget
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Plotted"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setRootIsDecorated(False)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setColumnWidth(0, 140)
        self.tree.setColumnWidth(1, 80)
        layout.addWidget(self.tree)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.plotAllBtn = QtWidgets.QPushButton("Plot All")
        self.unplotAllBtn = QtWidgets.QPushButton("Unplot All")
        self.deleteSelBtn = QtWidgets.QPushButton("Delete Sel")
        self.exportBtn = QtWidgets.QPushButton("Export Stats")
        for btn in (self.plotAllBtn, self.unplotAllBtn, self.deleteSelBtn, self.exportBtn):
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self.plotAllBtn.clicked.connect(self._plot_all)
        self.unplotAllBtn.clicked.connect(self._unplot_all)
        self.deleteSelBtn.clicked.connect(self._delete_selected)
        self.exportBtn.clicked.connect(self._export_stats)

        # Boolean operation buttons
        bool_layout = QtWidgets.QHBoxLayout()
        bool_label = QtWidgets.QLabel("Bool:")
        self.andBtn = QtWidgets.QPushButton("AND")
        self.orBtn = QtWidgets.QPushButton("OR")
        self.xorBtn = QtWidgets.QPushButton("XOR")
        self.subBtn = QtWidgets.QPushButton("SUB")
        bool_layout.addWidget(bool_label)
        for btn in (self.andBtn, self.orBtn, self.xorBtn, self.subBtn):
            btn.setMaximumWidth(50)
            bool_layout.addWidget(btn)
        layout.addLayout(bool_layout)

        self.andBtn.clicked.connect(lambda: self._boolean_op('AND'))
        self.orBtn.clicked.connect(lambda: self._boolean_op('OR'))
        self.xorBtn.clicked.connect(lambda: self._boolean_op('XOR'))
        self.subBtn.clicked.connect(lambda: self._boolean_op('SUBTRACT'))

        self.setWidget(container)
        self._current_window = None

        # Connect to window-switch signal
        if g.m is not None:
            g.m.setCurrentWindowSignal.sig.connect(self._on_window_changed)

    def showEvent(self, event):
        super().showEvent(event)
        # Reconnect in case g.m was set after construction
        if g.m is not None:
            try:
                g.m.setCurrentWindowSignal.sig.connect(self._on_window_changed)
            except (TypeError, RuntimeError):
                pass
        self._on_window_changed()

    def _disconnect_window(self):
        if self._current_window is not None:
            try:
                self._current_window.sigROICreated.disconnect(self._on_roi_created)
            except (TypeError, RuntimeError):
                pass
            try:
                self._current_window.sigROIRemoved.disconnect(self._on_roi_removed)
            except (TypeError, RuntimeError):
                pass
        self._current_window = None

    def _on_window_changed(self):
        self._disconnect_window()
        win = g.win
        if win is None:
            self.tree.clear()
            return
        self._current_window = win
        win.sigROICreated.connect(self._on_roi_created)
        win.sigROIRemoved.connect(self._on_roi_removed)
        self._rebuild_list()

    def _rebuild_list(self):
        self.tree.clear()
        if self._current_window is None:
            return
        for roi in self._current_window.rois:
            self._add_roi_item(roi)

    def _add_roi_item(self, roi):
        item = QtWidgets.QTreeWidgetItem()
        item.setData(0, QtCore.Qt.UserRole, roi)
        item.setText(0, getattr(roi, 'name', roi.kind))
        item.setText(1, roi.kind)
        item.setText(2, "Yes" if roi.traceWindow is not None else "No")
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        self.tree.addTopLevelItem(item)

    def _on_roi_created(self, roi):
        self._add_roi_item(roi)

    def _on_roi_removed(self, roi):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item is not None and item.data(0, QtCore.Qt.UserRole) is roi:
                self.tree.takeTopLevelItem(i)
                break

    def _on_double_click(self, item, column):
        if column == 0:
            # Allow inline rename — the tree is already editable
            self.tree.editItem(item, 0)
            # After edit, update roi name
            roi = item.data(0, QtCore.Qt.UserRole)
            if roi is not None:
                roi.name = item.text(0)

    def _plot_all(self):
        if self._current_window is None:
            return
        for roi in self._current_window.rois:
            if roi.traceWindow is None:
                roi.plot()
        self._rebuild_list()

    def _unplot_all(self):
        if self._current_window is None:
            return
        for roi in self._current_window.rois:
            roi.unplot()
        self._rebuild_list()

    def _delete_selected(self):
        items = self.tree.selectedItems()
        for item in items:
            roi = item.data(0, QtCore.Qt.UserRole)
            if roi is not None:
                roi.delete()

    def _boolean_op(self, operation):
        """Apply boolean operation to two selected ROIs."""
        items = self.tree.selectedItems()
        if len(items) != 2:
            g.alert("Select exactly 2 ROIs for boolean operations.")
            return
        roi_a = items[0].data(0, QtCore.Qt.UserRole)
        roi_b = items[1].data(0, QtCore.Qt.UserRole)
        if roi_a is None or roi_b is None:
            return
        from ..roi import boolean_roi_op
        result = boolean_roi_op(roi_a, roi_b, operation)
        if result is not None:
            result.name = f"{operation}({getattr(roi_a, 'name', 'A')}, {getattr(roi_b, 'name', 'B')})"

    def _export_stats(self):
        if self._current_window is None or len(self._current_window.rois) == 0:
            g.alert("No ROIs to export.")
            return
        from ..utils.roi_stats import compute_roi_stats
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export ROI Stats", "", "CSV Files (*.csv)")
        if not filename:
            return
        import csv
        rows = []
        for roi in self._current_window.rois:
            stats = compute_roi_stats(roi, self._current_window)
            stats['name'] = getattr(roi, 'name', roi.kind)
            stats['type'] = roi.kind
            rows.append(stats)
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        g.m.statusBar().showMessage(f"Exported {len(rows)} ROI stats to {filename}")

    def closeEvent(self, event):
        self._disconnect_window()
        self._destroyed = True
        event.accept()
