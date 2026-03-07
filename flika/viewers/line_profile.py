# -*- coding: utf-8 -*-
"""Line Profile viewer — shows intensity along a line ROI."""
from qtpy import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np


class LineProfileWidget(QtWidgets.QWidget):
    """Plot intensity values along a line ROI.

    Updates live when the ROI moves or the frame changes.
    Supports 2D, 3D (current frame), and 4D windows.
    """

    def __init__(self, line_roi, parent=None):
        super().__init__(parent)
        self.roi = line_roi
        self.window = line_roi.window
        self.setWindowTitle(f"Line Profile: {getattr(line_roi, 'name', 'line')}")
        self.resize(500, 300)

        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(title="Intensity Profile")
        self.plot_widget.setLabel('bottom', 'Distance (pixels)')
        self.plot_widget.setLabel('left', 'Intensity')
        self.curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)

        # Connect signals
        self.roi.sigRegionChanged.connect(self._update)
        self.window.sigTimeChanged.connect(self._update)
        self._update()

    def _update(self, *args):
        xx, yy = self.roi.getMask()
        if len(xx) == 0:
            self.curve.setData([], [])
            return

        win = self.window
        if win.nDims >= 3 and not win.metadata.get('is_rgb', False):
            frame = win.image[win.currentIndex]
        elif win.nDims == 2:
            frame = win.image
        else:
            frame = win.image
            if frame.ndim == 3 and win.metadata.get('is_rgb', False):
                frame = np.mean(frame, axis=2)

        values = frame[xx, yy]
        # Compute cumulative distance along the line
        dx = np.diff(xx.astype(float))
        dy = np.diff(yy.astype(float))
        dist = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
        self.curve.setData(dist, values)

    def closeEvent(self, event):
        try:
            self.roi.sigRegionChanged.disconnect(self._update)
        except (TypeError, RuntimeError):
            pass
        try:
            self.window.sigTimeChanged.disconnect(self._update)
        except (TypeError, RuntimeError):
            pass
        event.accept()
