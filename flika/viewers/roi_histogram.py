# -*- coding: utf-8 -*-
"""ROI Histogram viewer — shows live histogram of pixel values within an ROI."""
from qtpy import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np


class ROIHistogramWidget(QtWidgets.QWidget):
    """Live histogram of pixel intensities within an ROI.

    Updates on frame change and ROI movement.
    """

    def __init__(self, roi, parent=None):
        super().__init__(parent)
        self.roi = roi
        self.window = roi.window
        self.setWindowTitle(f"ROI Histogram: {getattr(roi, 'name', roi.kind)}")
        self.resize(500, 350)

        layout = QtWidgets.QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget(title="Pixel Histogram")
        self.plot_widget.setLabel('bottom', 'Intensity')
        self.plot_widget.setLabel('left', 'Count')
        self.bar_item = pg.BarGraphItem(x=[], height=[], width=1, brush='y')
        self.plot_widget.addItem(self.bar_item)
        layout.addWidget(self.plot_widget)

        # Bin count slider
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.addWidget(QtWidgets.QLabel("Bins:"))
        self.bin_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bin_slider.setRange(8, 512)
        self.bin_slider.setValue(256)
        self.bin_slider.valueChanged.connect(self._update)
        self.bin_label = QtWidgets.QLabel("256")
        ctrl_layout.addWidget(self.bin_slider)
        ctrl_layout.addWidget(self.bin_label)
        layout.addLayout(ctrl_layout)

        # Connect signals
        self.roi.sigRegionChanged.connect(self._update)
        self.window.sigTimeChanged.connect(self._update)
        self._update()

    def _update(self, *args):
        s1, s2 = self.roi.getMask()
        if np.size(s1) == 0:
            self.bar_item.setOpts(x=[], height=[], width=1)
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

        pixels = frame[s1, s2].ravel()
        nbins = self.bin_slider.value()
        self.bin_label.setText(str(nbins))

        counts, edges = np.histogram(pixels, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = edges[1] - edges[0] if len(edges) > 1 else 1
        self.bar_item.setOpts(x=centers, height=counts, width=width * 0.9)

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
