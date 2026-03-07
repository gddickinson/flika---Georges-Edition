# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/linescan.py'")
import numpy as np
from skimage.measure import profile_line
from qtpy import QtWidgets, QtCore
import pyqtgraph as pg
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox

__all__ = ['linescan']


class ProfilePlotWindow(QtWidgets.QWidget):
    """A simple window that displays a line profile plot using pyqtgraph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Line Profile')
        self.resize(500, 300)
        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Distance', units='pixels')
        self.plot_widget.setLabel('left', 'Intensity')
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        self._curve = self.plot_widget.plot(pen='y')

    def update_profile(self, profile):
        """Update the displayed profile data."""
        if profile is not None and len(profile) > 0:
            self._curve.setData(profile)
        else:
            self._curve.setData([])


class Linescan(BaseProcess):
    """linescan(line_width=1, integration='Mean', keepSourceWindow=False)

    Extract intensity profiles along a line ROI for each frame in a stack.

    For 2D images, produces a single 1D profile. For 3D stacks, produces
    a 2D array (nFrames x nPixels) representing the linescan kymograph.

    Requires a line ROI to be drawn on the current window.

    Parameters:
        line_width (int): Width of the line for profile integration (1-50, default 1).
        integration (str): How to combine pixels across the line width -
            'Mean', 'Max', or 'Min'.
        keepSourceWindow (bool): Whether to keep the source window open.

    Returns:
        flika.window.Window: A new window containing the linescan result.
    """

    def __init__(self):
        super().__init__()
        self._plot_window = None
        self._roi_connection = None
        self._time_connection = None

    def gui(self):
        self.gui_reset()
        line_width = SliderLabel(0)
        line_width.setRange(1, 50)
        line_width.setValue(1)
        integration = ComboBox()
        integration.addItems(['Mean', 'Max', 'Min'])
        self.items.append({'name': 'line_width', 'string': 'Line Width (pixels)', 'object': line_width})
        self.items.append({'name': 'integration', 'string': 'Integration Method', 'object': integration})
        super().gui()

        # Add a "Plot Profile" button to the dialog
        plot_btn = QtWidgets.QPushButton('Plot Profile')
        plot_btn.clicked.connect(self._show_profile_plot)
        self.ui.layout.insertWidget(self.ui.layout.count() - 1, plot_btn)

    def _show_profile_plot(self):
        """Open a live profile plot window that updates with ROI and frame changes."""
        if self._plot_window is not None:
            self._plot_window.close()
            self._disconnect_signals()

        self._plot_window = ProfilePlotWindow()
        self._plot_window.show()

        # Connect to ROI and frame signals for live updates
        self._connect_signals()
        # Draw initial profile
        self._update_plot()

    def _connect_signals(self):
        """Connect to ROI and time change signals for live profile updates."""
        self._disconnect_signals()
        if g.win is None:
            return
        roi = self._get_line_roi()
        if roi is not None and hasattr(roi, 'sigRegionChanged'):
            self._roi_connection = roi
            roi.sigRegionChanged.connect(self._update_plot)
        if hasattr(g.win, 'sigTimeChanged'):
            self._time_connection = g.win
            g.win.sigTimeChanged.connect(self._update_plot)

    def _disconnect_signals(self):
        """Disconnect any existing signal connections."""
        if self._roi_connection is not None:
            try:
                self._roi_connection.sigRegionChanged.disconnect(self._update_plot)
            except (TypeError, RuntimeError):
                pass
            self._roi_connection = None
        if self._time_connection is not None:
            try:
                self._time_connection.sigTimeChanged.disconnect(self._update_plot)
            except (TypeError, RuntimeError):
                pass
            self._time_connection = None

    def _update_plot(self):
        """Update the profile plot with the current frame and ROI."""
        if self._plot_window is None or g.win is None:
            return
        roi = self._get_line_roi()
        if roi is None:
            return

        line_width = int(self.getValue('line_width'))
        integration = self.getValue('integration')

        image = g.win.image
        if image.ndim == 3:
            frame_2d = image[g.win.currentIndex]
        elif image.ndim == 2:
            frame_2d = image
        else:
            return

        profile = self._get_profile(frame_2d, roi, line_width, integration)
        if profile is not None:
            self._plot_window.update_profile(profile)

    def _get_line_roi(self):
        """Get the current line ROI from the active window."""
        if g.win is None:
            return None
        rois = g.win.rois
        if not rois:
            return None
        # Find the last line ROI (check for line-type ROIs)
        for roi in reversed(rois):
            if hasattr(roi, 'pts') and len(roi.pts) >= 2:
                return roi
        return None

    @staticmethod
    def _get_profile(image_2d, roi, line_width, integration):
        """Extract an intensity profile along a line ROI.

        Parameters:
            image_2d (np.ndarray): 2D image array.
            roi: ROI object with pts attribute (list of [x, y] points).
            line_width (int): Width of the line for profile integration.
            integration (str): 'Mean', 'Max', or 'Min'.

        Returns:
            np.ndarray: 1D intensity profile along the line.
        """
        pts = roi.pts
        if pts is None or len(pts) < 2:
            return None

        # Convert from (x, y) to (row, col) for skimage
        start = (pts[0][1], pts[0][0])
        end = (pts[-1][1], pts[-1][0])

        # Select reduce function based on integration method
        if integration == 'Max':
            reduce_func = np.max
        elif integration == 'Min':
            reduce_func = np.min
        else:
            reduce_func = np.mean

        try:
            profile = profile_line(
                image_2d.astype(np.float64),
                start, end,
                linewidth=line_width,
                mode='constant',
                reduce_func=reduce_func
            )
        except Exception as e:
            logger.error("Error extracting line profile: {}".format(str(e)))
            return None

        return profile

    def __call__(self, line_width=1, integration='Mean', keepSourceWindow=False):
        self.start(keepSourceWindow)

        roi = self._get_line_roi()
        if roi is None:
            g.alert("No line ROI found. Please draw a line ROI on the image first.")
            self.oldwindow.reset()
            return None

        image = self.tif

        if image.ndim == 2:
            # Single frame: extract one profile
            profile = self._get_profile(image, roi, line_width, integration)
            if profile is None:
                g.alert("Failed to extract line profile.")
                self.oldwindow.reset()
                return None
            self.newtif = profile[np.newaxis, :]  # make 2D for window display

        elif image.ndim == 3:
            # Stack: extract profile for each frame
            nframes = image.shape[0]
            profiles = []
            for i in range(nframes):
                frame = image[i]
                profile = self._get_profile(frame, roi, line_width, integration)
                if profile is not None:
                    profiles.append(profile)
                else:
                    # If profile extraction fails, use zeros
                    if profiles:
                        profiles.append(np.zeros(len(profiles[0])))
                    else:
                        profiles.append(np.zeros(1))
            # Ensure all profiles have the same length
            max_len = max(len(p) for p in profiles)
            padded = []
            for p in profiles:
                if len(p) < max_len:
                    padded.append(np.pad(p, (0, max_len - len(p)), mode='constant'))
                else:
                    padded.append(p)
            self.newtif = np.array(padded)

        else:
            g.alert("Linescan requires a 2D or 3D image.")
            self.oldwindow.reset()
            return None

        self.newname = self.oldname + ' - Linescan'
        return self.end()

    def preview(self):
        pass


linescan = Linescan()

logger.debug("Completed 'reading process/linescan.py'")
