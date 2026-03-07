# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/__init__.py'")

from .stacks import *
from .math_ import *
from .filters import *
from .binary import *
from .roi import *
from .measure import *
from .color import *
from .overlay import *
from .file_ import *
from .compositing import *
from .colocalization import *
from .watershed import *
from .spt import *

def _show_results_table():
    """Open the SPT Results Table dock widget from the menu."""
    from .. import global_vars as g
    from ..viewers.results_table import ResultsTableWidget
    from qtpy.QtCore import Qt
    table = ResultsTableWidget.instance(g.m)
    g.m.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, table)

    # Populate with current window's data if available
    win = getattr(g, 'win', None)
    if win is not None:
        spt = win.metadata.get('spt', {}) if hasattr(win, 'metadata') else {}
        pdata = spt.get('particle_data')
        if pdata is not None:
            table.set_particle_data(pdata)

    table.show()
    table.raise_()


def setup_menus():
    logger.debug("Started 'process.__init__.setup_menus()'")
    from .. import global_vars as g
    if len(g.menus) > 0:
        logger.info("flika menubar already initialized.")
        return
    from qtpy import QtGui, QtWidgets
    imageMenu = QtWidgets.QMenu("Image")
    processMenu = QtWidgets.QMenu("Process")

    def addAction(menu, name, trigger):
        menu.addAction(QtWidgets.QAction(name, menu, triggered=trigger))

    stacksMenu = imageMenu.addMenu("Stacks")

    addAction(stacksMenu, "Duplicate", duplicate)
    addAction(stacksMenu, "Generate Random Image", generate_random_image.gui)
    addAction(stacksMenu, "Generate Phantom Volume", generate_phantom_volume.gui)
    addAction(stacksMenu, "Trim Frames", trim.gui)
    addAction(stacksMenu, "Deinterlace", deinterleave.gui)
    addAction(stacksMenu, "Z Project", zproject.gui)
    addAction(stacksMenu, "Pixel Binning", pixel_binning.gui)
    addAction(stacksMenu, "Frame Binning", frame_binning.gui)
    addAction(stacksMenu, "Resize", resize.gui)
    addAction(stacksMenu, "Concatenate Stacks", concatenate_stacks.gui)
    addAction(stacksMenu, "Change Data Type", change_datatype.gui)
    stacksMenu.addSeparator()
    addAction(stacksMenu, "Shear Transform", shear_transform.gui)
    addAction(stacksMenu, "Motion Correction", motion_correction.gui)

    colorMenu = imageMenu.addMenu("Color")
    addAction(colorMenu, "Split Channels", split_channels.gui)
    addAction(colorMenu, "Blend Channels", blend_channels.gui)
    addAction(colorMenu, "Channel Compositor", channel_compositor.gui)

    addAction(imageMenu, "Measure", measure.gui)
    addAction(imageMenu, "Set Value", set_value.gui)
    overlayMenu = imageMenu.addMenu("Overlay")
    addAction(overlayMenu, "Background", background.gui)
    addAction(overlayMenu, "Timestamp", time_stamp.gui)
    addAction(overlayMenu, "Scale Bar", scale_bar.gui)
    addAction(overlayMenu, "Track Overlay", lambda: __import__('flika.viewers.track_overlay', fromlist=['show_track_overlay']).show_track_overlay())

    binaryMenu = processMenu.addMenu("Binary")
    mathMenu = processMenu.addMenu("Math")
    filtersMenu = processMenu.addMenu("Filters")
    processMenu.addAction(QtWidgets.QAction("Image Calculator", processMenu, triggered=image_calculator.gui))

    colocMenu = processMenu.addMenu("Colocalization")
    addAction(colocMenu, "Colocalization Analysis", colocalization.gui)

    addAction(binaryMenu, "Threshold", threshold.gui)
    addAction(binaryMenu, "Adaptive Threshold", adaptive_threshold.gui)
    addAction(binaryMenu, "Canny Edge Detector", canny_edge_detector.gui)
    binaryMenu.addSeparator()
    addAction(binaryMenu, "Logically Combine", logically_combine.gui)
    addAction(binaryMenu, "Remove Small Blobs", remove_small_blobs.gui)
    addAction(binaryMenu, "Binary Erosion", binary_erosion.gui)
    addAction(binaryMenu, "Binary Dilation", binary_dilation.gui)
    addAction(binaryMenu, "Generate ROIs", generate_rois.gui)
    binaryMenu.addSeparator()
    addAction(binaryMenu, "Analyze Particles", analyze_particles.gui)
    binaryMenu.addSeparator()
    addAction(binaryMenu, "Distance Transform", distance_transform.gui)
    addAction(binaryMenu, "Watershed Segmentation", watershed_segmentation.gui)

    addAction(mathMenu, "Multiply", multiply.gui)
    addAction(mathMenu, "Divide", divide.gui)
    addAction(mathMenu, "Subtract", subtract.gui)
    addAction(mathMenu, "Power", power.gui)
    addAction(mathMenu, "Square Root", sqrt.gui)
    addAction(mathMenu, "Ratio By Baseline", ratio.gui)
    addAction(mathMenu, "Absolute Value", absolute_value.gui)
    addAction(mathMenu, "Subtract Trace", subtract_trace.gui)
    addAction(mathMenu, "Divide Trace", divide_trace.gui)
    mathMenu.addSeparator()
    addAction(mathMenu, "Histogram Equalize", histogram_equalize.gui)
    addAction(mathMenu, "Normalize", normalize.gui)

    addAction(filtersMenu, "Gaussian Blur", gaussian_blur.gui)
    addAction(filtersMenu, "Difference of Gaussians", difference_of_gaussians.gui)
    filtersMenu.addSeparator()
    addAction(filtersMenu, "Butterworth Filter", butterworth_filter.gui)
    addAction(filtersMenu, "Mean Filter", mean_filter.gui)
    addAction(filtersMenu, "Variance Filter", variance_filter.gui)
    addAction(filtersMenu, "Median Filter", median_filter.gui)
    addAction(filtersMenu, "Fourier Filter", fourier_filter.gui)
    addAction(filtersMenu, "Difference Filter", difference_filter.gui)
    addAction(filtersMenu, "Boxcar Differential", boxcar_differential_filter.gui)
    addAction(filtersMenu, "Wavelet Filter", wavelet_filter.gui)
    addAction(filtersMenu, "Bilateral Filter", bilateral_filter.gui)

    sptMenu = processMenu.addMenu("SPT Analysis")
    addAction(sptMenu, "SPT Control Panel", spt_analysis.gui)
    addAction(sptMenu, "Detect Particles", detect_particles.gui)
    addAction(sptMenu, "Link Particles", link_particles_process.gui)
    sptMenu.addSeparator()
    addAction(sptMenu, "Results Table", _show_results_table)

    g.menus.append(imageMenu)
    g.menus.append(processMenu)
    logger.debug("Completed 'process.__init__.setup_menus()'")

logger.debug("Completed 'reading process/__init__.py'")