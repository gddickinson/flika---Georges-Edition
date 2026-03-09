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
from .segmentation import *
from .detection import *
from .alignment import *
from .kymograph import *
from .mask_editor import *
from .export import *
from .linescan import *
from .background_sub import *
from .deconvolution import *
from .stitching import *
from .frap import *
from .fret import *
from .calcium import *
from .spectral import *
from .morphometry import *
from .structures import *
from .simulation import simulate

def _show_overlay_manager():
    """Open the Overlay Manager dock widget from the menu."""
    from .. import global_vars as g
    from ..viewers.overlay_manager import OverlayManagerPanel
    from qtpy.QtCore import Qt
    panel = OverlayManagerPanel.instance(g.m)
    g.m.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, panel)
    panel.show()
    panel.raise_()


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
    analyzeMenu = QtWidgets.QMenu("Analyze")

    def addAction(menu, name, trigger):
        menu.addAction(QtWidgets.QAction(name, menu, triggered=trigger))

    # ==================================================================
    # IMAGE MENU
    # ==================================================================

    # ---- Image > Stacks ----
    stacksMenu = imageMenu.addMenu("Stacks")
    addAction(stacksMenu, "Duplicate", duplicate)
    addAction(stacksMenu, "Trim Frames", trim.gui)
    addAction(stacksMenu, "Remove Frames", frame_remover.gui)
    addAction(stacksMenu, "Deinterlace", deinterleave.gui)
    addAction(stacksMenu, "Z Project", zproject.gui)
    addAction(stacksMenu, "Concatenate Stacks", concatenate_stacks.gui)
    stacksMenu.addSeparator()
    addAction(stacksMenu, "Pixel Binning", pixel_binning.gui)
    addAction(stacksMenu, "Frame Binning", frame_binning.gui)
    addAction(stacksMenu, "Resize", resize.gui)
    addAction(stacksMenu, "Change Data Type", change_datatype.gui)
    stacksMenu.addSeparator()
    addAction(stacksMenu, "Shear Transform", shear_transform.gui)
    addAction(stacksMenu, "Motion Correction", motion_correction.gui)
    addAction(stacksMenu, "Channel Alignment", channel_alignment.gui)
    addAction(stacksMenu, "Stitch Images", stitch_images.gui)

    # ---- Image > Generate ----
    genMenu = imageMenu.addMenu("Generate")
    addAction(genMenu, "Random Image", generate_random_image.gui)
    addAction(genMenu, "Phantom Volume", generate_phantom_volume.gui)

    # ---- Image > Color ----
    colorMenu = imageMenu.addMenu("Color")
    addAction(colorMenu, "Split Channels", split_channels.gui)
    addAction(colorMenu, "Blend Channels", blend_channels.gui)
    addAction(colorMenu, "Channel Compositor", channel_compositor.gui)
    colorMenu.addSeparator()
    addAction(colorMenu, "Convert Color Space", convert_color_space.gui)
    addAction(colorMenu, "Grayscale", grayscale.gui)

    # ---- Image > Overlay ----
    overlayMenu = imageMenu.addMenu("Overlay")
    addAction(overlayMenu, "Background", background.gui)
    addAction(overlayMenu, "Timestamp", time_stamp.gui)
    addAction(overlayMenu, "Scale Bar", scale_bar.gui)
    addAction(overlayMenu, "Grid", grid_overlay.gui)
    addAction(overlayMenu, "Track Overlay", lambda: __import__(
        'flika.viewers.track_overlay',
        fromlist=['show_track_overlay']).show_track_overlay())
    overlayMenu.addSeparator()
    addAction(overlayMenu, "Bake Overlays", bake_overlays_process.gui)

    addAction(imageMenu, "Set Value", set_value.gui)
    addAction(imageMenu, "Image Calculator", image_calculator.gui)

    # ---- Image > Export ----
    exportMenu = imageMenu.addMenu("Export")
    addAction(exportMenu, "Export Video", video_exporter.gui)
    addAction(exportMenu, "Batch Export", batch_export.gui)

    # ==================================================================
    # PROCESS MENU
    # ==================================================================

    # ---- Process > Filters ----
    filtersMenu = processMenu.addMenu("Filters")

    smoothMenu = filtersMenu.addMenu("Smoothing")
    addAction(smoothMenu, "Gaussian Blur", gaussian_blur.gui)
    addAction(smoothMenu, "Mean Filter", mean_filter.gui)
    addAction(smoothMenu, "Median Filter", median_filter.gui)
    addAction(smoothMenu, "Bilateral Filter", bilateral_filter.gui)
    addAction(smoothMenu, "TV Denoising", tv_denoise.gui)

    freqMenu = filtersMenu.addMenu("Frequency Domain")
    addAction(freqMenu, "Butterworth Filter", butterworth_filter.gui)
    addAction(freqMenu, "Fourier Filter", fourier_filter.gui)
    addAction(freqMenu, "Wavelet Filter", wavelet_filter.gui)

    edgeMenu = filtersMenu.addMenu("Edge Detection")
    addAction(edgeMenu, "Sobel", sobel_filter.gui)
    addAction(edgeMenu, "Laplacian", laplacian_filter.gui)
    addAction(edgeMenu, "Gaussian Laplace (LoG)", gaussian_laplace_filter.gui)
    addAction(edgeMenu, "Gaussian Gradient Magnitude",
              gaussian_gradient_magnitude_filter.gui)
    addAction(edgeMenu, "Canny Edge Detector", canny_edge_detector.gui)

    ridgeMenu = filtersMenu.addMenu("Ridge / Texture")
    addAction(ridgeMenu, "Sato Tubeness", sato_tubeness.gui)
    addAction(ridgeMenu, "Meijering Neuriteness", meijering_neuriteness.gui)
    addAction(ridgeMenu, "Hessian Filter", hessian_filter.gui)
    addAction(ridgeMenu, "Gabor Filter", gabor_filter.gui)

    rankMenu = filtersMenu.addMenu("Rank Filters")
    addAction(rankMenu, "Maximum Filter", maximum_filter.gui)
    addAction(rankMenu, "Minimum Filter", minimum_filter.gui)
    addAction(rankMenu, "Percentile Filter", percentile_filter.gui)
    addAction(rankMenu, "Variance Filter", variance_filter.gui)

    addAction(filtersMenu, "Difference of Gaussians", difference_of_gaussians.gui)
    addAction(filtersMenu, "Difference Filter", difference_filter.gui)
    addAction(filtersMenu, "Boxcar Differential", boxcar_differential_filter.gui)
    filtersMenu.addSeparator()
    addAction(filtersMenu, "Background Subtraction", background_subtract.gui)
    addAction(filtersMenu, "Bleach Correction", bleach_correction.gui)
    addAction(filtersMenu, "Flash Remover", flash_remover.gui)

    # ---- Process > Math ----
    mathMenu = processMenu.addMenu("Math")
    addAction(mathMenu, "Multiply", multiply.gui)
    addAction(mathMenu, "Divide", divide.gui)
    addAction(mathMenu, "Subtract", subtract.gui)
    addAction(mathMenu, "Power", power.gui)
    addAction(mathMenu, "Square Root", sqrt.gui)
    addAction(mathMenu, "Absolute Value", absolute_value.gui)
    mathMenu.addSeparator()
    addAction(mathMenu, "Ratio By Baseline", ratio.gui)
    addAction(mathMenu, "Subtract Trace", subtract_trace.gui)
    addAction(mathMenu, "Divide Trace", divide_trace.gui)
    mathMenu.addSeparator()
    addAction(mathMenu, "Histogram Equalize", histogram_equalize.gui)
    addAction(mathMenu, "Normalize", normalize.gui)

    # ---- Process > Binary / Morphology ----
    binaryMenu = processMenu.addMenu("Binary / Morphology")

    threshMenu = binaryMenu.addMenu("Thresholding")
    addAction(threshMenu, "Threshold", threshold.gui)
    addAction(threshMenu, "Adaptive Threshold", adaptive_threshold.gui)
    addAction(threshMenu, "Hysteresis Threshold", hysteresis_threshold.gui)
    addAction(threshMenu, "Multi-Otsu Threshold", multi_otsu_threshold.gui)

    morphMenu = binaryMenu.addMenu("Morphology")
    addAction(morphMenu, "Binary Erosion", binary_erosion.gui)
    addAction(morphMenu, "Binary Dilation", binary_dilation.gui)
    addAction(morphMenu, "Grayscale Opening", grayscale_opening.gui)
    addAction(morphMenu, "Grayscale Closing", grayscale_closing.gui)
    addAction(morphMenu, "Morphological Gradient", morphological_gradient.gui)
    morphMenu.addSeparator()
    addAction(morphMenu, "H-Maxima", h_maxima.gui)
    addAction(morphMenu, "H-Minima", h_minima.gui)
    addAction(morphMenu, "Area Opening", area_opening.gui)
    addAction(morphMenu, "Area Closing", area_closing.gui)
    addAction(morphMenu, "Flood Fill", flood_fill_process.gui)

    addAction(binaryMenu, "Logically Combine", logically_combine.gui)
    addAction(binaryMenu, "Remove Small Blobs", remove_small_blobs.gui)
    addAction(binaryMenu, "Remove Small Holes", remove_small_holes.gui)
    addAction(binaryMenu, "Generate ROIs", generate_rois.gui)
    addAction(binaryMenu, "Mask Editor", mask_editor.gui)

    # ---- Process > Segmentation ----
    segMenu = processMenu.addMenu("Segmentation")
    addAction(segMenu, "Connected Components", connected_components.gui)
    addAction(segMenu, "Region Properties", region_properties.gui)
    addAction(segMenu, "Clear Border", clear_border.gui)
    addAction(segMenu, "Expand Labels", expand_labels.gui)
    segMenu.addSeparator()
    addAction(segMenu, "Watershed Segmentation", watershed_segmentation.gui)
    addAction(segMenu, "Distance Transform", distance_transform.gui)
    addAction(segMenu, "Random Walker", random_walker_seg.gui)
    addAction(segMenu, "SLIC Superpixels", slic_superpixels.gui)
    segMenu.addSeparator()
    addAction(segMenu, "Find Boundaries", find_boundaries.gui)
    addAction(segMenu, "Find Contours", find_contours_process.gui)

    # ---- Process > Detection ----
    detectMenu = processMenu.addMenu("Detection")
    addAction(detectMenu, "Blob Detection (LoG)", blob_detection_log.gui)
    addAction(detectMenu, "Blob Detection (DoH)", blob_detection_doh.gui)
    addAction(detectMenu, "Peak Local Max", peak_local_max.gui)
    addAction(detectMenu, "Template Matching", template_match.gui)
    addAction(detectMenu, "Local Maxima", local_maxima_detect.gui)
    detectMenu.addSeparator()
    addAction(detectMenu, "Analyze Particles", analyze_particles.gui)

    # ---- Process > Deconvolution ----
    deconvMenu = processMenu.addMenu("Deconvolution")
    addAction(deconvMenu, "Richardson-Lucy", richardson_lucy.gui)
    addAction(deconvMenu, "Wiener Deconvolution", wiener_deconvolution.gui)
    addAction(deconvMenu, "Generate PSF", generate_psf.gui)

    # ---- Process > Structures ----
    structMenu = processMenu.addMenu("Structures")
    structMenu_network = structMenu.addMenu("Network Analysis")
    addAction(structMenu_network, "Frangi Vesselness", frangi_vesselness.gui)
    addAction(structMenu_network, "Skeletonize", skeletonize_process.gui)
    addAction(structMenu_network, "Medial Axis", medial_axis_process.gui)
    addAction(structMenu_network, "Skeleton Analysis", skeleton_analysis.gui)
    structMenu.addSeparator()
    addAction(structMenu, "Hough Lines", hough_lines.gui)
    addAction(structMenu, "Hough Circles", hough_circles.gui)
    addAction(structMenu, "Corner Detection", corner_detection.gui)
    structMenu.addSeparator()
    addAction(structMenu, "Local Binary Pattern",
              local_binary_pattern_process.gui)
    addAction(structMenu, "Structure Tensor", structure_tensor_analysis.gui)

    # ==================================================================
    # ANALYZE MENU
    # ==================================================================

    # ---- Analyze > Measure ----
    measureMenu = analyzeMenu.addMenu("Measure")
    addAction(measureMenu, "Measure", measure.gui)
    addAction(measureMenu, "Linescan", linescan.gui)
    addAction(measureMenu, "Kymograph", kymograph.gui)
    addAction(measureMenu, "Counting Tool", counting_tool.gui)

    # ---- Analyze > Colocalization ----
    addAction(analyzeMenu, "Colocalization", colocalization.gui)
    addAction(analyzeMenu, "Morphometry", morphometry_analysis.gui)

    # ---- Analyze > SPT Analysis ----
    sptMenu = analyzeMenu.addMenu("SPT Analysis")
    addAction(sptMenu, "SPT Control Panel", spt_analysis.gui)
    addAction(sptMenu, "Detect Particles", detect_particles.gui)
    addAction(sptMenu, "Link Particles", link_particles_process.gui)
    sptMenu.addSeparator()
    addAction(sptMenu, "Results Table", _show_results_table)

    # ---- Analyze > Dynamics ----
    dynamicsMenu = analyzeMenu.addMenu("Dynamics")
    addAction(dynamicsMenu, "FRAP Analysis", frap_analysis.gui)
    addAction(dynamicsMenu, "FRET Analysis", fret_analysis.gui)
    addAction(dynamicsMenu, "Calcium Analysis", calcium_analysis.gui)
    addAction(dynamicsMenu, "Spectral Unmixing", spectral_unmixing.gui)

    # ==================================================================
    # SIMULATION MENU
    # ==================================================================

    simulationMenu = QtWidgets.QMenu("Simulation")
    addAction(simulationMenu, "Simulation Builder...", simulate.gui)
    presetMenu = simulationMenu.addMenu("Quick Presets")
    from ..simulation.presets import PRESETS as _SIM_PRESETS
    for _preset_name in _SIM_PRESETS:
        addAction(presetMenu, _preset_name,
                  lambda checked=False, p=_preset_name: simulate.run(preset=p))
    simulationMenu.addSeparator()
    addAction(simulationMenu, "Run Benchmarks...", simulate.run_benchmarks_gui)

    g.menus.append(imageMenu)
    g.menus.append(processMenu)
    g.menus.append(analyzeMenu)
    g.menus.append(simulationMenu)
    logger.debug("Completed 'process.__init__.setup_menus()'")

logger.debug("Completed 'reading process/__init__.py'")
