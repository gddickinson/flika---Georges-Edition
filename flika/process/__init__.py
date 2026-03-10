"""
Process module for flika - provides image processing operations.
"""

# Import specific functions/classes from each module
from flika.process.binary import (
    adaptive_threshold,
    analyze_particles,
    area_closing,
    area_opening,
    binary_dilation,
    binary_erosion,
    canny_edge_detector,
    flood_fill_process,
    generate_rois,
    grayscale_closing,
    grayscale_opening,
    h_maxima,
    h_minima,
    hysteresis_threshold,
    logically_combine,
    morphological_gradient,
    multi_otsu_threshold,
    remove_small_blobs,
    remove_small_holes,
    threshold,
)
from flika.process.color import (
    blend_channels,
    convert_color_space,
    grayscale,
    split_channels,
)
from flika.process.file_ import close, open_file
from flika.process.filters import (
    bilateral_filter,
    bleach_correction,
    boxcar_differential_filter,
    butterworth_filter,
    difference_filter,
    difference_of_gaussians,
    flash_remover,
    fourier_filter,
    gabor_filter,
    gaussian_blur,
    gaussian_gradient_magnitude_filter,
    gaussian_laplace_filter,
    hessian_filter,
    laplacian_filter,
    maximum_filter,
    mean_filter,
    median_filter,
    meijering_neuriteness,
    minimum_filter,
    percentile_filter,
    sato_tubeness,
    sobel_filter,
    tv_denoise,
    variance_filter,
    wavelet_filter,
)
from flika.process.math_ import (
    absolute_value,
    divide,
    divide_trace,
    histogram_equalize,
    multiply,
    normalize,
    power,
    ratio,
    sqrt,
    subtract,
    subtract_trace,
)
from flika.process.measure import measure
from flika.process.overlay import (
    background,
    bake_overlays_process,
    counting_tool,
    grid_overlay,
    scale_bar,
    time_stamp,
)
from flika.process.roi import set_value
from flika.process.stacks import (
    change_datatype,
    concatenate_stacks,
    deinterleave,
    duplicate,
    flip_image,
    frame_binning,
    frame_remover,
    generate_phantom_volume,
    generate_random_image,
    image_calculator,
    motion_correction,
    pixel_binning,
    resize,
    rotate_90,
    rotate_custom,
    shear_transform,
    trim,
    zproject,
)
from flika.process.alignment import channel_alignment
from flika.process.background_sub import background_subtract, scaled_average_subtract
from flika.process.calcium import calcium_analysis
from flika.process.colocalization import colocalization
from flika.process.compositing import channel_compositor
from flika.process.deconvolution import generate_psf, richardson_lucy, wiener_deconvolution
from flika.process.detection import (
    blob_detection_doh,
    blob_detection_log,
    local_maxima_detect,
    peak_local_max,
    template_match,
)
from flika.process.export import batch_export, video_exporter
from flika.process.frap import frap_analysis
from flika.process.fret import fret_analysis
from flika.process.kymograph import kymograph
from flika.process.linescan import linescan
from flika.process.mask_editor import mask_editor
from flika.process.morphometry import morphometry_analysis
from flika.process.segmentation import (
    clear_border,
    connected_components,
    expand_labels,
    find_boundaries,
    find_contours_process,
    random_walker_seg,
    region_properties,
    slic_superpixels,
)
from flika.process.simulation import simulate
from flika.process.spectral import spectral_unmixing
from flika.process.spt import detect_particles, link_particles_process, spt_analysis
from flika.process.stitching import stitch_images
from flika.process.structures import (
    corner_detection,
    frangi_vesselness,
    hough_circles,
    hough_lines,
    local_binary_pattern_process,
    medial_axis_process,
    skeleton_analysis,
    skeletonize_process,
    structure_tensor_analysis,
)
from flika.process.watershed import distance_transform, watershed_segmentation

# Define what's available when using `from flika.process import *`
__all__ = [
    # binary module
    "threshold",
    "remove_small_blobs",
    "adaptive_threshold",
    "logically_combine",
    "binary_dilation",
    "binary_erosion",
    "generate_rois",
    "canny_edge_detector",
    "analyze_particles",
    "grayscale_opening",
    "grayscale_closing",
    "morphological_gradient",
    "h_maxima",
    "h_minima",
    "area_opening",
    "area_closing",
    "remove_small_holes",
    "flood_fill_process",
    "hysteresis_threshold",
    "multi_otsu_threshold",
    # color module
    "split_channels",
    "blend_channels",
    "convert_color_space",
    "grayscale",
    # file_ module
    "open_file",
    "close",
    # filters module
    "gaussian_blur",
    "difference_of_gaussians",
    "mean_filter",
    "variance_filter",
    "median_filter",
    "butterworth_filter",
    "boxcar_differential_filter",
    "wavelet_filter",
    "difference_filter",
    "fourier_filter",
    "bilateral_filter",
    "sobel_filter",
    "laplacian_filter",
    "gaussian_laplace_filter",
    "gaussian_gradient_magnitude_filter",
    "sato_tubeness",
    "meijering_neuriteness",
    "hessian_filter",
    "gabor_filter",
    "maximum_filter",
    "minimum_filter",
    "percentile_filter",
    "tv_denoise",
    "flash_remover",
    "bleach_correction",
    # math_ module
    "subtract",
    "multiply",
    "divide",
    "power",
    "sqrt",
    "ratio",
    "absolute_value",
    "subtract_trace",
    "divide_trace",
    "histogram_equalize",
    "normalize",
    # measure module
    "measure",
    # overlay module
    "time_stamp",
    "background",
    "scale_bar",
    "grid_overlay",
    "counting_tool",
    "bake_overlays_process",
    # roi module
    "set_value",
    # stacks module
    "deinterleave",
    "trim",
    "zproject",
    "image_calculator",
    "pixel_binning",
    "frame_binning",
    "resize",
    "concatenate_stacks",
    "duplicate",
    "generate_random_image",
    "generate_phantom_volume",
    "change_datatype",
    "frame_remover",
    "rotate_90",
    "rotate_custom",
    "flip_image",
    "shear_transform",
    "motion_correction",
    # alignment module
    "channel_alignment",
    # background_sub module
    "background_subtract",
    "scaled_average_subtract",
    # calcium module
    "calcium_analysis",
    # colocalization module
    "colocalization",
    # compositing module
    "channel_compositor",
    # deconvolution module
    "richardson_lucy",
    "wiener_deconvolution",
    "generate_psf",
    # detection module
    "blob_detection_log",
    "blob_detection_doh",
    "peak_local_max",
    "template_match",
    "local_maxima_detect",
    "analyze_particles",
    # export module
    "video_exporter",
    "batch_export",
    # frap module
    "frap_analysis",
    # fret module
    "fret_analysis",
    # kymograph module
    "kymograph",
    # linescan module
    "linescan",
    # mask_editor module
    "mask_editor",
    # morphometry module
    "morphometry_analysis",
    # segmentation module
    "connected_components",
    "region_properties",
    "clear_border",
    "expand_labels",
    "slic_superpixels",
    "find_boundaries",
    "find_contours_process",
    "random_walker_seg",
    # simulation module
    "simulate",
    # spectral module
    "spectral_unmixing",
    # spt module
    "spt_analysis",
    "detect_particles",
    "link_particles_process",
    # stitching module
    "stitch_images",
    # structures module
    "frangi_vesselness",
    "skeletonize_process",
    "medial_axis_process",
    "skeleton_analysis",
    "hough_lines",
    "hough_circles",
    "corner_detection",
    "local_binary_pattern_process",
    "structure_tensor_analysis",
    # watershed module
    "distance_transform",
    "watershed_segmentation",
]


def _show_results_table():
    """Open the SPT Results Table dock widget from the menu."""
    import flika.global_vars as g
    from flika.viewers.results_table import ResultsTableWidget
    from qtpy.QtCore import Qt

    table = ResultsTableWidget.instance(g.m)
    g.m.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, table)

    # Populate with current window's data if available
    win = getattr(g, "win", None)
    if win is not None:
        spt = win.metadata.get("spt", {}) if hasattr(win, "metadata") else {}
        pdata = spt.get("particle_data")
        if pdata is not None:
            table.set_particle_data(pdata)

    table.show()
    table.raise_()


def setup_menus():
    """Set up the flika menu structure for process operations."""
    import flika.global_vars as g

    if len(g.menus) > 0:
        print("flika menubar already initialized.")
        return
    from qtpy import QtWidgets

    imageMenu = QtWidgets.QMenu("Image")
    processMenu = QtWidgets.QMenu("Process")
    analyzeMenu = QtWidgets.QMenu("Analyze")
    simulationMenu = QtWidgets.QMenu("Simulation")

    def addAction(menu, name, trigger):
        menu.addAction(QtWidgets.QAction(name, menu, triggered=trigger))

    # ==================================================================
    # IMAGE MENU
    # ==================================================================

    # ---- Image > Stacks ----
    stacksMenu = imageMenu.addMenu("Stacks")

    addAction(stacksMenu, "Duplicate", duplicate)
    addAction(stacksMenu, "Generate Random Image", generate_random_image.gui)
    addAction(stacksMenu, "Trim Frames", trim.gui)
    addAction(stacksMenu, "Remove Frames", frame_remover.gui)
    addAction(stacksMenu, "Deinterlace", deinterleave.gui)
    addAction(stacksMenu, "Z Project", zproject.gui)
    addAction(stacksMenu, "Pixel Binning", pixel_binning.gui)
    addAction(stacksMenu, "Frame Binning", frame_binning.gui)
    addAction(stacksMenu, "Resize", resize.gui)
    addAction(stacksMenu, "Concatenate Stacks", concatenate_stacks.gui)
    addAction(stacksMenu, "Change Data Type", change_datatype.gui)
    stacksMenu.addSeparator()
    transformMenu = stacksMenu.addMenu("Transform")
    addAction(transformMenu, "Rotate 90\u00b0", rotate_90.gui)
    addAction(transformMenu, "Rotate Custom", rotate_custom.gui)
    addAction(transformMenu, "Flip Image", flip_image.gui)
    addAction(transformMenu, "Shear Transform", shear_transform.gui)
    stacksMenu.addSeparator()
    addAction(stacksMenu, "Generate Phantom Volume", generate_phantom_volume.gui)
    addAction(stacksMenu, "Motion Correction", motion_correction.gui)
    addAction(stacksMenu, "Channel Alignment", channel_alignment.gui)
    addAction(stacksMenu, "Stitch Images", stitch_images.gui)

    # ---- Image > Color ----
    colorMenu = imageMenu.addMenu("Color")
    addAction(colorMenu, "Split Channels", split_channels.gui)
    addAction(colorMenu, "Blend Channels", blend_channels.gui)
    addAction(colorMenu, "Channel Compositor", channel_compositor.gui)
    colorMenu.addSeparator()
    addAction(colorMenu, "Convert Color Space", convert_color_space.gui)
    addAction(colorMenu, "Grayscale", grayscale.gui)

    # ---- Image > Overlay ----
    addAction(imageMenu, "Measure", measure.gui)
    addAction(imageMenu, "Set Value", set_value.gui)
    overlayMenu = imageMenu.addMenu("Overlay")
    addAction(overlayMenu, "Background", background.gui)
    addAction(overlayMenu, "Timestamp", time_stamp.gui)
    addAction(overlayMenu, "Scale Bar", scale_bar.gui)
    addAction(overlayMenu, "Grid", grid_overlay.gui)
    addAction(
        overlayMenu,
        "Track Overlay",
        lambda: __import__(
            "flika.viewers.track_overlay", fromlist=["show_track_overlay"]
        ).show_track_overlay(),
    )
    overlayMenu.addSeparator()
    addAction(overlayMenu, "Bake Overlays", bake_overlays_process.gui)

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
    addAction(
        edgeMenu,
        "Gaussian Gradient Magnitude",
        gaussian_gradient_magnitude_filter.gui,
    )
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
    addAction(filtersMenu, "Scaled Average Subtract", scaled_average_subtract.gui)
    addAction(filtersMenu, "Bleach Correction", bleach_correction.gui)
    addAction(filtersMenu, "Flash Remover", flash_remover.gui)

    # ---- Process > Math ----
    mathMenu = processMenu.addMenu("Math")
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

    processMenu.addAction(
        QtWidgets.QAction(
            "Image Calculator", processMenu, triggered=image_calculator.gui
        )
    )

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
    addAction(structMenu, "Local Binary Pattern", local_binary_pattern_process.gui)
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

    # ---- Analyze > Colocalization / Morphometry ----
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

    addAction(simulationMenu, "Simulation Builder...", simulate.gui)
    presetMenu = simulationMenu.addMenu("Quick Presets")
    from flika.simulation.presets import PRESETS as _SIM_PRESETS

    for _preset_name in _SIM_PRESETS:
        addAction(
            presetMenu,
            _preset_name,
            lambda checked=False, p=_preset_name: simulate.run(preset=p),
        )
    simulationMenu.addSeparator()
    addAction(simulationMenu, "Run Benchmarks...", simulate.run_benchmarks_gui)

    g.menus.append(imageMenu)
    g.menus.append(processMenu)
    g.menus.append(analyzeMenu)
    g.menus.append(simulationMenu)
