# -*- coding: utf-8 -*-
"""
SPT Detection & Linking Control Groups
=======================================

Collapsible QGroupBox widgets exposing the full parameter sets for:
- ThunderSTORM detection (filter, detector, fitter, camera model)
- U-Track LAP linking (motion model, FRF, adaptive search, gap closing)
- Trackpy linking (link type, adaptive params)

These widgets are embedded in the SPTControlPanel Detection and Linking
tabs to provide expert-level control over the analysis pipeline.
"""
from qtpy import QtCore, QtWidgets


def _make_double_spin(value, lo, hi, decimals=3, suffix='', step=None):
    sb = QtWidgets.QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    if step is not None:
        sb.setSingleStep(step)
    return sb


def _make_int_spin(value, lo, hi, suffix=''):
    sb = QtWidgets.QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    return sb


# ---------------------------------------------------------------------------
# ThunderSTORM controls
# ---------------------------------------------------------------------------

class ThunderSTORMControlGroup(QtWidgets.QGroupBox):
    """Full ThunderSTORM detection parameter controls.

    Exposes all filter, detector, fitter, camera, and post-processing
    parameters matching the original ThunderSTORM ImageJ plugin
    (Ovesny et al., 2014).
    """

    def __init__(self, parent=None):
        super().__init__("ThunderSTORM Parameters", parent)
        self.setCheckable(False)
        self._build()
        self._connect()

    # ---- helpers for show/hide with labels ----
    @staticmethod
    def _set_row_visible(widget, visible):
        """Show/hide a widget and its QFormLayout label if present."""
        widget.setVisible(visible)
        parent = widget.parent()
        if parent is not None:
            lay = parent.layout()
            if lay is not None and hasattr(lay, 'labelForField'):
                lbl = lay.labelForField(widget)
                if lbl is not None:
                    lbl.setVisible(visible)

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # ==================================================================
        # Image Filter
        # ==================================================================
        filter_group = QtWidgets.QGroupBox("Image Filter")
        fl = QtWidgets.QFormLayout(filter_group)
        fl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems([
            'wavelet', 'gaussian', 'dog', 'lowered_gaussian',
            'difference_of_averaging', 'median', 'box', 'none',
            'wavelet+gaussian', 'wavelet+dog', 'wavelet+median'])
        fl.addRow("Filter type:", self.filter_type)

        # -- Wavelet params --
        self.wavelet_scale = _make_int_spin(2, 1, 5)
        self.wavelet_scale.setToolTip("B-spline wavelet scale (1-5)")
        fl.addRow("Wavelet scale:", self.wavelet_scale)

        self.wavelet_order = _make_int_spin(3, 1, 5)
        self.wavelet_order.setToolTip(
            "B-spline order (3 = cubic, default)")
        fl.addRow("Wavelet B-spline order:", self.wavelet_order)

        # -- Gaussian params --
        self.gauss_sigma = _make_double_spin(1.6, 0.1, 10.0, decimals=2,
                                              step=0.1, suffix=' px')
        self.gauss_sigma.setToolTip("Gaussian kernel sigma")
        fl.addRow("Gaussian sigma:", self.gauss_sigma)

        # -- DoG params --
        self.dog_sigma1 = _make_double_spin(1.0, 0.1, 10.0, decimals=2,
                                              step=0.1, suffix=' px')
        self.dog_sigma1.setToolTip("Narrow Gaussian sigma for DoG")
        fl.addRow("DoG sigma 1:", self.dog_sigma1)

        self.dog_sigma2 = _make_double_spin(1.6, 0.1, 20.0, decimals=2,
                                              step=0.1, suffix=' px')
        self.dog_sigma2.setToolTip("Wide Gaussian sigma for DoG")
        fl.addRow("DoG sigma 2:", self.dog_sigma2)

        # -- Lowered Gaussian params --
        self.lowered_sigma = _make_double_spin(1.6, 0.1, 10.0, decimals=2,
                                                step=0.1, suffix=' px')
        self.lowered_sigma.setToolTip(
            "Gaussian sigma for lowered Gaussian filter")
        fl.addRow("Lowered Gauss sigma:", self.lowered_sigma)

        self.lowered_size = _make_int_spin(3, 1, 21, suffix=' px')
        self.lowered_size.setToolTip(
            "Averaging kernel size for lowered Gaussian")
        fl.addRow("Lowered avg size:", self.lowered_size)

        # -- Difference of averaging params --
        self.diff_avg_size1 = _make_int_spin(3, 1, 21, suffix=' px')
        self.diff_avg_size1.setToolTip("First box filter size")
        fl.addRow("DiffAvg size 1:", self.diff_avg_size1)

        self.diff_avg_size2 = _make_int_spin(5, 1, 41, suffix=' px')
        self.diff_avg_size2.setToolTip("Second box filter size")
        fl.addRow("DiffAvg size 2:", self.diff_avg_size2)

        # -- Median / Box size --
        self.median_size = _make_int_spin(3, 1, 21, suffix=' px')
        self.median_size.setToolTip("Median filter kernel size")
        fl.addRow("Median size:", self.median_size)

        self.box_size = _make_int_spin(3, 1, 21, suffix=' px')
        self.box_size.setToolTip("Box (averaging) filter kernel size")
        fl.addRow("Box size:", self.box_size)

        # -- Compound secondary filter params --
        self.compound_secondary_sigma = _make_double_spin(
            1.0, 0.1, 10.0, decimals=2, step=0.1, suffix=' px')
        self.compound_secondary_sigma.setToolTip(
            "Sigma for compound secondary Gaussian filter")
        fl.addRow("Secondary sigma:", self.compound_secondary_sigma)

        self.compound_secondary_size = _make_int_spin(3, 1, 21, suffix=' px')
        self.compound_secondary_size.setToolTip(
            "Kernel size for compound secondary median filter")
        fl.addRow("Secondary size:", self.compound_secondary_size)

        self.compound_secondary_sigma1 = _make_double_spin(
            1.0, 0.1, 10.0, decimals=2, step=0.1, suffix=' px')
        self.compound_secondary_sigma1.setToolTip(
            "Narrow sigma for compound secondary DoG")
        fl.addRow("Secondary DoG σ1:", self.compound_secondary_sigma1)

        self.compound_secondary_sigma2 = _make_double_spin(
            2.0, 0.1, 20.0, decimals=2, step=0.1, suffix=' px')
        self.compound_secondary_sigma2.setToolTip(
            "Wide sigma for compound secondary DoG")
        fl.addRow("Secondary DoG σ2:", self.compound_secondary_sigma2)

        layout.addWidget(filter_group)

        # ==================================================================
        # Candidate Detector
        # ==================================================================
        det_group = QtWidgets.QGroupBox("Candidate Detector")
        dl = QtWidgets.QFormLayout(det_group)
        dl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.detector_type = QtWidgets.QComboBox()
        self.detector_type.addItems(['local_max', 'nms', 'centroid', 'grid'])
        dl.addRow("Detector type:", self.detector_type)

        self.threshold = QtWidgets.QComboBox()
        self.threshold.setEditable(True)
        self.threshold.addItems([
            'std(Wave.F1)', '2*std(Wave.F1)', '3*std(Wave.F1)',
            'mean(Wave.F1)+std(Wave.F1)',
            'median(Wave.F1)+std(Wave.F1)'])
        dl.addRow("Threshold:", self.threshold)

        # -- LocalMaximum params --
        self.lm_connectivity = QtWidgets.QComboBox()
        self.lm_connectivity.addItems(['8-neighbourhood', '4-neighbourhood'])
        self.lm_connectivity.setToolTip(
            "Pixel connectivity for local maximum detection")
        dl.addRow("Connectivity:", self.lm_connectivity)

        self.lm_min_distance = _make_int_spin(1, 1, 20, suffix=' px')
        self.lm_min_distance.setToolTip(
            "Minimum distance between detected peaks")
        dl.addRow("Min distance:", self.lm_min_distance)

        # -- NMS params --
        self.nms_connectivity = QtWidgets.QComboBox()
        self.nms_connectivity.addItems(['2 (8-connected)', '1 (4-connected)'])
        self.nms_connectivity.setToolTip(
            "Connectivity for non-maximum suppression")
        dl.addRow("NMS connectivity:", self.nms_connectivity)

        # -- Centroid params --
        self.centroid_connectivity = QtWidgets.QComboBox()
        self.centroid_connectivity.addItems(
            ['2 (8-connected)', '1 (4-connected)'])
        self.centroid_connectivity.setToolTip(
            "Connected components connectivity")
        dl.addRow("CC connectivity:", self.centroid_connectivity)

        self.centroid_min_area = _make_int_spin(1, 1, 100, suffix=' px²')
        self.centroid_min_area.setToolTip(
            "Minimum connected component area")
        dl.addRow("Min area:", self.centroid_min_area)

        self.watershed_cb = QtWidgets.QCheckBox()
        self.watershed_cb.setChecked(True)
        self.watershed_cb.setToolTip(
            "Apply watershed segmentation to split touching peaks")
        dl.addRow("Watershed:", self.watershed_cb)

        # -- Grid params --
        self.grid_spacing = _make_int_spin(10, 2, 100, suffix=' px')
        self.grid_spacing.setToolTip("Grid spacing for grid detector")
        dl.addRow("Grid spacing:", self.grid_spacing)

        # -- Common detector params --
        self.exclude_border = _make_int_spin(3, 0, 50, suffix=' px')
        self.exclude_border.setToolTip(
            "Border exclusion width (0 = no exclusion)")
        dl.addRow("Exclude border:", self.exclude_border)

        layout.addWidget(det_group)

        # ==================================================================
        # Sub-pixel Fitting
        # ==================================================================
        fit_group = QtWidgets.QGroupBox("Sub-pixel Fitting")
        ftl = QtWidgets.QFormLayout(fit_group)
        ftl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.fitter_type = QtWidgets.QComboBox()
        self.fitter_type.addItems([
            'gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle',
            'elliptical_gaussian_mle', 'phasor', 'radial_symmetry',
            'centroid', 'multi_emitter'])
        ftl.addRow("Fitter type:", self.fitter_type)

        self.fit_radius = _make_int_spin(3, 2, 10, suffix=' px')
        self.fit_radius.setToolTip(
            "Fitting ROI half-width (ROI size = 2×radius + 1)")
        ftl.addRow("Fit radius:", self.fit_radius)

        self.initial_sigma = _make_double_spin(1.3, 0.5, 5.0, decimals=2,
                                                step=0.1, suffix=' px')
        self.initial_sigma.setToolTip("Initial PSF sigma estimate for fitting")
        ftl.addRow("Initial sigma:", self.initial_sigma)

        self.max_iterations = _make_int_spin(500, 10, 5000)
        self.max_iterations.setToolTip(
            "Maximum Levenberg-Marquardt iterations (default 500, MLE 1000)")
        ftl.addRow("Max iterations:", self.max_iterations)

        # Multi-emitter options
        self.max_emitters = _make_int_spin(5, 2, 10)
        self.max_emitters.setToolTip(
            "Maximum number of emitters to fit per ROI")
        ftl.addRow("Max emitters:", self.max_emitters)

        self.p_value = QtWidgets.QComboBox()
        self.p_value.addItems(['1e-6', '1e-4', '1e-3', '0.01', '0.05'])
        self.p_value.setToolTip(
            "F-test p-value threshold for adding another emitter")
        ftl.addRow("p-value threshold:", self.p_value)

        layout.addWidget(fit_group)

        # ==================================================================
        # Camera Model
        # ==================================================================
        cam_group = QtWidgets.QGroupBox("Camera Model")
        cl = QtWidgets.QFormLayout(cam_group)
        cl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.pixel_size = _make_double_spin(108.0, 1.0, 10000.0,
                                             decimals=1, step=10.0,
                                             suffix=' nm')
        self.pixel_size.setToolTip("Camera pixel size in nanometers")
        cl.addRow("Pixel size:", self.pixel_size)

        self.photons_per_adu = _make_double_spin(3.6, 0.01, 100.0,
                                                  decimals=2, step=0.1)
        cl.addRow("Photons/ADU:", self.photons_per_adu)

        self.baseline = _make_double_spin(100.0, 0.0, 10000.0,
                                           decimals=1, step=10.0)
        cl.addRow("Camera baseline:", self.baseline)

        self.em_gain_cb = QtWidgets.QCheckBox()
        cl.addRow("EM gain enabled:", self.em_gain_cb)

        self.em_gain_value = _make_double_spin(100.0, 1.0, 1000.0,
                                                decimals=1, step=10.0)
        self.em_gain_value.setEnabled(False)
        cl.addRow("EM gain value:", self.em_gain_value)

        self.quantum_efficiency = _make_double_spin(1.0, 0.01, 1.0,
                                                     decimals=2, step=0.01)
        cl.addRow("Quantum efficiency:", self.quantum_efficiency)

        layout.addWidget(cam_group)

        # ==================================================================
        # Post-processing (collapsible)
        # ==================================================================
        sec_post = _CollapsibleSection("Post-processing")

        # Drift correction
        self.drift_correction_cb = QtWidgets.QCheckBox()
        self.drift_correction_cb.setToolTip(
            "Apply cross-correlation drift correction after detection")
        sec_post.addRow("Drift correction:", self.drift_correction_cb)

        self.drift_segment_frames = _make_int_spin(500, 10, 10000)
        self.drift_segment_frames.setToolTip(
            "Frames per segment for drift estimation")
        self.drift_segment_frames.setEnabled(False)
        sec_post.addRow("Segment frames:", self.drift_segment_frames)

        self.drift_smoothing = _make_double_spin(0.25, 0.01, 5.0,
                                                   decimals=2, step=0.05)
        self.drift_smoothing.setToolTip(
            "Cubic spline smoothing factor for drift trajectory")
        self.drift_smoothing.setEnabled(False)
        sec_post.addRow("Drift smoothing:", self.drift_smoothing)

        # Molecular merging
        self.merge_molecules_cb = QtWidgets.QCheckBox()
        self.merge_molecules_cb.setToolTip(
            "Merge re-appearing molecules across consecutive frames")
        sec_post.addRow("Merge molecules:", self.merge_molecules_cb)

        self.merge_max_distance = _make_double_spin(50.0, 1.0, 500.0,
                                                      decimals=1, step=5.0,
                                                      suffix=' nm')
        self.merge_max_distance.setToolTip(
            "Maximum distance for merging re-appearances")
        self.merge_max_distance.setEnabled(False)
        sec_post.addRow("Merge max dist:", self.merge_max_distance)

        self.merge_max_gap = _make_int_spin(1, 0, 10, suffix=' frames')
        self.merge_max_gap.setToolTip(
            "Maximum frame gap for molecule re-appearance")
        self.merge_max_gap.setEnabled(False)
        sec_post.addRow("Merge max gap:", self.merge_max_gap)

        # Localization filtering
        self.filter_loc_cb = QtWidgets.QCheckBox()
        self.filter_loc_cb.setToolTip(
            "Filter localizations by intensity, sigma, uncertainty")
        sec_post.addRow("Filter localizations:", self.filter_loc_cb)

        self.filter_min_intensity = _make_double_spin(
            0.0, 0.0, 1e8, decimals=1, step=100.0)
        self.filter_min_intensity.setToolTip("Minimum intensity (0 = off)")
        self.filter_min_intensity.setEnabled(False)
        sec_post.addRow("Min intensity:", self.filter_min_intensity)

        self.filter_max_uncertainty = _make_double_spin(
            0.0, 0.0, 1000.0, decimals=1, step=5.0, suffix=' nm')
        self.filter_max_uncertainty.setToolTip(
            "Maximum localization uncertainty (0 = off)")
        self.filter_max_uncertainty.setEnabled(False)
        sec_post.addRow("Max uncertainty:", self.filter_max_uncertainty)

        self.filter_min_sigma = _make_double_spin(
            0.0, 0.0, 10.0, decimals=2, step=0.1, suffix=' px')
        self.filter_min_sigma.setToolTip("Minimum PSF sigma (0 = off)")
        self.filter_min_sigma.setEnabled(False)
        sec_post.addRow("Min sigma:", self.filter_min_sigma)

        self.filter_max_sigma = _make_double_spin(
            0.0, 0.0, 50.0, decimals=2, step=0.5, suffix=' px')
        self.filter_max_sigma.setToolTip("Maximum PSF sigma (0 = off)")
        self.filter_max_sigma.setEnabled(False)
        sec_post.addRow("Max sigma:", self.filter_max_sigma)

        # Density filtering
        self.density_filter_cb = QtWidgets.QCheckBox()
        self.density_filter_cb.setToolTip(
            "Remove isolated localizations by local density")
        sec_post.addRow("Density filter:", self.density_filter_cb)

        self.density_radius = _make_double_spin(
            50.0, 1.0, 1000.0, decimals=1, step=10.0, suffix=' nm')
        self.density_radius.setToolTip("Search radius for neighbor counting")
        self.density_radius.setEnabled(False)
        sec_post.addRow("Density radius:", self.density_radius)

        self.density_min_neighbors = _make_int_spin(3, 1, 50)
        self.density_min_neighbors.setToolTip(
            "Minimum neighbors required to keep a localization")
        self.density_min_neighbors.setEnabled(False)
        sec_post.addRow("Min neighbors:", self.density_min_neighbors)

        # Duplicate removal
        self.remove_duplicates_cb = QtWidgets.QCheckBox()
        self.remove_duplicates_cb.setToolTip(
            "Remove duplicate detections within same frame")
        sec_post.addRow("Remove duplicates:", self.remove_duplicates_cb)

        self.duplicate_max_distance = _make_double_spin(
            20.0, 1.0, 200.0, decimals=1, step=5.0, suffix=' nm')
        self.duplicate_max_distance.setToolTip(
            "Distance threshold for duplicate removal")
        self.duplicate_max_distance.setEnabled(False)
        sec_post.addRow("Dup max dist:", self.duplicate_max_distance)

        layout.addWidget(sec_post)

        # ==================================================================
        # Rendering (collapsible)
        # ==================================================================
        sec_render = _CollapsibleSection("Super-resolution Rendering")

        self.render_method = QtWidgets.QComboBox()
        self.render_method.addItems([
            'gaussian', 'normalized_gaussian', 'histogram',
            'ash', 'scatter'])
        self.render_method.setToolTip("Super-resolution image rendering mode")
        sec_render.addRow("Method:", self.render_method)

        self.render_pixel_size = _make_double_spin(
            10.0, 0.1, 1000.0, decimals=1, step=1.0, suffix=' nm')
        self.render_pixel_size.setToolTip(
            "Output pixel size for super-resolution image")
        sec_render.addRow("Pixel size:", self.render_pixel_size)

        self.render_magnification = _make_int_spin(10, 1, 100)
        self.render_magnification.setToolTip(
            "Sub-pixel shifts per axis for ASH rendering")
        sec_render.addRow("ASH magnification:", self.render_magnification)

        layout.addWidget(sec_render)

        # Apply initial visibility
        self._on_filter_changed(self.filter_type.currentText())
        self._on_detector_changed(self.detector_type.currentText())
        self._on_fitter_changed(self.fitter_type.currentText())

    def _connect(self):
        self.filter_type.currentTextChanged.connect(self._on_filter_changed)
        self.detector_type.currentTextChanged.connect(self._on_detector_changed)
        self.fitter_type.currentTextChanged.connect(self._on_fitter_changed)
        self.em_gain_cb.toggled.connect(self.em_gain_value.setEnabled)
        # Post-processing enable/disable
        self.drift_correction_cb.toggled.connect(
            self.drift_segment_frames.setEnabled)
        self.drift_correction_cb.toggled.connect(
            self.drift_smoothing.setEnabled)
        self.merge_molecules_cb.toggled.connect(
            self.merge_max_distance.setEnabled)
        self.merge_molecules_cb.toggled.connect(
            self.merge_max_gap.setEnabled)
        self.filter_loc_cb.toggled.connect(
            self.filter_min_intensity.setEnabled)
        self.filter_loc_cb.toggled.connect(
            self.filter_max_uncertainty.setEnabled)
        self.filter_loc_cb.toggled.connect(
            self.filter_min_sigma.setEnabled)
        self.filter_loc_cb.toggled.connect(
            self.filter_max_sigma.setEnabled)
        self.density_filter_cb.toggled.connect(
            self.density_radius.setEnabled)
        self.density_filter_cb.toggled.connect(
            self.density_min_neighbors.setEnabled)
        self.remove_duplicates_cb.toggled.connect(
            self.duplicate_max_distance.setEnabled)

    def _on_filter_changed(self, text):
        ft = text.lower()
        has_wavelet = 'wavelet' in ft
        is_gaussian = ft == 'gaussian'
        is_dog = ft == 'dog'
        is_lowered = ft == 'lowered_gaussian'
        is_diff_avg = ft == 'difference_of_averaging'
        is_median = ft == 'median'
        is_box = ft == 'box'
        is_compound = '+' in ft

        # Wavelet params
        self._set_row_visible(self.wavelet_scale, has_wavelet)
        self._set_row_visible(self.wavelet_order, has_wavelet)

        # Gaussian params
        self._set_row_visible(self.gauss_sigma,
                              is_gaussian and not is_compound)

        # DoG params
        self._set_row_visible(self.dog_sigma1, is_dog and not is_compound)
        self._set_row_visible(self.dog_sigma2, is_dog and not is_compound)

        # Lowered Gaussian params
        self._set_row_visible(self.lowered_sigma, is_lowered)
        self._set_row_visible(self.lowered_size, is_lowered)

        # Diff averaging params
        self._set_row_visible(self.diff_avg_size1, is_diff_avg)
        self._set_row_visible(self.diff_avg_size2, is_diff_avg)

        # Median / box
        self._set_row_visible(self.median_size, is_median and not is_compound)
        self._set_row_visible(self.box_size, is_box)

        # Compound secondary params
        is_compound_gauss = ft == 'wavelet+gaussian'
        is_compound_dog = ft == 'wavelet+dog'
        is_compound_median = ft == 'wavelet+median'
        self._set_row_visible(self.compound_secondary_sigma,
                              is_compound_gauss)
        self._set_row_visible(self.compound_secondary_size,
                              is_compound_median)
        self._set_row_visible(self.compound_secondary_sigma1,
                              is_compound_dog)
        self._set_row_visible(self.compound_secondary_sigma2,
                              is_compound_dog)

    def _on_detector_changed(self, text):
        dt = text.lower()
        is_local_max = dt == 'local_max'
        is_nms = dt == 'nms'
        is_centroid = dt == 'centroid'
        is_grid = dt == 'grid'

        self._set_row_visible(self.lm_connectivity, is_local_max)
        self._set_row_visible(self.lm_min_distance, is_local_max)
        self._set_row_visible(self.nms_connectivity, is_nms)
        self._set_row_visible(self.centroid_connectivity, is_centroid)
        self._set_row_visible(self.centroid_min_area, is_centroid)
        self._set_row_visible(self.watershed_cb, is_centroid)
        self._set_row_visible(self.grid_spacing, is_grid)

    def _on_fitter_changed(self, text):
        is_multi = (text == 'multi_emitter')
        self._set_row_visible(self.max_emitters, is_multi)
        self._set_row_visible(self.p_value, is_multi)

    def get_params(self):
        """Return a dict of kwargs for ``ThunderSTORMDetector.__init__``."""
        filter_type = self.filter_type.currentText()
        filter_params = {}
        ft = filter_type.lower()

        if 'wavelet' in ft:
            filter_params['scale'] = self.wavelet_scale.value()
            filter_params['order'] = self.wavelet_order.value()
        if ft == 'gaussian':
            filter_params['sigma'] = self.gauss_sigma.value()
        elif ft == 'dog':
            filter_params['sigma1'] = self.dog_sigma1.value()
            filter_params['sigma2'] = self.dog_sigma2.value()
        elif ft == 'lowered_gaussian':
            filter_params['sigma'] = self.lowered_sigma.value()
            filter_params['size'] = self.lowered_size.value()
        elif ft == 'difference_of_averaging':
            filter_params['size1'] = self.diff_avg_size1.value()
            filter_params['size2'] = self.diff_avg_size2.value()
        elif ft == 'median':
            filter_params['size'] = self.median_size.value()
        elif ft == 'box':
            filter_params['size'] = self.box_size.value()
        # Compound filter secondary params
        elif ft == 'wavelet+gaussian':
            filter_params['secondary_sigma'] = \
                self.compound_secondary_sigma.value()
        elif ft == 'wavelet+dog':
            filter_params['secondary_sigma1'] = \
                self.compound_secondary_sigma1.value()
            filter_params['secondary_sigma2'] = \
                self.compound_secondary_sigma2.value()
        elif ft == 'wavelet+median':
            filter_params['secondary_size'] = \
                self.compound_secondary_size.value()

        # Detector params
        detector_type = self.detector_type.currentText()
        detector_params = {}
        dt = detector_type.lower()
        border_val = self.exclude_border.value()
        if border_val > 0:
            detector_params['exclude_border'] = border_val
        else:
            detector_params['exclude_border'] = False

        if dt == 'local_max':
            detector_params['connectivity'] = \
                self.lm_connectivity.currentText()
            detector_params['min_distance'] = self.lm_min_distance.value()
        elif dt == 'nms':
            conn_text = self.nms_connectivity.currentText()
            detector_params['connectivity'] = int(conn_text[0])
        elif dt == 'centroid':
            conn_text = self.centroid_connectivity.currentText()
            detector_params['connectivity'] = int(conn_text[0])
            detector_params['min_area'] = self.centroid_min_area.value()
            detector_params['use_watershed'] = self.watershed_cb.isChecked()
        elif dt == 'grid':
            detector_params['spacing'] = self.grid_spacing.value()

        # Fitter params
        fitter_type = self.fitter_type.currentText()
        fitter_params = {
            'initial_sigma': self.initial_sigma.value(),
            'max_iterations': self.max_iterations.value(),
        }
        if fitter_type == 'multi_emitter':
            fitter_params['max_emitters'] = self.max_emitters.value()
            fitter_params['p_value_threshold'] = float(
                self.p_value.currentText())

        # Camera params
        camera_params = {
            'pixel_size': self.pixel_size.value(),
            'photons_per_adu': self.photons_per_adu.value(),
            'baseline': self.baseline.value(),
            'is_emccd': self.em_gain_cb.isChecked(),
            'em_gain': (self.em_gain_value.value()
                        if self.em_gain_cb.isChecked() else 1.0),
            'quantum_efficiency': self.quantum_efficiency.value(),
        }

        threshold_text = self.threshold.currentText().strip()
        try:
            threshold = float(threshold_text)
        except ValueError:
            threshold = threshold_text if threshold_text else None

        roi_size = self.fit_radius.value() * 2 + 1

        result = {
            'filter_type': filter_type,
            'detector_type': detector_type,
            'fitter_type': fitter_type,
            'threshold': threshold,
            'roi_size': roi_size,
            'camera_params': camera_params,
            'filter_params': filter_params,
            'detector_params': detector_params,
            'fitter_params': fitter_params,
        }

        # Post-processing params
        if self.drift_correction_cb.isChecked():
            result['drift_correction'] = {
                'segment_frames': self.drift_segment_frames.value(),
                'smoothing': self.drift_smoothing.value(),
            }
        if self.merge_molecules_cb.isChecked():
            result['merge_molecules'] = {
                'max_distance': self.merge_max_distance.value(),
                'max_frame_gap': self.merge_max_gap.value(),
            }
        if self.filter_loc_cb.isChecked():
            filt = {}
            v = self.filter_min_intensity.value()
            if v > 0:
                filt['min_intensity'] = v
            v = self.filter_max_uncertainty.value()
            if v > 0:
                filt['max_uncertainty'] = v
            v = self.filter_min_sigma.value()
            if v > 0:
                filt['min_sigma'] = v
            v = self.filter_max_sigma.value()
            if v > 0:
                filt['max_sigma'] = v
            if filt:
                result['filter_localizations'] = filt
        if self.density_filter_cb.isChecked():
            result['density_filter'] = {
                'radius': self.density_radius.value(),
                'min_neighbors': self.density_min_neighbors.value(),
            }
        if self.remove_duplicates_cb.isChecked():
            result['remove_duplicates'] = {
                'max_distance': self.duplicate_max_distance.value(),
            }

        # Rendering params
        result['rendering'] = {
            'method': self.render_method.currentText(),
            'pixel_size': self.render_pixel_size.value(),
            'magnification': self.render_magnification.value(),
        }

        return result

    def set_params(self, params):
        """Set controls from a dict (e.g. from a saved configuration)."""
        if 'filter_type' in params:
            idx = self.filter_type.findText(params['filter_type'])
            if idx >= 0:
                self.filter_type.setCurrentIndex(idx)
        if 'detector_type' in params:
            idx = self.detector_type.findText(params['detector_type'])
            if idx >= 0:
                self.detector_type.setCurrentIndex(idx)
        if 'fitter_type' in params:
            idx = self.fitter_type.findText(params['fitter_type'])
            if idx >= 0:
                self.fitter_type.setCurrentIndex(idx)
        if 'threshold' in params and params['threshold'] is not None:
            self.threshold.setCurrentText(str(params['threshold']))

        # Filter params
        fp = params.get('filter_params', {})
        if 'scale' in fp:
            self.wavelet_scale.setValue(fp['scale'])
        if 'order' in fp:
            self.wavelet_order.setValue(fp['order'])
        if 'sigma' in fp:
            ft = params.get('filter_type', '').lower()
            if ft == 'gaussian':
                self.gauss_sigma.setValue(fp['sigma'])
            elif ft == 'lowered_gaussian':
                self.lowered_sigma.setValue(fp['sigma'])
        if 'sigma1' in fp:
            self.dog_sigma1.setValue(fp['sigma1'])
        if 'sigma2' in fp:
            self.dog_sigma2.setValue(fp['sigma2'])
        if 'size' in fp:
            ft = params.get('filter_type', '').lower()
            if ft == 'median':
                self.median_size.setValue(fp['size'])
            elif ft == 'box':
                self.box_size.setValue(fp['size'])
            elif ft == 'lowered_gaussian':
                self.lowered_size.setValue(fp['size'])
        if 'size1' in fp:
            self.diff_avg_size1.setValue(fp['size1'])
        if 'size2' in fp:
            self.diff_avg_size2.setValue(fp['size2'])
        if 'secondary_sigma' in fp:
            self.compound_secondary_sigma.setValue(fp['secondary_sigma'])
        if 'secondary_size' in fp:
            self.compound_secondary_size.setValue(fp['secondary_size'])
        if 'secondary_sigma1' in fp:
            self.compound_secondary_sigma1.setValue(fp['secondary_sigma1'])
        if 'secondary_sigma2' in fp:
            self.compound_secondary_sigma2.setValue(fp['secondary_sigma2'])

        # Detector params
        dp = params.get('detector_params', {})
        if 'connectivity' in dp:
            dt = params.get('detector_type', '').lower()
            if dt == 'local_max':
                idx = self.lm_connectivity.findText(str(dp['connectivity']))
                if idx >= 0:
                    self.lm_connectivity.setCurrentIndex(idx)
            elif dt == 'nms':
                v = int(dp['connectivity'])
                text = f'{v} ({"8" if v == 2 else "4"}-connected)'
                idx = self.nms_connectivity.findText(text)
                if idx >= 0:
                    self.nms_connectivity.setCurrentIndex(idx)
            elif dt == 'centroid':
                v = int(dp['connectivity'])
                text = f'{v} ({"8" if v == 2 else "4"}-connected)'
                idx = self.centroid_connectivity.findText(text)
                if idx >= 0:
                    self.centroid_connectivity.setCurrentIndex(idx)
        if 'min_distance' in dp:
            self.lm_min_distance.setValue(dp['min_distance'])
        if 'min_area' in dp:
            self.centroid_min_area.setValue(dp['min_area'])
        if 'use_watershed' in dp:
            self.watershed_cb.setChecked(dp['use_watershed'])
        if 'spacing' in dp:
            self.grid_spacing.setValue(dp['spacing'])
        if 'exclude_border' in dp:
            eb = dp['exclude_border']
            if isinstance(eb, bool):
                self.exclude_border.setValue(3 if eb else 0)
            elif isinstance(eb, int):
                self.exclude_border.setValue(eb)

        # Fitter params
        ftp = params.get('fitter_params', {})
        if 'initial_sigma' in ftp:
            self.initial_sigma.setValue(ftp['initial_sigma'])
        if 'max_iterations' in ftp:
            self.max_iterations.setValue(ftp['max_iterations'])
        if 'max_emitters' in ftp:
            self.max_emitters.setValue(ftp['max_emitters'])
        if 'p_value_threshold' in ftp:
            idx = self.p_value.findText(str(ftp['p_value_threshold']))
            if idx >= 0:
                self.p_value.setCurrentIndex(idx)

        # Camera params
        cp = params.get('camera_params', {})
        if 'pixel_size' in cp:
            self.pixel_size.setValue(cp['pixel_size'])
        if 'photons_per_adu' in cp:
            self.photons_per_adu.setValue(cp['photons_per_adu'])
        if 'baseline' in cp:
            self.baseline.setValue(cp['baseline'])
        if 'is_emccd' in cp:
            self.em_gain_cb.setChecked(cp['is_emccd'])
        if 'em_gain' in cp:
            self.em_gain_value.setValue(cp['em_gain'])
        if 'quantum_efficiency' in cp:
            self.quantum_efficiency.setValue(cp['quantum_efficiency'])

        if 'roi_size' in params:
            self.fit_radius.setValue(params['roi_size'] // 2)

        # Post-processing
        if 'drift_correction' in params:
            self.drift_correction_cb.setChecked(True)
            dc = params['drift_correction']
            if 'segment_frames' in dc:
                self.drift_segment_frames.setValue(dc['segment_frames'])
            if 'smoothing' in dc:
                self.drift_smoothing.setValue(dc['smoothing'])
        if 'merge_molecules' in params:
            self.merge_molecules_cb.setChecked(True)
            mm = params['merge_molecules']
            if 'max_distance' in mm:
                self.merge_max_distance.setValue(mm['max_distance'])
            if 'max_frame_gap' in mm:
                self.merge_max_gap.setValue(mm['max_frame_gap'])
        if 'filter_localizations' in params:
            self.filter_loc_cb.setChecked(True)
            fl = params['filter_localizations']
            if 'min_intensity' in fl:
                self.filter_min_intensity.setValue(fl['min_intensity'])
            if 'max_uncertainty' in fl:
                self.filter_max_uncertainty.setValue(fl['max_uncertainty'])
            if 'min_sigma' in fl:
                self.filter_min_sigma.setValue(fl['min_sigma'])
            if 'max_sigma' in fl:
                self.filter_max_sigma.setValue(fl['max_sigma'])
        if 'density_filter' in params:
            self.density_filter_cb.setChecked(True)
            df = params['density_filter']
            if 'radius' in df:
                self.density_radius.setValue(df['radius'])
            if 'min_neighbors' in df:
                self.density_min_neighbors.setValue(df['min_neighbors'])
        if 'remove_duplicates' in params:
            self.remove_duplicates_cb.setChecked(True)
            rd = params['remove_duplicates']
            if 'max_distance' in rd:
                self.duplicate_max_distance.setValue(rd['max_distance'])

        # Rendering
        if 'rendering' in params:
            rp = params['rendering']
            if 'method' in rp:
                idx = self.render_method.findText(rp['method'])
                if idx >= 0:
                    self.render_method.setCurrentIndex(idx)
            if 'pixel_size' in rp:
                self.render_pixel_size.setValue(rp['pixel_size'])
            if 'magnification' in rp:
                self.render_magnification.setValue(rp['magnification'])


# ---------------------------------------------------------------------------
# Collapsible group helper
# ---------------------------------------------------------------------------

class _CollapsibleSection(QtWidgets.QWidget):
    """A section with a toggle button that shows/hides its contents."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.toggled.connect(self._on_toggle)

        self._content = QtWidgets.QWidget()
        self._content.setVisible(False)
        self._content_layout = QtWidgets.QFormLayout(self._content)
        self._content_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._content_layout.setSpacing(4)
        self._content_layout.setContentsMargins(12, 4, 4, 4)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._toggle)
        lay.addWidget(self._content)

    def _on_toggle(self, checked):
        self._toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked
            else QtCore.Qt.ArrowType.RightArrow)
        self._content.setVisible(checked)

    def addRow(self, label, widget):
        self._content_layout.addRow(label, widget)


# ---------------------------------------------------------------------------
# U-Track Detection controls
# ---------------------------------------------------------------------------

class UTrackDetectionControlGroup(QtWidgets.QGroupBox):
    """Advanced U-Track detection parameters.

    Exposes DoG bandpass, mixture fitting, local background, and
    auto-estimation controls beyond the basic PSF sigma / alpha / min
    intensity that appear in the main Detection tab.
    """

    def __init__(self, parent=None):
        super().__init__("U-Track Detection — Advanced", parent)
        self.setCheckable(False)
        self._build()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Bandpass filtering ---
        sec_bp = _CollapsibleSection("Bandpass Filtering")

        self.dog_ratio = _make_double_spin(5.0, 0.0, 20.0, decimals=1,
                                            step=0.5)
        self.dog_ratio.setToolTip(
            "Ratio of large-to-small sigma for DoG bandpass.\n"
            "Set to 0 to disable DoG (single Gaussian, legacy mode).")
        sec_bp.addRow("DoG ratio:", self.dog_ratio)

        layout.addWidget(sec_bp)

        # --- Sub-pixel fitting ---
        sec_fit = _CollapsibleSection("Sub-pixel Fitting")

        self.mixture_separation = _make_double_spin(3.0, 0.0, 10.0,
                                                      decimals=1, step=0.5)
        self.mixture_separation.setToolTip(
            "Max separation (in PSF sigma units) for attempting\n"
            "2-Gaussian mixture fit on overlapping particles.\n"
            "Set to 0 to disable mixture fitting.")
        sec_fit.addRow("Mixture separation:", self.mixture_separation)

        layout.addWidget(sec_fit)

        # --- Background estimation ---
        sec_bg = _CollapsibleSection("Background Estimation")

        self.local_bg_inner = _make_double_spin(3.0, 0.0, 10.0,
                                                  decimals=1, step=0.5,
                                                  suffix=' σ')
        self.local_bg_inner.setToolTip(
            "Inner radius of local background annulus (in PSF sigma units).\n"
            "Set to 0 for global background estimation (legacy).")
        sec_bg.addRow("BG annulus inner:", self.local_bg_inner)

        self.local_bg_outer = _make_double_spin(5.0, 1.0, 20.0,
                                                  decimals=1, step=0.5,
                                                  suffix=' σ')
        self.local_bg_outer.setToolTip(
            "Outer radius of local background annulus (in PSF sigma units).")
        sec_bg.addRow("BG annulus outer:", self.local_bg_outer)

        layout.addWidget(sec_bg)

        # --- Auto-estimation ---
        sec_auto = _CollapsibleSection("Auto-estimation")

        self.auto_psf_btn = QtWidgets.QPushButton("Estimate PSF σ from image")
        self.auto_psf_btn.setToolTip(
            "Estimate PSF sigma from the autocorrelation of the current image.")
        sec_auto.addRow("", self.auto_psf_btn)

        self.auto_noise_btn = QtWidgets.QPushButton("Estimate noise from image")
        self.auto_noise_btn.setToolTip(
            "Estimate noise standard deviation from the image Laplacian.")
        sec_auto.addRow("", self.auto_noise_btn)

        layout.addWidget(sec_auto)

    def get_params(self):
        """Return advanced detection kwargs for ``UTrackDetector``."""
        return {
            'dog_ratio': self.dog_ratio.value(),
            'mixture_separation': self.mixture_separation.value(),
            'local_bg_inner': self.local_bg_inner.value(),
            'local_bg_outer': self.local_bg_outer.value(),
        }

    def set_params(self, params):
        """Set controls from a dict."""
        if 'dog_ratio' in params:
            self.dog_ratio.setValue(params['dog_ratio'])
        if 'mixture_separation' in params:
            self.mixture_separation.setValue(params['mixture_separation'])
        if 'local_bg_inner' in params:
            self.local_bg_inner.setValue(params['local_bg_inner'])
        if 'local_bg_outer' in params:
            self.local_bg_outer.setValue(params['local_bg_outer'])


# ---------------------------------------------------------------------------
# U-Track LAP controls
# ---------------------------------------------------------------------------

class UTrackLAPControlGroup(QtWidgets.QGroupBox):
    """Advanced U-Track LAP linking parameters.

    Exposes motion model, multi-round (FRF), adaptive search,
    merge/split detection, cost matrix, Kalman filter, and gap closing
    controls — matching the full U-Track 2.5 parameter set.
    """

    def __init__(self, parent=None):
        super().__init__("U-Track LAP Parameters", parent)
        self.setCheckable(False)
        self._build()
        self._connect()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Primary controls (always visible) ---
        primary = QtWidgets.QFormLayout()
        primary.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        primary.setSpacing(4)

        self.motion_model = QtWidgets.QComboBox()
        self.motion_model.addItems(['brownian', 'linear', 'confined', 'mixed'])
        self.motion_model.setCurrentText('mixed')
        primary.addRow("Motion model:", self.motion_model)

        self.multi_round_cb = QtWidgets.QCheckBox()
        self.multi_round_cb.setChecked(True)
        self.multi_round_cb.setToolTip(
            "Forward-Reverse-Forward multi-round tracking")
        primary.addRow("Multi-round (FRF):", self.multi_round_cb)

        self.tracking_rounds = _make_int_spin(3, 1, 5)
        primary.addRow("Tracking rounds:", self.tracking_rounds)

        self.velocity_persistence = _make_double_spin(1.0, 0.0, 1.0,
                                                       decimals=2, step=0.05)
        self.velocity_persistence.setToolTip(
            "Fraction of velocity retained per step in linear model.\n"
            "1.0 = constant velocity (U-Track 2.5 default).\n"
            "0.0 = Brownian (no velocity memory).")
        primary.addRow("Velocity persistence:", self.velocity_persistence)

        self.merge_split_cb = QtWidgets.QCheckBox()
        self.merge_split_cb.setChecked(True)
        self.merge_split_cb.setToolTip(
            "Detect merge and split events during gap closing")
        primary.addRow("Merge/split detection:", self.merge_split_cb)

        self.use_intensity_cb = QtWidgets.QCheckBox()
        self.use_intensity_cb.setToolTip(
            "Include intensity ratio in the linking cost matrix")
        primary.addRow("Use intensity costs:", self.use_intensity_cb)

        self.use_velocity_angle_cb = QtWidgets.QCheckBox()
        self.use_velocity_angle_cb.setToolTip(
            "Include velocity angle in the linking cost matrix")
        primary.addRow("Use velocity angle costs:", self.use_velocity_angle_cb)

        layout.addLayout(primary)

        # --- Kalman Filter section ---
        sec_kalman = _CollapsibleSection("Kalman Filter")

        self.process_noise_brownian = _make_double_spin(1.0, 0.01, 50.0,
                                                          decimals=2, step=0.1)
        self.process_noise_brownian.setToolTip(
            "Process noise for the Brownian motion model (q).")
        sec_kalman.addRow("Process noise (Brownian):", self.process_noise_brownian)

        self.process_noise_linear = _make_double_spin(0.5, 0.01, 50.0,
                                                        decimals=2, step=0.1)
        self.process_noise_linear.setToolTip(
            "Process noise for the linear (directed) motion model.")
        sec_kalman.addRow("Process noise (Linear):", self.process_noise_linear)

        self.process_noise_confined = _make_double_spin(1.0, 0.01, 50.0,
                                                          decimals=2, step=0.1)
        self.process_noise_confined.setToolTip(
            "Process noise for the confined motion model.")
        sec_kalman.addRow("Process noise (Confined):", self.process_noise_confined)

        self.measurement_noise = _make_double_spin(1.0, 0.01, 50.0,
                                                     decimals=2, step=0.1)
        self.measurement_noise.setToolTip(
            "Measurement noise (localization precision).\n"
            "Can be auto-estimated from detection uncertainties.")
        sec_kalman.addRow("Measurement noise:", self.measurement_noise)

        self.confinement_spring = _make_double_spin(0.1, 0.001, 1.0,
                                                      decimals=3, step=0.01)
        self.confinement_spring.setToolTip(
            "Spring constant for confined (Ornstein-Uhlenbeck) model.\n"
            "Higher = stronger confinement.")
        sec_kalman.addRow("Confinement spring:", self.confinement_spring)

        self.auto_estimate_noise_cb = QtWidgets.QCheckBox()
        self.auto_estimate_noise_cb.setToolTip(
            "Auto-set measurement noise from median localization\n"
            "uncertainty (Cramér-Rao bound from detection).")
        sec_kalman.addRow("Auto-estimate noise:", self.auto_estimate_noise_cb)

        layout.addWidget(sec_kalman)

        # --- Adaptive Search section ---
        sec_search = _CollapsibleSection("Adaptive Search")

        self.adaptive_search_cb = QtWidgets.QCheckBox()
        self.adaptive_search_cb.setChecked(True)
        self.adaptive_search_cb.setToolTip(
            "Adapt search radius based on local particle density")
        sec_search.addRow("Enable adaptive search:", self.adaptive_search_cb)

        self.time_reach_confidence = _make_int_spin(5, 1, 50,
                                                      suffix=' frames')
        self.time_reach_confidence.setToolTip(
            "Number of frames before Kalman prediction reaches\n"
            "full confidence (ramp-up period for new tracks).")
        sec_search.addRow("Confidence ramp-up:", self.time_reach_confidence)

        self.density_scaling = _make_double_spin(0.5, 0.0, 2.0,
                                                   decimals=2, step=0.05)
        self.density_scaling.setToolTip(
            "Search radius = nearest-neighbor distance × this factor.\n"
            "Lower = more conservative in dense regions.")
        sec_search.addRow("Density scaling:", self.density_scaling)

        self.min_search_radius = _make_double_spin(1.0, 0.1, 20.0,
                                                     decimals=1, step=0.5,
                                                     suffix=' px')
        self.min_search_radius.setToolTip("Minimum allowed search radius.")
        sec_search.addRow("Min search radius:", self.min_search_radius)

        self.max_search_radius = _make_double_spin(50.0, 5.0, 200.0,
                                                     decimals=1, step=5.0,
                                                     suffix=' px')
        self.max_search_radius.setToolTip("Maximum allowed search radius.")
        sec_search.addRow("Max search radius:", self.max_search_radius)

        layout.addWidget(sec_search)

        # --- Cost Matrix section ---
        sec_cost = _CollapsibleSection("Cost Matrix")

        self.intensity_weight = _make_double_spin(0.1, 0.0, 5.0,
                                                    decimals=2, step=0.05)
        self.intensity_weight.setToolTip(
            "Weight for intensity ratio cost term: |log(I2/I1)|.")
        sec_cost.addRow("Intensity weight:", self.intensity_weight)

        self.velocity_angle_weight = _make_double_spin(0.1, 0.0, 5.0,
                                                         decimals=2, step=0.05)
        self.velocity_angle_weight.setToolTip(
            "Weight for velocity angle cost term: (1−cos θ) × speed.")
        sec_cost.addRow("Velocity angle weight:", self.velocity_angle_weight)

        self.uncertainty_weight = _make_double_spin(1.0, 0.0, 10.0,
                                                      decimals=2, step=0.1)
        self.uncertainty_weight.setToolTip(
            "Weight for localization uncertainty normalization.\n"
            "Cost is divided by (1 + w × (σ_track² + σ_det²)).\n"
            "Set to 0 to disable uncertainty weighting.")
        sec_cost.addRow("Uncertainty weight:", self.uncertainty_weight)

        self.alt_cost_percentile = _make_double_spin(90.0, 50.0, 99.9,
                                                       decimals=1, step=1.0,
                                                       suffix=' %')
        self.alt_cost_percentile.setToolTip(
            "Percentile of valid costs used to compute the\n"
            "alternative (birth/death) cost threshold.")
        sec_cost.addRow("Alt cost percentile:", self.alt_cost_percentile)

        self.alt_cost_factor = _make_double_spin(1.05, 1.0, 2.0,
                                                   decimals=2, step=0.05)
        self.alt_cost_factor.setToolTip(
            "Multiplier applied to the percentile to get\n"
            "the final alternative cost.")
        sec_cost.addRow("Alt cost factor:", self.alt_cost_factor)

        layout.addWidget(sec_cost)

        # --- Amplitude Gating section ---
        sec_gate = _CollapsibleSection("Amplitude Gating")

        self.amp_gate_min = _make_double_spin(0.0, 0.0, 100000.0,
                                                decimals=1, step=10.0)
        self.amp_gate_min.setToolTip(
            "Minimum amplitude for linking (localizations below\n"
            "this threshold are excluded before linking).")
        sec_gate.addRow("Min amplitude:", self.amp_gate_min)

        self.amp_gate_max = _make_double_spin(0.0, 0.0, 100000.0,
                                                decimals=1, step=100.0)
        self.amp_gate_max.setToolTip(
            "Maximum amplitude for linking. Set to 0 for no upper limit.")
        sec_gate.addRow("Max amplitude:", self.amp_gate_max)

        layout.addWidget(sec_gate)

        # --- Gap Closing section ---
        sec_gap = _CollapsibleSection("Gap Closing")

        self.gap_closing_max_dist = _make_double_spin(15.0, 1.0, 200.0,
                                                        decimals=1, step=1.0,
                                                        suffix=' px')
        self.gap_closing_max_dist.setToolTip(
            "Maximum distance for gap closing (separate from\n"
            "frame-to-frame max distance).")
        sec_gap.addRow("Max distance:", self.gap_closing_max_dist)

        self.gap_penalty = _make_double_spin(1.5, 1.0, 10.0,
                                               decimals=2, step=0.1)
        self.gap_penalty.setToolTip(
            "Multiplicative penalty per gap frame (legacy mode).\n"
            "Not used when d²/dt normalization is active.")
        sec_gap.addRow("Gap penalty:", self.gap_penalty)

        self.mobility_scaling_cb = QtWidgets.QCheckBox()
        self.mobility_scaling_cb.setChecked(True)
        self.mobility_scaling_cb.setToolTip(
            "Scale gap closing cost by track mobility (mean step size).")
        sec_gap.addRow("Mobility scaling:", self.mobility_scaling_cb)

        self.intensity_ratio_lo = _make_double_spin(0.5, 0.0, 1.0,
                                                      decimals=2, step=0.05)
        self.intensity_ratio_lo.setToolTip(
            "Lower bound for merge/split intensity ratio validation.")
        sec_gap.addRow("Intensity ratio min:", self.intensity_ratio_lo)

        self.intensity_ratio_hi = _make_double_spin(2.0, 1.0, 10.0,
                                                      decimals=2, step=0.1)
        self.intensity_ratio_hi.setToolTip(
            "Upper bound for merge/split intensity ratio validation.")
        sec_gap.addRow("Intensity ratio max:", self.intensity_ratio_hi)

        layout.addWidget(sec_gap)

        # --- Post-tracking section ---
        sec_post = _CollapsibleSection("Post-tracking Analysis")

        self.msd_max_lag = _make_int_spin(10, 1, 100, suffix=' frames')
        self.msd_max_lag.setToolTip("Maximum lag for MSD computation.")
        sec_post.addRow("MSD max lag:", self.msd_max_lag)

        self.velocity_smoothing = _make_int_spin(3, 1, 21, suffix=' pts')
        self.velocity_smoothing.setToolTip(
            "Window size for velocity smoothing (moving average).")
        sec_post.addRow("Velocity smoothing:", self.velocity_smoothing)

        layout.addWidget(sec_post)

    def _connect(self):
        self.multi_round_cb.toggled.connect(self.tracking_rounds.setEnabled)
        # Sync checkboxes with weight spinboxes
        self.use_intensity_cb.toggled.connect(
            lambda on: self.intensity_weight.setValue(
                0.1 if on else 0.0))
        self.use_velocity_angle_cb.toggled.connect(
            lambda on: self.velocity_angle_weight.setValue(
                0.1 if on else 0.0))

    def get_config(self):
        """Return a dict of kwargs for ``TrackingConfig``."""
        rounds = (self.tracking_rounds.value()
                  if self.multi_round_cb.isChecked() else 1)

        amp_max = self.amp_gate_max.value()
        if amp_max <= 0:
            amp_max = float('inf')

        return {
            'motion_model': self.motion_model.currentText(),
            'num_tracking_rounds': rounds,
            'velocity_persistence': self.velocity_persistence.value(),
            'merge_split_enabled': self.merge_split_cb.isChecked(),
            # Kalman filter
            'process_noise_brownian': self.process_noise_brownian.value(),
            'process_noise_linear': self.process_noise_linear.value(),
            'process_noise_confined': self.process_noise_confined.value(),
            'measurement_noise': self.measurement_noise.value(),
            'confinement_spring': self.confinement_spring.value(),
            'auto_estimate_noise': self.auto_estimate_noise_cb.isChecked(),
            # Adaptive search
            'time_reach_confidence': self.time_reach_confidence.value(),
            'density_scaling_factor': self.density_scaling.value(),
            'min_search_radius': self.min_search_radius.value(),
            'max_search_radius': self.max_search_radius.value(),
            # Cost matrix
            'intensity_weight': self.intensity_weight.value(),
            'velocity_angle_weight': self.velocity_angle_weight.value(),
            'uncertainty_weight': self.uncertainty_weight.value(),
            'alternative_cost_percentile': self.alt_cost_percentile.value(),
            'alternative_cost_factor': self.alt_cost_factor.value(),
            # Amplitude gating
            'amplitude_gate': (self.amp_gate_min.value(), amp_max),
            # Gap closing
            'gap_closing_max_distance': self.gap_closing_max_dist.value(),
            'gap_penalty': self.gap_penalty.value(),
            'mobility_scaling': self.mobility_scaling_cb.isChecked(),
            'intensity_ratio_range': (self.intensity_ratio_lo.value(),
                                      self.intensity_ratio_hi.value()),
            # Post-tracking
            'msd_max_lag': self.msd_max_lag.value(),
            'velocity_smoothing_window': self.velocity_smoothing.value(),
        }

    def set_config(self, config):
        """Set controls from a dict."""
        if 'motion_model' in config:
            idx = self.motion_model.findText(config['motion_model'])
            if idx >= 0:
                self.motion_model.setCurrentIndex(idx)
        if 'num_tracking_rounds' in config:
            rounds = config['num_tracking_rounds']
            self.multi_round_cb.setChecked(rounds > 1)
            self.tracking_rounds.setValue(rounds)
        if 'velocity_persistence' in config:
            self.velocity_persistence.setValue(config['velocity_persistence'])
        if 'merge_split_enabled' in config:
            self.merge_split_cb.setChecked(config['merge_split_enabled'])
        # Kalman filter
        if 'process_noise_brownian' in config:
            self.process_noise_brownian.setValue(config['process_noise_brownian'])
        if 'process_noise_linear' in config:
            self.process_noise_linear.setValue(config['process_noise_linear'])
        if 'process_noise_confined' in config:
            self.process_noise_confined.setValue(config['process_noise_confined'])
        if 'measurement_noise' in config:
            self.measurement_noise.setValue(config['measurement_noise'])
        if 'confinement_spring' in config:
            self.confinement_spring.setValue(config['confinement_spring'])
        if 'auto_estimate_noise' in config:
            self.auto_estimate_noise_cb.setChecked(config['auto_estimate_noise'])
        # Adaptive search
        if 'time_reach_confidence' in config:
            self.time_reach_confidence.setValue(config['time_reach_confidence'])
        if 'density_scaling_factor' in config:
            self.density_scaling.setValue(config['density_scaling_factor'])
        if 'min_search_radius' in config:
            self.min_search_radius.setValue(config['min_search_radius'])
        if 'max_search_radius' in config:
            self.max_search_radius.setValue(config['max_search_radius'])
        # Cost matrix
        if 'intensity_weight' in config:
            self.intensity_weight.setValue(config['intensity_weight'])
            self.use_intensity_cb.setChecked(config['intensity_weight'] > 0)
        if 'velocity_angle_weight' in config:
            self.velocity_angle_weight.setValue(config['velocity_angle_weight'])
            self.use_velocity_angle_cb.setChecked(
                config['velocity_angle_weight'] > 0)
        if 'uncertainty_weight' in config:
            self.uncertainty_weight.setValue(config['uncertainty_weight'])
        if 'alternative_cost_percentile' in config:
            self.alt_cost_percentile.setValue(
                config['alternative_cost_percentile'])
        if 'alternative_cost_factor' in config:
            self.alt_cost_factor.setValue(config['alternative_cost_factor'])
        # Amplitude gating
        if 'amplitude_gate' in config:
            lo, hi = config['amplitude_gate']
            self.amp_gate_min.setValue(lo)
            self.amp_gate_max.setValue(hi if hi < float('inf') else 0.0)
        # Gap closing
        if 'gap_closing_max_distance' in config:
            self.gap_closing_max_dist.setValue(
                config['gap_closing_max_distance'])
        if 'gap_penalty' in config:
            self.gap_penalty.setValue(config['gap_penalty'])
        if 'mobility_scaling' in config:
            self.mobility_scaling_cb.setChecked(config['mobility_scaling'])
        if 'intensity_ratio_range' in config:
            lo, hi = config['intensity_ratio_range']
            self.intensity_ratio_lo.setValue(lo)
            self.intensity_ratio_hi.setValue(hi)
        # Post-tracking
        if 'msd_max_lag' in config:
            self.msd_max_lag.setValue(config['msd_max_lag'])
        if 'velocity_smoothing_window' in config:
            self.velocity_smoothing.setValue(
                config['velocity_smoothing_window'])


# ---------------------------------------------------------------------------
# Trackpy controls
# ---------------------------------------------------------------------------

class TrackpyControlGroup(QtWidgets.QGroupBox):
    """Trackpy linking parameter controls.

    Exposes the four linking types and adaptive search parameters.
    """

    def __init__(self, parent=None):
        super().__init__("Trackpy Parameters", parent)
        self.setCheckable(False)
        self._build()
        self._connect()

    def _build(self):
        layout = QtWidgets.QFormLayout(self)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.link_type = QtWidgets.QComboBox()
        self.link_type.addItems([
            'standard', 'adaptive', 'velocityPredict',
            'adaptive + velocityPredict'])
        layout.addRow("Link type:", self.link_type)

        self.adaptive_stop = _make_double_spin(0.0, 0.0, 20.0,
                                                decimals=1, step=0.5)
        self.adaptive_stop.setToolTip(
            "Minimum search range for adaptive mode (0 = auto)")
        self.adaptive_stop.setVisible(False)
        self._stop_label = QtWidgets.QLabel("Adaptive stop:")
        self._stop_label.setVisible(False)
        layout.addRow(self._stop_label, self.adaptive_stop)

        self.adaptive_step = _make_double_spin(0.95, 0.5, 0.99,
                                                decimals=2, step=0.01)
        self.adaptive_step.setToolTip(
            "Multiplicative factor for adaptive search reduction")
        self.adaptive_step.setVisible(False)
        self._step_label = QtWidgets.QLabel("Adaptive step:")
        self._step_label.setVisible(False)
        layout.addRow(self._step_label, self.adaptive_step)

    def _connect(self):
        self.link_type.currentTextChanged.connect(self._on_type_changed)

    def _on_type_changed(self, text):
        is_adaptive = 'adaptive' in text.lower()
        self.adaptive_stop.setVisible(is_adaptive)
        self._stop_label.setVisible(is_adaptive)
        self.adaptive_step.setVisible(is_adaptive)
        self._step_label.setVisible(is_adaptive)

    def get_params(self):
        """Return kwargs for ``link_with_trackpy``."""
        link_type = self.link_type.currentText()
        params = {'link_type': link_type}
        if 'adaptive' in link_type.lower():
            stop_val = self.adaptive_stop.value()
            params['adaptive_stop'] = stop_val if stop_val > 0 else None
            params['adaptive_step'] = self.adaptive_step.value()
        return params

    def set_params(self, params):
        """Set controls from a dict."""
        if 'link_type' in params:
            idx = self.link_type.findText(params['link_type'])
            if idx >= 0:
                self.link_type.setCurrentIndex(idx)
        if 'adaptive_stop' in params and params['adaptive_stop'] is not None:
            self.adaptive_stop.setValue(params['adaptive_stop'])
        if 'adaptive_step' in params:
            self.adaptive_step.setValue(params['adaptive_step'])
