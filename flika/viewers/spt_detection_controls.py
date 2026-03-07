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
from qtpy import QtWidgets


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

    Exposes all filter, detector, fitter, and camera parameters matching
    the original spt_batch_analysis plugin and the ThunderSTORM ImageJ
    plugin (Ovesny et al., 2014).
    """

    def __init__(self, parent=None):
        super().__init__("ThunderSTORM Parameters", parent)
        self.setCheckable(False)
        self._build()
        self._connect()

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Image Filter ---
        filter_group = QtWidgets.QGroupBox("Image Filter")
        fl = QtWidgets.QFormLayout(filter_group)
        fl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems([
            'wavelet', 'gaussian', 'dog', 'lowered_gaussian',
            'median', 'box', 'none'])
        fl.addRow("Filter type:", self.filter_type)

        self.wavelet_scale = _make_int_spin(2, 1, 5)
        fl.addRow("Wavelet scale:", self.wavelet_scale)

        layout.addWidget(filter_group)

        # --- Detector ---
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
            'std(Wave.F1)', '2*std(Wave.F1)', '3*std(Wave.F1)'])
        dl.addRow("Threshold:", self.threshold)

        self.watershed_cb = QtWidgets.QCheckBox()
        self.watershed_cb.setToolTip(
            "Apply watershed segmentation (centroid detector only)")
        self.watershed_cb.setVisible(False)
        self._watershed_label = QtWidgets.QLabel("Watershed:")
        self._watershed_label.setVisible(False)
        dl.addRow(self._watershed_label, self.watershed_cb)

        layout.addWidget(det_group)

        # --- Fitter ---
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
        ftl.addRow("Fit radius:", self.fit_radius)

        self.initial_sigma = _make_double_spin(1.3, 0.5, 5.0, decimals=2,
                                                step=0.1, suffix=' px')
        ftl.addRow("Initial sigma:", self.initial_sigma)

        # Multi-emitter options (visible only when fitter=multi_emitter)
        self.multi_emitter_cb = QtWidgets.QCheckBox()
        self.multi_emitter_cb.setVisible(False)
        self._multi_label = QtWidgets.QLabel("Multi-emitter:")
        self._multi_label.setVisible(False)
        ftl.addRow(self._multi_label, self.multi_emitter_cb)

        self.max_emitters = _make_int_spin(5, 2, 10)
        self.max_emitters.setVisible(False)
        self._max_em_label = QtWidgets.QLabel("Max emitters:")
        self._max_em_label.setVisible(False)
        ftl.addRow(self._max_em_label, self.max_emitters)

        self.p_value = QtWidgets.QComboBox()
        self.p_value.addItems(['1e-6', '1e-4', '1e-3', '0.01', '0.05'])
        self.p_value.setVisible(False)
        self._pval_label = QtWidgets.QLabel("p-value threshold:")
        self._pval_label.setVisible(False)
        ftl.addRow(self._pval_label, self.p_value)

        layout.addWidget(fit_group)

        # --- Camera Model ---
        cam_group = QtWidgets.QGroupBox("Camera Model")
        cl = QtWidgets.QFormLayout(cam_group)
        cl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

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

    def _connect(self):
        self.filter_type.currentTextChanged.connect(self._on_filter_changed)
        self.detector_type.currentTextChanged.connect(self._on_detector_changed)
        self.fitter_type.currentTextChanged.connect(self._on_fitter_changed)
        self.em_gain_cb.toggled.connect(self.em_gain_value.setEnabled)

    def _on_filter_changed(self, text):
        is_wavelet = (text == 'wavelet')
        self.wavelet_scale.setVisible(is_wavelet)
        # Find parent label
        form = self.wavelet_scale.parent()
        if form is not None:
            label = form.layout()
            if hasattr(label, 'labelForField'):
                lbl = label.labelForField(self.wavelet_scale)
                if lbl:
                    lbl.setVisible(is_wavelet)

    def _on_detector_changed(self, text):
        is_centroid = (text == 'centroid')
        self.watershed_cb.setVisible(is_centroid)
        self._watershed_label.setVisible(is_centroid)

    def _on_fitter_changed(self, text):
        is_multi = (text == 'multi_emitter')
        self.multi_emitter_cb.setVisible(is_multi)
        self._multi_label.setVisible(is_multi)
        self.max_emitters.setVisible(is_multi)
        self._max_em_label.setVisible(is_multi)
        self.p_value.setVisible(is_multi)
        self._pval_label.setVisible(is_multi)

    def get_params(self):
        """Return a dict of kwargs for ``ThunderSTORMDetector.__init__``."""
        filter_type = self.filter_type.currentText()
        filter_params = {}
        if filter_type == 'wavelet':
            filter_params['scale'] = self.wavelet_scale.value()

        detector_type = self.detector_type.currentText()
        detector_params = {}
        if detector_type == 'centroid' and self.watershed_cb.isChecked():
            detector_params['use_watershed'] = True

        fitter_type = self.fitter_type.currentText()
        fitter_params = {
            'initial_sigma': self.initial_sigma.value(),
        }
        if fitter_type == 'multi_emitter':
            fitter_params['max_emitters'] = self.max_emitters.value()
            fitter_params['p_value_threshold'] = float(self.p_value.currentText())

        camera_params = {
            'photons_per_adu': self.photons_per_adu.value(),
            'baseline': self.baseline.value(),
            'is_emccd': self.em_gain_cb.isChecked(),
            'em_gain': self.em_gain_value.value() if self.em_gain_cb.isChecked() else 1.0,
            'quantum_efficiency': self.quantum_efficiency.value(),
        }

        threshold_text = self.threshold.currentText().strip()
        try:
            threshold = float(threshold_text)
        except ValueError:
            threshold = threshold_text if threshold_text else None

        roi_size = self.fit_radius.value() * 2 + 1

        return {
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

        fp = params.get('fitter_params', {})
        if 'initial_sigma' in fp:
            self.initial_sigma.setValue(fp['initial_sigma'])
        if 'max_emitters' in fp:
            self.max_emitters.setValue(fp['max_emitters'])
        if 'p_value_threshold' in fp:
            idx = self.p_value.findText(str(fp['p_value_threshold']))
            if idx >= 0:
                self.p_value.setCurrentIndex(idx)

        cp = params.get('camera_params', {})
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
        elif 'filter_params' in params:
            ffp = params['filter_params']
            if 'scale' in ffp:
                self.wavelet_scale.setValue(ffp['scale'])


# ---------------------------------------------------------------------------
# U-Track LAP controls
# ---------------------------------------------------------------------------

class UTrackLAPControlGroup(QtWidgets.QGroupBox):
    """Advanced U-Track LAP linking parameters.

    Exposes motion model, multi-round (FRF), adaptive search,
    merge/split detection, and cost matrix controls.
    """

    def __init__(self, parent=None):
        super().__init__("U-Track LAP Parameters", parent)
        self.setCheckable(False)
        self._build()
        self._connect()

    def _build(self):
        layout = QtWidgets.QFormLayout(self)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.motion_model = QtWidgets.QComboBox()
        self.motion_model.addItems(['brownian', 'linear', 'confined', 'mixed'])
        self.motion_model.setCurrentText('mixed')
        layout.addRow("Motion model:", self.motion_model)

        self.multi_round_cb = QtWidgets.QCheckBox()
        self.multi_round_cb.setChecked(True)
        self.multi_round_cb.setToolTip(
            "Forward-Reverse-Forward multi-round tracking")
        layout.addRow("Multi-round (FRF):", self.multi_round_cb)

        self.tracking_rounds = _make_int_spin(3, 1, 5)
        layout.addRow("Tracking rounds:", self.tracking_rounds)

        self.adaptive_search_cb = QtWidgets.QCheckBox()
        self.adaptive_search_cb.setChecked(True)
        self.adaptive_search_cb.setToolTip(
            "Adapt search radius based on local particle density")
        layout.addRow("Adaptive search:", self.adaptive_search_cb)

        self.velocity_persistence = _make_double_spin(0.8, 0.0, 1.0,
                                                       decimals=2, step=0.05)
        self.velocity_persistence.setToolTip(
            "Weight for velocity-based prediction (0=none, 1=full)")
        layout.addRow("Velocity persistence:", self.velocity_persistence)

        self.merge_split_cb = QtWidgets.QCheckBox()
        self.merge_split_cb.setChecked(True)
        self.merge_split_cb.setToolTip(
            "Detect merge and split events during gap closing")
        layout.addRow("Merge/split detection:", self.merge_split_cb)

        self.use_intensity_cb = QtWidgets.QCheckBox()
        self.use_intensity_cb.setToolTip(
            "Include intensity ratio in the linking cost matrix")
        layout.addRow("Use intensity costs:", self.use_intensity_cb)

        self.use_velocity_angle_cb = QtWidgets.QCheckBox()
        self.use_velocity_angle_cb.setToolTip(
            "Include velocity angle in the linking cost matrix")
        layout.addRow("Use velocity angle costs:", self.use_velocity_angle_cb)

    def _connect(self):
        self.multi_round_cb.toggled.connect(self.tracking_rounds.setEnabled)

    def get_config(self):
        """Return a dict of kwargs for ``TrackingConfig`` / ``UTrackLinker``."""
        rounds = self.tracking_rounds.value() if self.multi_round_cb.isChecked() else 1
        intensity_w = 0.1 if self.use_intensity_cb.isChecked() else 0.0
        angle_w = 0.1 if self.use_velocity_angle_cb.isChecked() else 0.0

        return {
            'motion_model': self.motion_model.currentText(),
            'num_tracking_rounds': rounds,
            'velocity_persistence': self.velocity_persistence.value(),
            'merge_split_enabled': self.merge_split_cb.isChecked(),
            'intensity_weight': intensity_w,
            'velocity_angle_weight': angle_w,
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
        if 'intensity_weight' in config:
            self.use_intensity_cb.setChecked(config['intensity_weight'] > 0)
        if 'velocity_angle_weight' in config:
            self.use_velocity_angle_cb.setChecked(
                config['velocity_angle_weight'] > 0)


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
