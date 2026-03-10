# -*- coding: utf-8 -*-
"""Simulation dialog for microscopy data generation."""
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Signal

from .engine import SimulationConfig
from .noise import CameraConfig
from .fluorophores import FLUOROPHORE_PRESETS
from .presets import PRESETS


class _CollapsibleSection(QtWidgets.QWidget):
    """A section with a toggle button that shows/hides its contents."""

    def __init__(self, title, parent=None, expanded=False):
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

        if expanded:
            self._toggle.setChecked(True)

    def _on_toggle(self, checked):
        self._toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if checked
            else QtCore.Qt.ArrowType.RightArrow)
        self._content.setVisible(checked)

    def addRow(self, label, widget):
        self._content_layout.addRow(label, widget)

    def addWidget(self, widget):
        self._content_layout.addRow(widget)


class SimulationWorker(QtCore.QThread):
    """Background worker for simulation execution."""
    sig_progress = Signal(int, str)
    sig_finished = Signal(object, object)
    sig_error = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config

    def run(self):
        try:
            from .engine import SimulationEngine
            engine = SimulationEngine(self._config)
            stack, metadata = engine.run(
                progress_callback=lambda p, m: self.sig_progress.emit(p, m))
            self.sig_finished.emit(stack, metadata)
        except Exception as e:
            self.sig_error.emit(str(e))


class SimulationDialog(QtWidgets.QDialog):
    """Microscopy Simulation Builder dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Microscopy Simulation Builder")
        self.setMinimumSize(550, 600)
        self._worker = None

        layout = QtWidgets.QVBoxLayout(self)

        # Tab widget
        self._tabs = QtWidgets.QTabWidget()
        layout.addWidget(self._tabs)

        self._build_sample_tab()
        self._build_microscope_tab()
        self._build_output_tab()
        self._build_quickstart_tab()
        # Move Quick Start to first position
        self._tabs.tabBar().moveTab(self._tabs.count() - 1, 0)
        self._tabs.setCurrentIndex(0)

        # Bottom bar
        bottom = QtWidgets.QHBoxLayout()
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._status_label = QtWidgets.QLabel("Ready")
        bottom.addWidget(self._progress, 1)
        bottom.addWidget(self._status_label)

        self._btn_generate = QtWidgets.QPushButton("Generate")
        self._btn_generate.clicked.connect(self._on_generate)
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.setEnabled(False)
        bottom.addWidget(self._btn_generate)
        bottom.addWidget(self._btn_cancel)
        layout.addLayout(bottom)

    # ------------------------------------------------------------------
    # Tab 1: Quick Start
    # ------------------------------------------------------------------
    def _build_quickstart_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)

        lay.addWidget(QtWidgets.QLabel("Select a preset to get started:"))

        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItems(list(PRESETS.keys()))
        self._preset_combo.currentTextChanged.connect(
            self._on_preset_changed)
        lay.addWidget(self._preset_combo)

        # Description
        self._preset_desc = QtWidgets.QTextEdit()
        self._preset_desc.setReadOnly(True)
        self._preset_desc.setMaximumHeight(120)
        lay.addWidget(self._preset_desc)

        btn_customize = QtWidgets.QPushButton("Customize...")
        btn_customize.clicked.connect(lambda: self._tabs.setCurrentIndex(1))
        lay.addWidget(btn_customize)

        lay.addStretch()
        self._tabs.addTab(tab, "Quick Start")
        self._on_preset_changed(self._preset_combo.currentText())

    # ------------------------------------------------------------------
    # Tab 2: Sample
    # ------------------------------------------------------------------
    def _build_sample_tab(self):
        tab = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)

        # Structure section
        sec_struct = _CollapsibleSection("Structure", expanded=True)
        self._structure_combo = QtWidgets.QComboBox()
        self._structure_combo.addItems([
            'beads', 'filaments', 'cells', 'cell_field'])
        sec_struct.addRow("Type:", self._structure_combo)

        self._n_beads = QtWidgets.QSpinBox()
        self._n_beads.setRange(1, 10000)
        self._n_beads.setValue(100)
        sec_struct.addRow("Count:", self._n_beads)

        self._cell_type_combo = QtWidgets.QComboBox()
        self._cell_type_combo.addItems([
            'round', 'elongated', 'irregular', 'neuron'])
        sec_struct.addRow("Cell type:", self._cell_type_combo)

        self._organelles_check = QtWidgets.QCheckBox("Include organelles")
        sec_struct.addRow("", self._organelles_check)

        self._n_filaments = QtWidgets.QSpinBox()
        self._n_filaments.setRange(1, 200)
        self._n_filaments.setValue(10)
        sec_struct.addRow("Filaments:", self._n_filaments)

        self._persistence = QtWidgets.QDoubleSpinBox()
        self._persistence.setRange(1, 200)
        self._persistence.setValue(20.0)
        sec_struct.addRow("Persistence:", self._persistence)

        lay.addWidget(sec_struct)

        # Fluorophore section
        sec_fluor = _CollapsibleSection("Fluorophore", expanded=True)
        self._fluor_combo = QtWidgets.QComboBox()
        self._fluor_combo.addItems(list(FLUOROPHORE_PRESETS.keys()))
        sec_fluor.addRow("Preset:", self._fluor_combo)

        self._photons_spin = QtWidgets.QDoubleSpinBox()
        self._photons_spin.setRange(100, 100000)
        self._photons_spin.setValue(2000)
        sec_fluor.addRow("Photons/frame:", self._photons_spin)

        self._label_density = QtWidgets.QDoubleSpinBox()
        self._label_density.setRange(0.01, 1.0)
        self._label_density.setValue(1.0)
        self._label_density.setSingleStep(0.1)
        sec_fluor.addRow("Label density:", self._label_density)

        lay.addWidget(sec_fluor)

        # Dynamics section
        sec_dyn = _CollapsibleSection("Dynamics")
        self._motion_combo = QtWidgets.QComboBox()
        self._motion_combo.addItems([
            'static', 'brownian', 'directed', 'confined', 'anomalous'])
        sec_dyn.addRow("Motion:", self._motion_combo)

        self._diffusion_coeff = QtWidgets.QDoubleSpinBox()
        self._diffusion_coeff.setRange(0.001, 100)
        self._diffusion_coeff.setValue(0.1)
        self._diffusion_coeff.setDecimals(3)
        sec_dyn.addRow("D (um^2/s):", self._diffusion_coeff)

        self._bleach_rate = QtWidgets.QDoubleSpinBox()
        self._bleach_rate.setRange(0, 1)
        self._bleach_rate.setValue(0.0)
        self._bleach_rate.setDecimals(4)
        self._bleach_rate.setSingleStep(0.001)
        sec_dyn.addRow("Bleach rate:", self._bleach_rate)

        self._blink_on = QtWidgets.QDoubleSpinBox()
        self._blink_on.setRange(0, 1000)
        self._blink_on.setValue(0.0)
        sec_dyn.addRow("Blink on rate:", self._blink_on)

        self._blink_off = QtWidgets.QDoubleSpinBox()
        self._blink_off.setRange(0, 1000)
        self._blink_off.setValue(0.0)
        sec_dyn.addRow("Blink off rate:", self._blink_off)

        lay.addWidget(sec_dyn)
        lay.addStretch()

        scroll.setWidget(inner)
        tab_lay = QtWidgets.QVBoxLayout(tab)
        tab_lay.setContentsMargins(0, 0, 0, 0)
        tab_lay.addWidget(scroll)
        self._tabs.addTab(tab, "Sample")

    # ------------------------------------------------------------------
    # Tab 3: Microscope
    # ------------------------------------------------------------------
    def _build_microscope_tab(self):
        tab = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)

        # Modality section
        sec_mod = _CollapsibleSection("Modality", expanded=True)
        self._modality_combo = QtWidgets.QComboBox()
        self._modality_combo.addItems([
            'widefield', 'tirf', 'confocal', 'lightsheet',
            'sim', 'sted', 'smlm', 'dnapaint', 'flim'])
        sec_mod.addRow("Type:", self._modality_combo)

        # TIRF params
        self._tirf_depth = QtWidgets.QDoubleSpinBox()
        self._tirf_depth.setRange(50, 500)
        self._tirf_depth.setValue(100)
        sec_mod.addRow("TIRF depth (nm):", self._tirf_depth)

        # Confocal params
        self._pinhole = QtWidgets.QDoubleSpinBox()
        self._pinhole.setRange(0.1, 5.0)
        self._pinhole.setValue(1.0)
        sec_mod.addRow("Pinhole (AU):", self._pinhole)

        # Light-sheet params
        self._sheet_thickness = QtWidgets.QDoubleSpinBox()
        self._sheet_thickness.setRange(0.5, 20)
        self._sheet_thickness.setValue(2.0)
        sec_mod.addRow("Sheet thickness:", self._sheet_thickness)

        # SMLM params
        self._smlm_density = QtWidgets.QDoubleSpinBox()
        self._smlm_density.setRange(0.1, 100)
        self._smlm_density.setValue(2.0)
        sec_mod.addRow("SMLM density:", self._smlm_density)

        # DNA-PAINT params
        self._dp_kon = QtWidgets.QDoubleSpinBox()
        self._dp_kon.setRange(0.001, 10)
        self._dp_kon.setValue(0.1)
        self._dp_kon.setDecimals(3)
        sec_mod.addRow("k_on:", self._dp_kon)

        self._dp_koff = QtWidgets.QDoubleSpinBox()
        self._dp_koff.setRange(0.1, 100)
        self._dp_koff.setValue(5.0)
        sec_mod.addRow("k_off:", self._dp_koff)

        # FLIM params
        self._flim_bins = QtWidgets.QSpinBox()
        self._flim_bins.setRange(32, 4096)
        self._flim_bins.setValue(256)
        sec_mod.addRow("Time bins:", self._flim_bins)

        self._flim_range = QtWidgets.QDoubleSpinBox()
        self._flim_range.setRange(5, 100)
        self._flim_range.setValue(25.0)
        sec_mod.addRow("Time range (ns):", self._flim_range)

        # SIM params
        self._sim_orientations = QtWidgets.QSpinBox()
        self._sim_orientations.setRange(2, 5)
        self._sim_orientations.setValue(3)
        sec_mod.addRow("SIM orientations:", self._sim_orientations)

        lay.addWidget(sec_mod)

        # Optics section
        sec_opt = _CollapsibleSection("Optics", expanded=True)
        self._na_spin = QtWidgets.QDoubleSpinBox()
        self._na_spin.setRange(0.1, 1.7)
        self._na_spin.setValue(1.4)
        self._na_spin.setDecimals(2)
        sec_opt.addRow("NA:", self._na_spin)

        self._wavelength_spin = QtWidgets.QDoubleSpinBox()
        self._wavelength_spin.setRange(300, 900)
        self._wavelength_spin.setValue(520)
        sec_opt.addRow("Wavelength (nm):", self._wavelength_spin)

        self._ri_spin = QtWidgets.QDoubleSpinBox()
        self._ri_spin.setRange(1.0, 1.7)
        self._ri_spin.setValue(1.515)
        self._ri_spin.setDecimals(3)
        sec_opt.addRow("n (immersion):", self._ri_spin)

        self._psf_combo = QtWidgets.QComboBox()
        self._psf_combo.addItems([
            'gaussian', 'airy', 'born_wolf', 'vectorial'])
        sec_opt.addRow("PSF model:", self._psf_combo)

        lay.addWidget(sec_opt)

        # Camera section
        sec_cam = _CollapsibleSection("Camera")
        self._cam_type = QtWidgets.QComboBox()
        self._cam_type.addItems(['CCD', 'EMCCD', 'sCMOS'])
        self._cam_type.setCurrentText('sCMOS')
        sec_cam.addRow("Type:", self._cam_type)

        self._cam_pixsize = QtWidgets.QDoubleSpinBox()
        self._cam_pixsize.setRange(1, 25)
        self._cam_pixsize.setValue(6.5)
        sec_cam.addRow("Pixel size (um):", self._cam_pixsize)

        self._cam_qe = QtWidgets.QDoubleSpinBox()
        self._cam_qe.setRange(0.1, 1.0)
        self._cam_qe.setValue(0.82)
        self._cam_qe.setDecimals(2)
        sec_cam.addRow("QE:", self._cam_qe)

        self._cam_read = QtWidgets.QDoubleSpinBox()
        self._cam_read.setRange(0, 50)
        self._cam_read.setValue(1.5)
        sec_cam.addRow("Read noise (e-):", self._cam_read)

        self._cam_dark = QtWidgets.QDoubleSpinBox()
        self._cam_dark.setRange(0, 10)
        self._cam_dark.setValue(0.06)
        self._cam_dark.setDecimals(3)
        sec_cam.addRow("Dark current:", self._cam_dark)

        self._cam_gain = QtWidgets.QDoubleSpinBox()
        self._cam_gain.setRange(0.1, 10)
        self._cam_gain.setValue(1.0)
        sec_cam.addRow("Gain (ADU/e-):", self._cam_gain)

        self._cam_emgain = QtWidgets.QDoubleSpinBox()
        self._cam_emgain.setRange(1, 1000)
        self._cam_emgain.setValue(300)
        sec_cam.addRow("EM gain:", self._cam_emgain)

        self._cam_baseline = QtWidgets.QSpinBox()
        self._cam_baseline.setRange(0, 10000)
        self._cam_baseline.setValue(100)
        sec_cam.addRow("Baseline (ADU):", self._cam_baseline)

        self._cam_bits = QtWidgets.QSpinBox()
        self._cam_bits.setRange(8, 32)
        self._cam_bits.setValue(16)
        sec_cam.addRow("Bit depth:", self._cam_bits)

        self._cam_exposure = QtWidgets.QDoubleSpinBox()
        self._cam_exposure.setRange(0.001, 60)
        self._cam_exposure.setValue(0.03)
        self._cam_exposure.setDecimals(3)
        sec_cam.addRow("Exposure (s):", self._cam_exposure)

        lay.addWidget(sec_cam)
        lay.addStretch()

        scroll.setWidget(inner)
        tab_lay = QtWidgets.QVBoxLayout(tab)
        tab_lay.setContentsMargins(0, 0, 0, 0)
        tab_lay.addWidget(scroll)
        self._tabs.addTab(tab, "Microscope")

    # ------------------------------------------------------------------
    # Tab 4: Output
    # ------------------------------------------------------------------
    def _build_output_tab(self):
        tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(tab)

        sec_dim = _CollapsibleSection("Dimensions", expanded=True)
        self._width_spin = QtWidgets.QSpinBox()
        self._width_spin.setRange(16, 4096)
        self._width_spin.setValue(256)
        sec_dim.addRow("Width (px):", self._width_spin)

        self._height_spin = QtWidgets.QSpinBox()
        self._height_spin.setRange(16, 4096)
        self._height_spin.setValue(256)
        sec_dim.addRow("Height (px):", self._height_spin)

        self._z_spin = QtWidgets.QSpinBox()
        self._z_spin.setRange(1, 500)
        self._z_spin.setValue(1)
        sec_dim.addRow("Z slices:", self._z_spin)

        self._t_spin = QtWidgets.QSpinBox()
        self._t_spin.setRange(1, 100000)
        self._t_spin.setValue(1)
        sec_dim.addRow("Time frames:", self._t_spin)

        self._pixsize_spin = QtWidgets.QDoubleSpinBox()
        self._pixsize_spin.setRange(0.01, 10)
        self._pixsize_spin.setValue(0.1)
        self._pixsize_spin.setDecimals(3)
        sec_dim.addRow("Pixel size (um):", self._pixsize_spin)

        self._zstep_spin = QtWidgets.QDoubleSpinBox()
        self._zstep_spin.setRange(0.1, 10)
        self._zstep_spin.setValue(0.3)
        sec_dim.addRow("Z step (um):", self._zstep_spin)

        self._dt_spin = QtWidgets.QDoubleSpinBox()
        self._dt_spin.setRange(0.001, 60)
        self._dt_spin.setValue(0.03)
        self._dt_spin.setDecimals(3)
        sec_dim.addRow("Frame interval (s):", self._dt_spin)

        lay.addWidget(sec_dim)

        sec_out = _CollapsibleSection("Output Options", expanded=True)
        self._gt_check = QtWidgets.QCheckBox("Include ground truth")
        self._gt_check.setChecked(True)
        sec_out.addRow("", self._gt_check)

        self._win_name = QtWidgets.QLineEdit("Simulation")
        sec_out.addRow("Window name:", self._win_name)

        self._bg_level = QtWidgets.QDoubleSpinBox()
        self._bg_level.setRange(0, 10000)
        self._bg_level.setValue(10)
        sec_out.addRow("Background:", self._bg_level)

        lay.addWidget(sec_out)
        lay.addStretch()
        self._tabs.addTab(tab, "Output")

    # ------------------------------------------------------------------
    # Config assembly
    # ------------------------------------------------------------------
    def _build_config(self):
        """Assemble SimulationConfig from dialog widgets."""
        # Structure params
        struct_type = self._structure_combo.currentText()
        struct_params = {}
        if struct_type == 'beads':
            struct_params['n_beads'] = self._n_beads.value()
        elif struct_type == 'filaments':
            struct_params['n_filaments'] = self._n_filaments.value()
            struct_params['persistence_length'] = self._persistence.value()
        elif struct_type == 'cells':
            struct_params['cell_type'] = self._cell_type_combo.currentText()
            struct_params['organelles'] = self._organelles_check.isChecked()
        elif struct_type == 'cell_field':
            struct_params['n_cells'] = self._n_beads.value()
            struct_params['cell_type'] = self._cell_type_combo.currentText()

        # Motion params
        motion_type = self._motion_combo.currentText()
        motion_params = {}
        if motion_type != 'static':
            motion_params['D'] = self._diffusion_coeff.value()

        # Modality params
        modality = self._modality_combo.currentText()
        mod_params = {'background': self._bg_level.value()}
        if modality == 'tirf':
            mod_params['penetration_depth'] = self._tirf_depth.value()
        elif modality == 'confocal':
            mod_params['pinhole_au'] = self._pinhole.value()
        elif modality == 'lightsheet':
            mod_params['sheet_thickness'] = self._sheet_thickness.value()
        elif modality == 'smlm':
            mod_params['density_per_frame'] = self._smlm_density.value()
        elif modality == 'dnapaint':
            mod_params['k_on'] = self._dp_kon.value()
            mod_params['k_off'] = self._dp_koff.value()
        elif modality == 'flim':
            mod_params['time_bins'] = self._flim_bins.value()
            mod_params['time_range'] = self._flim_range.value()

        camera = CameraConfig(
            type=self._cam_type.currentText(),
            pixel_size=self._cam_pixsize.value(),
            quantum_efficiency=self._cam_qe.value(),
            read_noise=self._cam_read.value(),
            dark_current=self._cam_dark.value(),
            gain=self._cam_gain.value(),
            em_gain=self._cam_emgain.value(),
            baseline=self._cam_baseline.value(),
            bit_depth=self._cam_bits.value(),
            exposure_time=self._cam_exposure.value(),
        )

        return SimulationConfig(
            nx=self._width_spin.value(),
            ny=self._height_spin.value(),
            nz=self._z_spin.value(),
            nt=self._t_spin.value(),
            pixel_size=self._pixsize_spin.value(),
            z_step=self._zstep_spin.value(),
            dt=self._dt_spin.value(),
            modality=modality,
            wavelength=self._wavelength_spin.value(),
            NA=self._na_spin.value(),
            n_immersion=self._ri_spin.value(),
            psf_model=self._psf_combo.currentText(),
            structure_type=struct_type,
            structure_params=struct_params,
            fluorophore=self._fluor_combo.currentText(),
            labeling_density=self._label_density.value(),
            camera=camera,
            motion_type=motion_type,
            motion_params=motion_params,
            modality_params=mod_params,
        )

    def _load_preset(self, config):
        """Load a SimulationConfig into the dialog widgets."""
        self._width_spin.setValue(config.nx)
        self._height_spin.setValue(config.ny)
        self._z_spin.setValue(config.nz)
        self._t_spin.setValue(config.nt)
        self._pixsize_spin.setValue(config.pixel_size)
        self._zstep_spin.setValue(config.z_step)
        self._dt_spin.setValue(config.dt)

        idx = self._modality_combo.findText(config.modality)
        if idx >= 0:
            self._modality_combo.setCurrentIndex(idx)
        self._wavelength_spin.setValue(config.wavelength)
        self._na_spin.setValue(config.NA)
        self._ri_spin.setValue(config.n_immersion)
        idx = self._psf_combo.findText(config.psf_model)
        if idx >= 0:
            self._psf_combo.setCurrentIndex(idx)

        idx = self._structure_combo.findText(config.structure_type)
        if idx >= 0:
            self._structure_combo.setCurrentIndex(idx)
        sp = config.structure_params
        self._n_beads.setValue(sp.get('n_beads', sp.get('n_cells', 100)))
        self._cell_type_combo.setCurrentText(
            sp.get('cell_type', 'round'))
        self._organelles_check.setChecked(sp.get('organelles', False))
        self._n_filaments.setValue(sp.get('n_filaments', 10))
        self._persistence.setValue(sp.get('persistence_length', 20.0))

        idx = self._fluor_combo.findText(config.fluorophore)
        if idx >= 0:
            self._fluor_combo.setCurrentIndex(idx)

        idx = self._motion_combo.findText(config.motion_type)
        if idx >= 0:
            self._motion_combo.setCurrentIndex(idx)
        self._diffusion_coeff.setValue(
            config.motion_params.get('D', 0.1))

        # Camera
        self._cam_type.setCurrentText(config.camera.type)
        self._cam_pixsize.setValue(config.camera.pixel_size)
        self._cam_qe.setValue(config.camera.quantum_efficiency)
        self._cam_read.setValue(config.camera.read_noise)
        self._cam_dark.setValue(config.camera.dark_current)
        self._cam_gain.setValue(config.camera.gain)
        self._cam_emgain.setValue(config.camera.em_gain)
        self._cam_baseline.setValue(config.camera.baseline)
        self._cam_bits.setValue(config.camera.bit_depth)
        self._cam_exposure.setValue(config.camera.exposure_time)

        # Modality params
        mp = config.modality_params
        self._tirf_depth.setValue(mp.get('penetration_depth', 100))
        self._pinhole.setValue(mp.get('pinhole_au', 1.0))
        self._sheet_thickness.setValue(mp.get('sheet_thickness', 2.0))
        self._smlm_density.setValue(mp.get('density_per_frame', 2.0))
        self._dp_kon.setValue(mp.get('k_on', 0.1))
        self._dp_koff.setValue(mp.get('k_off', 5.0))
        self._flim_bins.setValue(mp.get('time_bins', 256))
        self._flim_range.setValue(mp.get('time_range', 25.0))
        self._bg_level.setValue(mp.get('background', 10))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_preset_changed(self, name):
        if name in PRESETS:
            config = PRESETS[name]
            self._load_preset(config)
            desc = (f"Modality: {config.modality}\n"
                    f"Structure: {config.structure_type}\n"
                    f"Size: {config.nx}x{config.ny}"
                    f" z={config.nz} t={config.nt}\n"
                    f"Fluorophore: {config.fluorophore}\n"
                    f"Motion: {config.motion_type}")
            self._preset_desc.setPlainText(desc)

    def _on_generate(self):
        config = self._build_config()
        self._btn_generate.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setValue(0)

        self._worker = SimulationWorker(config, self)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_error.connect(self._on_error)
        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._btn_generate.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._status_label.setText("Cancelled")

    def _on_progress(self, pct, msg):
        self._progress.setValue(pct)
        self._status_label.setText(msg)

    def _on_finished(self, stack, metadata):
        self._btn_generate.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._status_label.setText("Done!")
        self._progress.setValue(100)

        # Create flika window
        try:
            from flika.window import Window
            name = self._win_name.text() or "Simulation"
            w = Window(stack, name=name)
            if self._gt_check.isChecked():
                w.metadata['simulation'] = metadata
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Could not create window:\n{e}")

    def _on_error(self, msg):
        self._btn_generate.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._status_label.setText("Error!")
        QtWidgets.QMessageBox.critical(
            self, "Simulation Error", msg)
