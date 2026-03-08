# -*- coding: utf-8 -*-
"""
SPT Control Panel
=================

Unified Single Particle Tracking analysis control panel.  This tabbed
dockable widget replaces the interfaces of the detect_puffs and pynsight
plugins with a single coherent workflow.

Six tabs expose all stages of an SPT pipeline:

1. **Detection** -- find particles in each frame
2. **Linking** -- connect particles across frames into tracks
3. **Analysis** -- compute geometric, kinematic, and spatial features
4. **Classification** -- train / apply motion classifiers
5. **Batch** -- process multiple files in sequence
6. **Visualization** -- launch dedicated plot windows

Usage::

    from flika.viewers.spt_control_panel import SPTControlPanel
    panel = SPTControlPanel.instance(parent=g.m)
    panel.show()

"""
import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Signal, Qt
import pyqtgraph as pg

from ..logger import logger


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_double_spin(value, lo, hi, decimals=3, suffix='', step=None):
    """Create a QDoubleSpinBox with sensible defaults."""
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
    """Create a QSpinBox with sensible defaults."""
    sb = QtWidgets.QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    return sb


def _make_hline():
    """Create a horizontal line separator."""
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return line


def _get_spt_data(win):
    """Safely return the ``window.metadata['spt']`` dict, creating if needed."""
    if win is None:
        return None
    md = getattr(win, 'metadata', None)
    if md is None:
        return None
    if 'spt' not in md:
        md['spt'] = {}
    return md['spt']


# ---------------------------------------------------------------------------
# SPTControlPanel
# ---------------------------------------------------------------------------

class SPTControlPanel(QtWidgets.QDockWidget):
    """Main SPT analysis control panel.

    Tabbed dock widget with 6 tabs:
    Tab 1 - Detection: method selector, parameters, "Detect" button
    Tab 2 - Linking: method selector, parameters, "Link" button
    Tab 3 - Analysis: feature checkboxes, "Compute Features" button
    Tab 4 - Classification: train/predict, classification results
    Tab 5 - Batch: file list, output dir, "Run Batch" button
    Tab 6 - Visualization: buttons to open each viz window

    Works on ``g.win`` (single window) or file list (batch mode).
    Results are stored in ``window.metadata['spt']`` dict.
    """

    # Singleton ---------------------------------------------------------

    _instance = None

    @classmethod
    def instance(cls, parent=None):
        """Return the singleton panel, creating it if necessary.

        If the previous instance was closed (not visible), a fresh one is
        created so that state is always clean.
        """
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent)
        return cls._instance

    # Signals -----------------------------------------------------------

    sigDetectionFinished = Signal(int)      # num_localizations
    sigLinkingFinished = Signal(int, float) # num_tracks, mean_length
    sigFeaturesComputed = Signal(int)       # num_features
    sigClassificationDone = Signal(dict)    # counts per class
    sigBatchProgress = Signal(int, int)     # current, total

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__("SPT Analysis", parent)
        self.setObjectName("SPTControlPanel")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(340)

        self._build_ui()
        self._connect_signals()

        logger.debug("SPTControlPanel created")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the tab widget and all six tab pages."""
        self.tabs = QtWidgets.QTabWidget()
        self.setWidget(self.tabs)

        self._build_detection_tab()
        self._build_linking_tab()
        self._build_analysis_tab()
        self._build_classification_tab()
        self._build_batch_tab()
        self._build_visualization_tab()

    # ---- Tab 1: Detection --------------------------------------------

    def _build_detection_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Method selector
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.det_method_combo = QtWidgets.QComboBox()
        self.det_method_combo.addItems(['U-Track', 'ThunderSTORM', 'AI Localizer'])
        method_layout.addWidget(self.det_method_combo)
        layout.addLayout(method_layout)

        layout.addWidget(_make_hline())

        # Parameters
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.det_psf_sigma = _make_double_spin(1.5, 0.5, 10.0, decimals=2,
                                                suffix=' px', step=0.1)
        form.addRow("PSF sigma:", self.det_psf_sigma)

        self.det_alpha = _make_double_spin(0.05, 0.001, 0.5, decimals=3,
                                            step=0.005)
        form.addRow("Significance (alpha):", self.det_alpha)

        self.det_min_intensity = _make_double_spin(0.0, 0.0, 10000.0,
                                                    decimals=1, step=10.0)
        form.addRow("Min intensity:", self.det_min_intensity)

        layout.addLayout(form)

        # U-Track advanced detection controls (hidden by default)
        from .spt_detection_controls import (ThunderSTORMControlGroup,
                                              UTrackDetectionControlGroup)
        self.utrack_det_controls = UTrackDetectionControlGroup()
        self.utrack_det_controls.setVisible(False)
        layout.addWidget(self.utrack_det_controls)

        # ThunderSTORM control group (hidden by default)
        self.ts_controls = ThunderSTORMControlGroup()
        self.ts_controls.setVisible(False)
        layout.addWidget(self.ts_controls)

        layout.addWidget(_make_hline())

        # Detect button
        self.det_button = QtWidgets.QPushButton("Detect")
        self.det_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        layout.addWidget(self.det_button)

        # Show/Hide Particles button
        self.det_show_particles_btn = QtWidgets.QPushButton("Show Particles")
        self.det_show_particles_btn.setCheckable(True)
        self.det_show_particles_btn.setToolTip(
            "Toggle display of detected particles on the image")
        layout.addWidget(self.det_show_particles_btn)

        # Show Results Table button
        self.det_results_table_btn = QtWidgets.QPushButton("Show Results Table")
        self.det_results_table_btn.setToolTip(
            "Open the SPT Results spreadsheet (sortable, filterable, "
            "exportable)")
        layout.addWidget(self.det_results_table_btn)

        # Progress bar
        self.det_progress = QtWidgets.QProgressBar()
        self.det_progress.setTextVisible(True)
        self.det_progress.setValue(0)
        layout.addWidget(self.det_progress)

        # Results
        self.det_result_label = QtWidgets.QLabel("No detection run yet.")
        self.det_result_label.setWordWrap(True)
        layout.addWidget(self.det_result_label)

        layout.addStretch()
        self.tabs.addTab(page, "Detection")

    # ---- Tab 2: Linking ----------------------------------------------

    def _build_linking_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Method selector
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.link_method_combo = QtWidgets.QComboBox()
        self.link_method_combo.addItems(['Greedy', 'U-Track LAP', 'Trackpy'])
        method_layout.addWidget(self.link_method_combo)
        layout.addLayout(method_layout)

        layout.addWidget(_make_hline())

        # Parameters
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.link_max_distance = _make_double_spin(5.0, 1.0, 50.0,
                                                    decimals=1, suffix=' px',
                                                    step=1.0)
        form.addRow("Max distance:", self.link_max_distance)

        self.link_max_gap = _make_int_spin(1, 0, 50, suffix=' frames')
        form.addRow("Max gap:", self.link_max_gap)

        self.link_min_length = _make_int_spin(3, 1, 100, suffix=' pts')
        form.addRow("Min track length:", self.link_min_length)

        layout.addLayout(form)

        # U-Track LAP control group (hidden by default)
        from .spt_detection_controls import UTrackLAPControlGroup, TrackpyControlGroup
        self.utrack_controls = UTrackLAPControlGroup()
        self.utrack_controls.setVisible(False)
        layout.addWidget(self.utrack_controls)

        # Trackpy control group (hidden by default)
        self.trackpy_controls = TrackpyControlGroup()
        self.trackpy_controls.setVisible(False)
        layout.addWidget(self.trackpy_controls)

        layout.addWidget(_make_hline())

        # Link button
        self.link_button = QtWidgets.QPushButton("Link")
        self.link_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        layout.addWidget(self.link_button)

        # Progress bar
        self.link_progress = QtWidgets.QProgressBar()
        self.link_progress.setTextVisible(True)
        self.link_progress.setValue(0)
        layout.addWidget(self.link_progress)

        # Results
        self.link_result_label = QtWidgets.QLabel("No linking run yet.")
        self.link_result_label.setWordWrap(True)
        layout.addWidget(self.link_result_label)

        layout.addStretch()
        self.tabs.addTab(page, "Linking")

    # ---- Tab 3: Analysis ---------------------------------------------

    def _build_analysis_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Physical parameters
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.ana_pixel_size = _make_double_spin(108.0, 1.0, 10000.0,
                                                 decimals=1, suffix=' nm',
                                                 step=1.0)
        form.addRow("Pixel size:", self.ana_pixel_size)

        self.ana_frame_interval = _make_double_spin(0.05, 0.0001, 100.0,
                                                     decimals=4, suffix=' s',
                                                     step=0.001)
        form.addRow("Frame interval:", self.ana_frame_interval)

        layout.addLayout(form)
        layout.addWidget(_make_hline())

        # Feature group checkboxes
        group_box = QtWidgets.QGroupBox("Feature Groups")
        group_layout = QtWidgets.QVBoxLayout(group_box)

        self.ana_geometric_cb = QtWidgets.QCheckBox("Geometric")
        self.ana_geometric_cb.setToolTip(
            "Radius of gyration, asymmetry, fractal dimension, "
            "net displacement, efficiency, straightness")
        self.ana_geometric_cb.setChecked(True)
        group_layout.addWidget(self.ana_geometric_cb)

        self.ana_kinematic_cb = QtWidgets.QCheckBox("Kinematic")
        self.ana_kinematic_cb.setToolTip(
            "Velocity, MSD, diffusion coefficient, direction, "
            "angular correlation")
        self.ana_kinematic_cb.setChecked(True)
        group_layout.addWidget(self.ana_kinematic_cb)

        self.ana_spatial_cb = QtWidgets.QCheckBox("Spatial")
        self.ana_spatial_cb.setToolTip(
            "Nearest-neighbor distances, local density, neighbor counts")
        self.ana_spatial_cb.setChecked(True)
        group_layout.addWidget(self.ana_spatial_cb)

        layout.addWidget(group_box)
        layout.addWidget(_make_hline())

        # Compute button
        self.ana_compute_button = QtWidgets.QPushButton("Compute Features")
        self.ana_compute_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        layout.addWidget(self.ana_compute_button)

        # Progress bar
        self.ana_progress = QtWidgets.QProgressBar()
        self.ana_progress.setTextVisible(True)
        self.ana_progress.setValue(0)
        layout.addWidget(self.ana_progress)

        # Results table
        self.ana_result_label = QtWidgets.QLabel("No features computed yet.")
        self.ana_result_label.setWordWrap(True)
        layout.addWidget(self.ana_result_label)

        self.ana_table = QtWidgets.QTableWidget(0, 5)
        self.ana_table.setHorizontalHeaderLabels(
            ["Feature", "Mean", "Std", "Min", "Max"])
        self.ana_table.horizontalHeader().setStretchLastSection(True)
        self.ana_table.setAlternatingRowColors(True)
        self.ana_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ana_table.setMinimumHeight(150)
        layout.addWidget(self.ana_table)

        self.tabs.addTab(page, "Analysis")

    # ---- Tab 4: Classification ---------------------------------------

    def _build_classification_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Info label
        info = QtWidgets.QLabel(
            "Classify tracks into motion modes (e.g. Mobile, Confined, "
            "Trapped) using a Support Vector Machine trained on computed "
            "features.")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addWidget(_make_hline())

        # Model controls
        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)

        self.cls_train_button = QtWidgets.QPushButton("Train SVM")
        self.cls_train_button.setToolTip(
            "Train a classifier on feature data with user-provided labels.")
        model_layout.addWidget(self.cls_train_button)

        self.cls_load_button = QtWidgets.QPushButton("Load Model")
        self.cls_load_button.setToolTip(
            "Load a previously saved .pkl model file.")
        model_layout.addWidget(self.cls_load_button)

        self.cls_save_button = QtWidgets.QPushButton("Save Model")
        self.cls_save_button.setToolTip(
            "Save the current trained model to a .pkl file.")
        self.cls_save_button.setEnabled(False)
        model_layout.addWidget(self.cls_save_button)

        self.cls_model_label = QtWidgets.QLabel("No model loaded.")
        self.cls_model_label.setWordWrap(True)
        model_layout.addWidget(self.cls_model_label)

        layout.addWidget(model_group)
        layout.addWidget(_make_hline())

        # Prediction
        self.cls_predict_button = QtWidgets.QPushButton("Predict")
        self.cls_predict_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        self.cls_predict_button.setToolTip(
            "Classify all tracks in the current window.")
        self.cls_predict_button.setEnabled(False)
        layout.addWidget(self.cls_predict_button)

        # Results
        self.cls_result_label = QtWidgets.QLabel("No classification yet.")
        self.cls_result_label.setWordWrap(True)
        layout.addWidget(self.cls_result_label)

        self.cls_results_group = QtWidgets.QGroupBox("Class Counts")
        self.cls_results_layout = QtWidgets.QFormLayout(self.cls_results_group)
        self._cls_count_labels = {}
        for cls_name in ('Mobile', 'Confined', 'Trapped', 'Other'):
            lbl = QtWidgets.QLabel("--")
            self.cls_results_layout.addRow(f"{cls_name}:", lbl)
            self._cls_count_labels[cls_name] = lbl
        layout.addWidget(self.cls_results_group)

        layout.addStretch()

        # Disable classification buttons if sklearn not available
        self._classifier_model = None
        self._check_sklearn_available()

        self.tabs.addTab(page, "Classification")

    # ---- Tab 5: Batch ------------------------------------------------

    def _build_batch_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # File list
        file_group = QtWidgets.QGroupBox("Input Files")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        self.batch_file_list = QtWidgets.QListWidget()
        self.batch_file_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.batch_file_list.setMinimumHeight(100)
        file_layout.addWidget(self.batch_file_list)

        btn_row = QtWidgets.QHBoxLayout()
        self.batch_add_btn = QtWidgets.QPushButton("Add...")
        self.batch_remove_btn = QtWidgets.QPushButton("Remove")
        self.batch_clear_btn = QtWidgets.QPushButton("Clear")
        btn_row.addWidget(self.batch_add_btn)
        btn_row.addWidget(self.batch_remove_btn)
        btn_row.addWidget(self.batch_clear_btn)
        file_layout.addLayout(btn_row)

        layout.addWidget(file_group)

        # Output directory
        out_layout = QtWidgets.QHBoxLayout()
        out_layout.addWidget(QtWidgets.QLabel("Output dir:"))
        self.batch_output_edit = QtWidgets.QLineEdit()
        self.batch_output_edit.setPlaceholderText("Same as input file")
        self.batch_output_edit.setReadOnly(True)
        out_layout.addWidget(self.batch_output_edit)
        self.batch_output_btn = QtWidgets.QPushButton("Browse...")
        out_layout.addWidget(self.batch_output_btn)
        layout.addLayout(out_layout)

        layout.addWidget(_make_hline())

        # Expert config
        config_layout = QtWidgets.QHBoxLayout()
        config_layout.addWidget(QtWidgets.QLabel("Expert config:"))
        self.batch_config_combo = QtWidgets.QComboBox()
        self.batch_config_combo.addItems([
            'Custom',
            'Fast Membrane Proteins',
            'Slow Confined Proteins',
            'Vesicle Trafficking',
            'Single Molecule',
        ])
        self.batch_config_combo.setToolTip(
            "Predefined parameter sets for common experiments.")
        config_layout.addWidget(self.batch_config_combo)
        layout.addLayout(config_layout)

        layout.addWidget(_make_hline())

        # Run button
        self.batch_run_button = QtWidgets.QPushButton("Run Batch")
        self.batch_run_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        layout.addWidget(self.batch_run_button)

        # Progress
        self.batch_progress = QtWidgets.QProgressBar()
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setValue(0)
        layout.addWidget(self.batch_progress)

        self.batch_status_label = QtWidgets.QLabel("Ready.")
        self.batch_status_label.setWordWrap(True)
        layout.addWidget(self.batch_status_label)

        layout.addStretch()
        self.tabs.addTab(page, "Batch")

    # ---- Tab 6: Visualization ----------------------------------------

    def _build_visualization_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        info = QtWidgets.QLabel(
            "Open specialized visualization windows for SPT data in the "
            "current image window.  Each button checks that the required "
            "data (localizations, tracks, features) is available first.")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addWidget(_make_hline())

        # Visualization buttons
        self.viz_track_overlay_btn = QtWidgets.QPushButton("Track Overlay")
        self.viz_track_overlay_btn.setToolTip(
            "Overlay linked tracks on the current image window.")
        layout.addWidget(self.viz_track_overlay_btn)

        self.viz_track_detail_btn = QtWidgets.QPushButton("Track Detail")
        self.viz_track_detail_btn.setToolTip(
            "Open a detail inspector for individual tracks.")
        layout.addWidget(self.viz_track_detail_btn)

        self.viz_all_tracks_btn = QtWidgets.QPushButton("All Tracks Plot")
        self.viz_all_tracks_btn.setToolTip(
            "Show all track paths in a single coordinate plot.")
        layout.addWidget(self.viz_all_tracks_btn)

        self.viz_diffusion_btn = QtWidgets.QPushButton("Diffusion Analysis")
        self.viz_diffusion_btn.setToolTip(
            "MSD curves and diffusion coefficient analysis.")
        layout.addWidget(self.viz_diffusion_btn)

        self.viz_flower_btn = QtWidgets.QPushButton("Flower Plot")
        self.viz_flower_btn.setToolTip(
            "Origin-centered track display (rose / flower plot).")
        layout.addWidget(self.viz_flower_btn)

        self.viz_chart_dock_btn = QtWidgets.QPushButton("Chart Dock")
        self.viz_chart_dock_btn.setToolTip(
            "General-purpose scatter / histogram dock for feature data.")
        layout.addWidget(self.viz_chart_dock_btn)

        layout.addStretch()
        self.tabs.addTab(page, "Visualization")

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Wire all button clicks and combo changes to their handlers."""
        # Detection
        self.det_button.clicked.connect(self._on_detect)
        self.det_method_combo.currentIndexChanged.connect(
            self._on_det_method_changed)
        self.det_show_particles_btn.toggled.connect(self._on_toggle_particles)
        self.det_results_table_btn.clicked.connect(self._on_show_results_table)
        self.utrack_det_controls.auto_psf_btn.clicked.connect(
            self._on_auto_estimate_psf)
        self.utrack_det_controls.auto_noise_btn.clicked.connect(
            self._on_auto_estimate_noise)

        # Linking
        self.link_button.clicked.connect(self._on_link)
        self.link_method_combo.currentIndexChanged.connect(
            self._on_link_method_changed)

        # Analysis
        self.ana_compute_button.clicked.connect(self._on_compute_features)

        # Classification
        self.cls_train_button.clicked.connect(self._on_train_classifier)
        self.cls_load_button.clicked.connect(self._on_load_model)
        self.cls_save_button.clicked.connect(self._on_save_model)
        self.cls_predict_button.clicked.connect(self._on_predict)

        # Batch
        self.batch_add_btn.clicked.connect(self._on_batch_add)
        self.batch_remove_btn.clicked.connect(self._on_batch_remove)
        self.batch_clear_btn.clicked.connect(self._on_batch_clear)
        self.batch_output_btn.clicked.connect(self._on_batch_browse_output)
        self.batch_run_button.clicked.connect(self._on_batch_run)
        self.batch_config_combo.currentTextChanged.connect(
            self._on_batch_config_changed)

        # Visualization
        self.viz_track_overlay_btn.clicked.connect(self._on_viz_track_overlay)
        self.viz_track_detail_btn.clicked.connect(self._on_viz_track_detail)
        self.viz_all_tracks_btn.clicked.connect(self._on_viz_all_tracks)
        self.viz_diffusion_btn.clicked.connect(self._on_viz_diffusion)
        self.viz_flower_btn.clicked.connect(self._on_viz_flower)
        self.viz_chart_dock_btn.clicked.connect(self._on_viz_chart_dock)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_window(self):
        """Return the current flika image window, or ``None``."""
        from .. import global_vars as g
        return getattr(g, 'win', None)

    def _require_window(self):
        """Return the current window, showing a warning if absent."""
        win = self._current_window()
        if win is None:
            QtWidgets.QMessageBox.warning(
                self, "No Window",
                "No image window is selected.  Open or select an image first.")
        return win

    def _require_spt_key(self, key, label=None):
        """Return ``window.metadata['spt'][key]`` or warn and return None."""
        win = self._require_window()
        if win is None:
            return None
        spt = _get_spt_data(win)
        if spt is None or key not in spt:
            QtWidgets.QMessageBox.warning(
                self, "Missing Data",
                f"No {label or key} data found.  "
                f"Run the appropriate analysis step first.")
            return None
        return spt[key]

    def _check_sklearn_available(self):
        """Disable classification buttons if scikit-learn is missing."""
        try:
            import sklearn  # noqa: F401
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
            self.cls_train_button.setEnabled(False)
            self.cls_load_button.setEnabled(False)
            self.cls_predict_button.setEnabled(False)
            self.cls_save_button.setEnabled(False)
            self.cls_model_label.setText(
                "scikit-learn not installed.  Classification is disabled.")
            logger.warning("scikit-learn not available; "
                           "SPT classification disabled")

    # ------------------------------------------------------------------
    # Tab 1 callbacks -- Detection
    # ------------------------------------------------------------------

    def _on_det_method_changed(self, index):
        """Update UI visibility when the detection method changes."""
        method = self.det_method_combo.currentText()
        is_ts = (method == 'ThunderSTORM')
        is_ai = (method == 'AI Localizer')
        is_utrack = (method == 'U-Track')

        # Show/hide method-specific controls
        self.ts_controls.setVisible(is_ts)
        self.utrack_det_controls.setVisible(is_utrack)

        # Show/hide U-Track-specific controls
        self.det_alpha.setEnabled(not is_ts and not is_ai)
        self.det_min_intensity.setEnabled(not is_ai)

        if is_ts:
            self.det_alpha.setToolTip("Not used by ThunderSTORM method.")
        elif is_ai:
            self.det_alpha.setToolTip("Not used by AI Localizer.")
        else:
            self.det_alpha.setToolTip("")

    def _on_detect(self):
        """Run particle detection on the current window."""
        win = self._require_window()
        if win is None:
            return

        method = self.det_method_combo.currentText()
        psf_sigma = self.det_psf_sigma.value()
        alpha = self.det_alpha.value()
        min_intensity = self.det_min_intensity.value()

        spt = _get_spt_data(win)
        if spt is None:
            return

        self.det_button.setEnabled(False)
        self.det_result_label.setText("Detecting...")
        self.det_progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            image_data = np.asarray(win.image, dtype=np.float64)
            if image_data.ndim == 2:
                # Single frame -- wrap in a 3D array
                image_data = image_data[np.newaxis]

            n_frames = image_data.shape[0]
            all_locs = []

            if method == 'U-Track':
                from ..spt.detection.utrack_detector import UTrackDetector
                adv = self.utrack_det_controls.get_params()
                detector = UTrackDetector(
                    psf_sigma=psf_sigma,
                    alpha=alpha,
                    min_intensity=min_intensity,
                    **adv)

                for f in range(n_frames):
                    frame_locs = detector.detect_frame(image_data[f])
                    if len(frame_locs) > 0:
                        # Prepend frame index column
                        frame_col = np.full((len(frame_locs), 1), f,
                                            dtype=np.float64)
                        frame_locs = np.hstack([frame_col, frame_locs])
                        all_locs.append(frame_locs)

                    pct = int(100 * (f + 1) / n_frames)
                    self.det_progress.setValue(pct)
                    if f % 10 == 0:
                        QtWidgets.QApplication.processEvents()

            elif method == 'ThunderSTORM':
                from ..spt.detection.thunderstorm import ThunderSTORMDetector
                ts_params = self.ts_controls.get_params()
                detector = ThunderSTORMDetector(**ts_params)

                # detect_stack returns (M, 8): [frame, x, y, intensity,
                #   sigma_x, sigma_y, background, uncertainty]
                def _progress_cb(f):
                    pct = int(100 * (f + 1) / n_frames)
                    self.det_progress.setValue(pct)
                    if f % 10 == 0:
                        QtWidgets.QApplication.processEvents()

                stack_locs = detector.detect_stack(image_data,
                                                    callback=_progress_cb)
                if len(stack_locs) > 0:
                    all_locs.append(stack_locs)

            elif method == 'AI Localizer':
                # Attempt to use AI localizer; fall back if unavailable
                try:
                    from ..ai.particle_localizer import ParticleLocalizer
                    localizer = ParticleLocalizer()
                    for f in range(n_frames):
                        frame_locs = localizer.predict(image_data[f])
                        if frame_locs is not None and len(frame_locs) > 0:
                            frame_col = np.full((len(frame_locs), 1), f,
                                                dtype=np.float64)
                            frame_locs = np.hstack([frame_col, frame_locs])
                            all_locs.append(frame_locs)

                        pct = int(100 * (f + 1) / n_frames)
                        self.det_progress.setValue(pct)
                        if f % 10 == 0:
                            QtWidgets.QApplication.processEvents()
                except ImportError:
                    QtWidgets.QMessageBox.warning(
                        self, "AI Localizer",
                        "AI Localizer module not available.  "
                        "Install the required dependencies or choose "
                        "a different detection method.")
                    return

            # Concatenate results
            if all_locs:
                localizations = np.vstack(all_locs)
            else:
                localizations = np.empty((0, 4))

            # Normalize coordinates: detectors output x=col(dim2),
            # y=row(dim1) assuming row-major images, but flika stores
            # images as (dim1, dim2) = (cols, rows).  Swap columns 1,2
            # so that localizations[:,1] = dim1 and [:,2] = dim2,
            # matching pyqtgraph's display convention.
            if len(localizations) > 0:
                localizations[:, 1], localizations[:, 2] = (
                    localizations[:, 2].copy(), localizations[:, 1].copy())

            # Build ParticleData model
            from ..spt.particle_data import ParticleData
            det_params = {
                'method': method,
                'psf_sigma': psf_sigma,
                'alpha': alpha,
                'min_intensity': min_intensity,
            }
            if method == 'ThunderSTORM':
                det_params.update(self.ts_controls.get_params())

            pdata = ParticleData.from_numpy(localizations)
            pdata._detection_params = det_params

            # Store ParticleData and sync legacy keys
            spt['particle_data'] = pdata
            legacy = pdata.to_spt_dict()
            spt['localizations'] = legacy['localizations']
            spt['detection_method'] = method
            spt['detection_params'] = det_params

            n_total = len(localizations)
            per_frame = n_total / max(n_frames, 1)
            self.det_result_label.setText(
                f"Detected {n_total} particles across {n_frames} frames "
                f"({per_frame:.1f} per frame).")
            self.det_progress.setValue(100)
            self.sigDetectionFinished.emit(n_total)
            logger.info("Detection complete: %d localizations (%s)",
                        n_total, method)

            # Auto-show particles and update results table
            if n_total > 0:
                self._draw_scatter_points(win, localizations)
                self.det_show_particles_btn.blockSignals(True)
                self.det_show_particles_btn.setChecked(True)
                self.det_show_particles_btn.setText("Hide Particles")
                self.det_show_particles_btn.blockSignals(False)
            self._update_results_table(spt)

        except Exception as exc:
            logger.exception("Detection failed: %s", exc)
            self.det_result_label.setText(f"Detection failed: {exc}")
            self.det_progress.setValue(0)
        finally:
            self.det_button.setEnabled(True)

    def _on_auto_estimate_psf(self):
        """Estimate PSF sigma from the current image autocorrelation."""
        win = self._current_window()
        if win is None:
            self.det_result_label.setText("No window selected.")
            return
        try:
            image = np.asarray(win.image, dtype=np.float64)
            if image.ndim == 3:
                # Use the middle frame
                image = image[image.shape[0] // 2]
            from ..spt.detection.utrack_detector import UTrackDetector
            sigma = UTrackDetector.auto_estimate_psf_sigma(image)
            self.det_psf_sigma.setValue(sigma)
            self.det_result_label.setText(
                f"Auto-estimated PSF sigma: {sigma:.2f} px")
        except Exception as exc:
            self.det_result_label.setText(f"PSF estimation failed: {exc}")

    def _on_auto_estimate_noise(self):
        """Estimate image noise from the Laplacian MAD."""
        win = self._current_window()
        if win is None:
            self.det_result_label.setText("No window selected.")
            return
        try:
            image = np.asarray(win.image, dtype=np.float64)
            if image.ndim == 3:
                image = image[image.shape[0] // 2]
            from ..spt.detection.utrack_detector import UTrackDetector
            noise = UTrackDetector.auto_estimate_noise(image)
            self.det_result_label.setText(
                f"Auto-estimated noise: {noise:.2f}")
        except Exception as exc:
            self.det_result_label.setText(f"Noise estimation failed: {exc}")

    def _on_toggle_particles(self, checked):
        """Show or hide detected particle scatter points on the image."""
        win = self._current_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None:
            return

        if checked:
            # Show particles
            locs = spt.get('localizations')
            if locs is None or len(locs) == 0:
                self.det_show_particles_btn.setChecked(False)
                return
            self._draw_scatter_points(win, locs)
            self.det_show_particles_btn.setText("Hide Particles")
        else:
            # Hide particles
            self._clear_scatter_points(win)
            self.det_show_particles_btn.setText("Show Particles")

    def _draw_scatter_points(self, window, locs):
        """Draw detection scatter points on a window.

        Localizations are stored with [:,1] = dim1 (pyqtgraph x) and
        [:,2] = dim2 (pyqtgraph y), matching the display convention.
        """
        from qtpy.QtGui import QColor
        if not hasattr(window, 'scatterPoints'):
            return
        # Clear existing
        for frame_pts in window.scatterPoints:
            frame_pts.clear()

        color = QColor(0, 255, 0, 180)  # green
        for det in locs:
            frame = int(det[0])
            if 0 <= frame < len(window.scatterPoints):
                window.scatterPoints[frame].append([det[1], det[2], color, 5])

        if hasattr(window, 'updateindex'):
            window.updateindex()

    def _clear_scatter_points(self, window):
        """Clear all detection scatter points from a window."""
        if not hasattr(window, 'scatterPoints'):
            return
        for frame_pts in window.scatterPoints:
            frame_pts.clear()
        if hasattr(window, 'updateindex'):
            window.updateindex()

    def _on_show_results_table(self):
        """Open (or bring to front) the SPT Results Table dock widget."""
        from .. import global_vars as g
        win = self._current_window()
        spt = _get_spt_data(win) if win else None

        from .results_table import ResultsTableWidget
        table = ResultsTableWidget.instance(g.m)
        from qtpy.QtCore import Qt as _Qt
        g.m.addDockWidget(_Qt.DockWidgetArea.BottomDockWidgetArea, table)

        if spt is not None:
            pdata = spt.get('particle_data')
            if pdata is not None:
                table.set_particle_data(pdata)
            elif 'localizations' in spt:
                import pandas as pd
                locs = spt['localizations']
                if hasattr(locs, 'shape') and len(locs) > 0:
                    n_cols = locs.shape[1] if locs.ndim > 1 else 1
                    if n_cols == 4:
                        cols = ['frame', 'x', 'y', 'intensity']
                    elif n_cols == 5:
                        cols = ['frame', 'x', 'y', 'intensity', 'track_id']
                    else:
                        cols = [f'col_{i}' for i in range(n_cols)]
                    table.set_dataframe(pd.DataFrame(locs, columns=cols))

        table.show()
        table.raise_()

    def _update_results_table(self, spt):
        """Refresh the results table if it is currently open."""
        from .results_table import ResultsTableWidget
        table = ResultsTableWidget._instance
        if table is None or not table.isVisible():
            return
        pdata = spt.get('particle_data')
        if pdata is not None:
            table.set_particle_data(pdata)

    # ------------------------------------------------------------------
    # Tab 2 callbacks -- Linking
    # ------------------------------------------------------------------

    def _on_link_method_changed(self, index):
        """Update UI visibility when the linking method changes."""
        method = self.link_method_combo.currentText()

        # Show/hide method-specific control groups
        self.utrack_controls.setVisible(method == 'U-Track LAP')
        self.trackpy_controls.setVisible(method == 'Trackpy')

        if method == 'Trackpy':
            self.link_max_gap.setToolTip("'memory' parameter in trackpy")
        else:
            self.link_max_gap.setToolTip("")

    def _on_link(self):
        """Run particle linking on detected localizations."""
        win = self._require_window()
        if win is None:
            return

        locs = self._require_spt_key('localizations', 'localization')
        if locs is None:
            return
        if len(locs) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Localizations",
                "No localizations found.  Run detection first.")
            return

        method = self.link_method_combo.currentText()
        max_distance = self.link_max_distance.value()
        max_gap = self.link_max_gap.value()
        min_length = self.link_min_length.value()

        spt = _get_spt_data(win)
        self.link_button.setEnabled(False)
        self.link_result_label.setText("Linking...")
        self.link_progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            if method == 'Greedy':
                from ..spt.linking.greedy_linker import link_particles
                tracks, stats = link_particles(
                    locs, max_distance=max_distance,
                    max_gap=max_gap, min_track_length=min_length)
                self.link_progress.setValue(100)

            elif method == 'U-Track LAP':
                from ..spt.linking.utrack_linker import (
                    UTrackLinker, TrackingConfig)
                utrack_config = self.utrack_controls.get_config()
                config = TrackingConfig(
                    max_distance=max_distance,
                    max_gap=max_gap,
                    min_track_length=min_length,
                    **utrack_config)
                linker = UTrackLinker(config=config)
                tracks, stats = linker.link(locs)
                self.link_progress.setValue(100)

            elif method == 'Trackpy':
                try:
                    from ..spt.linking.trackpy_linker import link_with_trackpy
                    tp_params = self.trackpy_controls.get_params()
                    tracks, stats = link_with_trackpy(
                        locs, search_range=max_distance,
                        memory=max_gap, min_track_length=min_length,
                        **tp_params)
                    self.link_progress.setValue(100)
                except ImportError:
                    QtWidgets.QMessageBox.warning(
                        self, "Trackpy Not Installed",
                        "The 'trackpy' package is not installed.  "
                        "Install it with: pip install trackpy")
                    return

            # Build per-localization track_id column
            track_ids = np.full(len(locs), -1, dtype=np.int64)
            for tid, track_indices in enumerate(tracks):
                for idx in track_indices:
                    track_ids[idx] = tid

            # Build localizations array with track_id appended
            # Columns: [frame, x, y, intensity, track_id]
            if locs.shape[1] == 4:
                linked_locs = np.column_stack([locs, track_ids])
            else:
                # Ensure at least frame, x, y, then track_id
                linked_locs = np.column_stack([locs, track_ids])

            spt['localizations'] = linked_locs
            spt['tracks'] = tracks
            spt['linking_method'] = method
            linking_params = {
                'method': method,
                'max_distance': max_distance,
                'max_gap': max_gap,
                'min_track_length': min_length,
            }
            if method == 'U-Track LAP':
                linking_params.update(self.utrack_controls.get_config())
            elif method == 'Trackpy':
                linking_params.update(self.trackpy_controls.get_params())
            spt['linking_params'] = linking_params

            # Update ParticleData if it exists
            pdata = spt.get('particle_data')
            if pdata is not None:
                pdata.set_tracks(tracks, linking_params=linking_params)
                # Sync legacy keys
                legacy = pdata.to_spt_dict()
                spt['tracks_dict'] = legacy.get('tracks_dict', {})

            n_tracks = len(tracks)
            if n_tracks > 0:
                lengths = [len(t) for t in tracks]
                mean_len = np.mean(lengths)
                median_len = np.median(lengths)
                max_len = np.max(lengths)
            else:
                mean_len = median_len = max_len = 0

            self.link_result_label.setText(
                f"{n_tracks} tracks linked.  "
                f"Mean length: {mean_len:.1f}, "
                f"median: {median_len:.0f}, "
                f"max: {max_len:.0f} points.")
            self.sigLinkingFinished.emit(n_tracks, float(mean_len))
            logger.info("Linking complete: %d tracks (%s)", n_tracks, method)

            # Refresh results table if open
            self._update_results_table(spt)

        except Exception as exc:
            logger.exception("Linking failed: %s", exc)
            self.link_result_label.setText(f"Linking failed: {exc}")
            self.link_progress.setValue(0)
        finally:
            self.link_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Tab 3 callbacks -- Analysis
    # ------------------------------------------------------------------

    def _on_compute_features(self):
        """Compute track features for all linked tracks."""
        win = self._require_window()
        if win is None:
            return

        tracks = self._require_spt_key('tracks', 'track')
        if tracks is None:
            return

        spt = _get_spt_data(win)
        locs = spt.get('localizations')
        if locs is None or len(locs) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No localization data found.")
            return

        pixel_size = self.ana_pixel_size.value()
        frame_interval = self.ana_frame_interval.value()
        enable_geometric = self.ana_geometric_cb.isChecked()
        enable_kinematic = self.ana_kinematic_cb.isChecked()
        enable_spatial = self.ana_spatial_cb.isChecked()

        self.ana_compute_button.setEnabled(False)
        self.ana_result_label.setText("Computing features...")
        self.ana_progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            from ..spt.features.feature_calculator import FeatureCalculator
            calc = FeatureCalculator(
                pixel_size=pixel_size,
                frame_interval=frame_interval,
                enable_geometric=enable_geometric,
                enable_kinematic=enable_kinematic,
                enable_spatial=enable_spatial)

            features_by_track = {}
            n_tracks = len(tracks)
            for i, track_indices in enumerate(tracks):
                positions = locs[track_indices, 1:3]  # x, y columns
                try:
                    feat = calc.compute_track_features(positions)
                    features_by_track[i] = feat
                except Exception as exc:
                    logger.warning("Feature computation failed for track %d: "
                                   "%s", i, exc)
                    features_by_track[i] = {'n_points': len(positions)}

                pct = int(100 * (i + 1) / max(n_tracks, 1))
                self.ana_progress.setValue(pct)
                if i % 20 == 0:
                    QtWidgets.QApplication.processEvents()

            spt['features_by_track'] = features_by_track
            spt['feature_params'] = {
                'pixel_size': pixel_size,
                'frame_interval': frame_interval,
                'geometric': enable_geometric,
                'kinematic': enable_kinematic,
                'spatial': enable_spatial,
            }

            # Build summary statistics table
            self._populate_feature_table(features_by_track)

            n_feat = len(self._get_feature_names(features_by_track))
            self.ana_result_label.setText(
                f"Computed {n_feat} features for {len(features_by_track)} "
                f"tracks.")
            self.ana_progress.setValue(100)
            self.sigFeaturesComputed.emit(n_feat)
            logger.info("Feature computation complete: %d features, "
                        "%d tracks", n_feat, len(features_by_track))

            # Update ParticleData with features and refresh results table
            pdata = spt.get('particle_data')
            if pdata is not None:
                pdata.set_features(features_by_track)
                self._update_results_table(spt)

        except Exception as exc:
            logger.exception("Feature computation failed: %s", exc)
            self.ana_result_label.setText(f"Feature computation failed: {exc}")
            self.ana_progress.setValue(0)
        finally:
            self.ana_compute_button.setEnabled(True)

    def _get_feature_names(self, features_by_track):
        """Collect all feature names across tracks."""
        names = set()
        for feat_dict in features_by_track.values():
            names.update(feat_dict.keys())
        names.discard('n_points')
        return sorted(names)

    def _populate_feature_table(self, features_by_track):
        """Fill the QTableWidget with summary statistics."""
        feature_names = self._get_feature_names(features_by_track)
        self.ana_table.setRowCount(len(feature_names))

        for row, name in enumerate(feature_names):
            values = []
            for feat_dict in features_by_track.values():
                val = feat_dict.get(name)
                if val is not None and np.isfinite(val):
                    values.append(val)

            if values:
                arr = np.array(values)
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                min_val = np.min(arr)
                max_val = np.max(arr)
            else:
                mean_val = std_val = min_val = max_val = float('nan')

            items = [
                QtWidgets.QTableWidgetItem(name),
                QtWidgets.QTableWidgetItem(f"{mean_val:.4g}"),
                QtWidgets.QTableWidgetItem(f"{std_val:.4g}"),
                QtWidgets.QTableWidgetItem(f"{min_val:.4g}"),
                QtWidgets.QTableWidgetItem(f"{max_val:.4g}"),
            ]
            for col, item in enumerate(items):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.ana_table.setItem(row, col, item)

        self.ana_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Tab 4 callbacks -- Classification
    # ------------------------------------------------------------------

    def _on_train_classifier(self):
        """Train an SVM classifier on labeled feature data."""
        if not self._sklearn_available:
            return

        features_by_track = self._require_spt_key(
            'features_by_track', 'feature')
        if features_by_track is None:
            return

        # Ask user for label CSV file
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Track Labels", "",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            import pandas as pd
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            # Expect CSV with columns: track_id, label
            labels_df = pd.read_csv(path)
            if 'track_id' not in labels_df.columns or \
               'label' not in labels_df.columns:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Labels",
                    "Label CSV must have 'track_id' and 'label' columns.")
                return

            # Build feature matrix for labeled tracks
            feature_names = self._get_feature_names(features_by_track)
            X_rows = []
            y_labels = []
            for _, row in labels_df.iterrows():
                tid = int(row['track_id'])
                if tid in features_by_track:
                    feat = features_by_track[tid]
                    fvec = [feat.get(fn, 0.0) for fn in feature_names]
                    if all(np.isfinite(v) for v in fvec):
                        X_rows.append(fvec)
                        y_labels.append(str(row['label']))

            if len(X_rows) < 2:
                QtWidgets.QMessageBox.warning(
                    self, "Insufficient Data",
                    "Need at least 2 labeled tracks with valid features "
                    "to train a classifier.")
                return

            X = np.array(X_rows)
            y = np.array(y_labels)

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True)),
            ])
            model.fit(X, y)

            self._classifier_model = model
            self._classifier_feature_names = feature_names
            classes = sorted(set(y))

            self.cls_model_label.setText(
                f"SVM trained on {len(X)} tracks, "
                f"{len(classes)} classes: {', '.join(classes)}")
            self.cls_predict_button.setEnabled(True)
            self.cls_save_button.setEnabled(True)
            logger.info("SVM classifier trained: %d tracks, %d classes",
                        len(X), len(classes))

        except Exception as exc:
            logger.exception("Classifier training failed: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Training Error",
                f"Classifier training failed:\n{exc}")

    def _on_load_model(self):
        """Load a previously saved classifier model."""
        if not self._sklearn_available:
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Classifier Model", "",
            "Pickle Files (*.pkl);;All Files (*)")
        if not path:
            return

        try:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self._classifier_model = data['model']
            self._classifier_feature_names = data['feature_names']

            classes = list(self._classifier_model.classes_)
            self.cls_model_label.setText(
                f"Model loaded from {path}\n"
                f"Classes: {', '.join(str(c) for c in classes)}")
            self.cls_predict_button.setEnabled(True)
            self.cls_save_button.setEnabled(True)
            logger.info("Loaded classifier model from %s", path)

        except Exception as exc:
            logger.exception("Model loading failed: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Failed to load model:\n{exc}")

    def _on_save_model(self):
        """Save the current classifier model to a file."""
        if self._classifier_model is None:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Classifier Model", "spt_classifier.pkl",
            "Pickle Files (*.pkl);;All Files (*)")
        if not path:
            return

        try:
            import pickle
            data = {
                'model': self._classifier_model,
                'feature_names': self._classifier_feature_names,
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Saved classifier model to %s", path)
        except Exception as exc:
            logger.exception("Model saving failed: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Save Error",
                f"Failed to save model:\n{exc}")

    def _on_predict(self):
        """Classify all tracks in the current window."""
        if self._classifier_model is None:
            return

        features_by_track = self._require_spt_key(
            'features_by_track', 'feature')
        if features_by_track is None:
            return

        win = self._current_window()
        spt = _get_spt_data(win)

        try:
            feature_names = self._classifier_feature_names

            # Build feature matrix for all tracks
            track_ids = []
            X_rows = []
            for tid, feat in features_by_track.items():
                fvec = [feat.get(fn, 0.0) for fn in feature_names]
                if all(np.isfinite(v) for v in fvec):
                    track_ids.append(tid)
                    X_rows.append(fvec)

            if len(X_rows) == 0:
                self.cls_result_label.setText(
                    "No tracks with valid features to classify.")
                return

            X = np.array(X_rows)
            predictions = self._classifier_model.predict(X)

            # Store predictions
            classification = {}
            for tid, pred in zip(track_ids, predictions):
                classification[tid] = str(pred)

            spt['classification'] = classification

            # Count per class
            counts = {}
            for label in predictions:
                label = str(label)
                counts[label] = counts.get(label, 0) + 1

            # Update UI
            for cls_name, lbl in self._cls_count_labels.items():
                lbl.setText(str(counts.get(cls_name, 0)))

            # Add any unexpected class names
            for cls_name, count in sorted(counts.items()):
                if cls_name not in self._cls_count_labels:
                    lbl = QtWidgets.QLabel(str(count))
                    self.cls_results_layout.addRow(f"{cls_name}:", lbl)
                    self._cls_count_labels[cls_name] = lbl

            total_classified = len(predictions)
            total_tracks = len(features_by_track)
            self.cls_result_label.setText(
                f"Classified {total_classified} of {total_tracks} tracks.")
            self.sigClassificationDone.emit(counts)
            logger.info("Classification complete: %d tracks, %s",
                        total_classified, counts)

        except Exception as exc:
            logger.exception("Classification failed: %s", exc)
            self.cls_result_label.setText(f"Classification failed: {exc}")

    # ------------------------------------------------------------------
    # Tab 5 callbacks -- Batch
    # ------------------------------------------------------------------

    def _on_batch_add(self):
        """Add files to the batch list."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add Files for Batch Processing", "",
            "Image Files (*.tif *.tiff *.h5 *.hdf5 *.npy *.zarr);;"
            "All Files (*)")
        for p in paths:
            # Avoid duplicates
            existing = [self.batch_file_list.item(i).text()
                        for i in range(self.batch_file_list.count())]
            if p not in existing:
                self.batch_file_list.addItem(p)

    def _on_batch_remove(self):
        """Remove selected files from the batch list."""
        for item in self.batch_file_list.selectedItems():
            row = self.batch_file_list.row(item)
            self.batch_file_list.takeItem(row)

    def _on_batch_clear(self):
        """Clear all files from the batch list."""
        self.batch_file_list.clear()

    def _on_batch_browse_output(self):
        """Browse for the batch output directory."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if path:
            self.batch_output_edit.setText(path)

    def _on_batch_config_changed(self, config_name):
        """Apply expert configuration presets to detection/linking params."""
        presets = {
            'Fast Membrane Proteins': {
                'psf_sigma': 1.5, 'alpha': 0.01, 'min_intensity': 50,
                'max_distance': 8.0, 'max_gap': 2, 'min_length': 5,
                'pixel_size': 108.0, 'frame_interval': 0.033,
            },
            'Slow Confined Proteins': {
                'psf_sigma': 1.8, 'alpha': 0.05, 'min_intensity': 30,
                'max_distance': 3.0, 'max_gap': 1, 'min_length': 10,
                'pixel_size': 108.0, 'frame_interval': 0.05,
            },
            'Vesicle Trafficking': {
                'psf_sigma': 2.5, 'alpha': 0.05, 'min_intensity': 100,
                'max_distance': 15.0, 'max_gap': 3, 'min_length': 5,
                'pixel_size': 108.0, 'frame_interval': 0.1,
            },
            'Single Molecule': {
                'psf_sigma': 1.2, 'alpha': 0.001, 'min_intensity': 20,
                'max_distance': 5.0, 'max_gap': 0, 'min_length': 3,
                'pixel_size': 160.0, 'frame_interval': 0.02,
            },
        }

        if config_name == 'Custom' or config_name not in presets:
            return

        p = presets[config_name]
        self.det_psf_sigma.setValue(p['psf_sigma'])
        self.det_alpha.setValue(p['alpha'])
        self.det_min_intensity.setValue(p['min_intensity'])
        self.link_max_distance.setValue(p['max_distance'])
        self.link_max_gap.setValue(p['max_gap'])
        self.link_min_length.setValue(p['min_length'])
        self.ana_pixel_size.setValue(p['pixel_size'])
        self.ana_frame_interval.setValue(p['frame_interval'])

        logger.info("Applied batch config preset: %s", config_name)

    def _on_batch_run(self):
        """Run the full SPT pipeline on all files in the batch list."""
        from .. import global_vars as g

        n_files = self.batch_file_list.count()
        if n_files == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Files",
                "Add files to the batch list first.")
            return

        output_dir = self.batch_output_edit.text().strip() or None

        self.batch_run_button.setEnabled(False)
        self.batch_progress.setMaximum(n_files)
        self.batch_progress.setValue(0)
        self.batch_status_label.setText("Starting batch processing...")
        QtWidgets.QApplication.processEvents()

        results = []
        for i in range(n_files):
            file_path = self.batch_file_list.item(i).text()
            self.batch_status_label.setText(
                f"Processing file {i + 1}/{n_files}: "
                f"{file_path.split('/')[-1]}")
            QtWidgets.QApplication.processEvents()

            try:
                result = self._run_single_file(file_path, output_dir)
                results.append(result)
                self.batch_file_list.item(i).setForeground(
                    QtGui.QBrush(QtGui.QColor(0, 128, 0)))
            except Exception as exc:
                logger.exception("Batch processing failed for %s: %s",
                                 file_path, exc)
                results.append({'file': file_path, 'error': str(exc)})
                self.batch_file_list.item(i).setForeground(
                    QtGui.QBrush(QtGui.QColor(200, 0, 0)))

            self.batch_progress.setValue(i + 1)
            self.sigBatchProgress.emit(i + 1, n_files)
            QtWidgets.QApplication.processEvents()

        # Summary
        n_success = sum(1 for r in results if 'error' not in r)
        n_fail = len(results) - n_success
        self.batch_status_label.setText(
            f"Batch complete: {n_success} succeeded, {n_fail} failed "
            f"out of {n_files} files.")
        self.batch_run_button.setEnabled(True)
        logger.info("Batch processing complete: %d/%d succeeded",
                     n_success, n_files)

    def _run_single_file(self, file_path, output_dir):
        """Run detection + linking + features on a single file.

        Returns a dict with track count and file path info.
        """
        import os
        from ..spt.detection.utrack_detector import UTrackDetector
        from ..spt.linking.greedy_linker import link_particles
        from ..spt.features.feature_calculator import FeatureCalculator

        # Load image data
        from ..io.registry import FormatRegistry
        registry = FormatRegistry.instance()
        image_data = registry.load(file_path)
        if image_data is None:
            raise RuntimeError(f"Failed to load: {file_path}")

        image_data = np.asarray(image_data, dtype=np.float64)
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis]

        # Detection
        adv_det = self.utrack_det_controls.get_params()
        detector = UTrackDetector(
            psf_sigma=self.det_psf_sigma.value(),
            alpha=self.det_alpha.value(),
            min_intensity=self.det_min_intensity.value(),
            **adv_det)

        all_locs = []
        for f in range(image_data.shape[0]):
            frame_locs = detector.detect_frame(image_data[f])
            if len(frame_locs) > 0:
                frame_col = np.full((len(frame_locs), 1), f, dtype=np.float64)
                frame_locs = np.hstack([frame_col, frame_locs])
                all_locs.append(frame_locs)

        if all_locs:
            locs = np.vstack(all_locs)
        else:
            locs = np.empty((0, 4))

        # Linking
        tracks, stats = link_particles(
            locs,
            max_distance=self.link_max_distance.value(),
            max_gap=self.link_max_gap.value(),
            min_track_length=self.link_min_length.value())

        # Features
        calc = FeatureCalculator(
            pixel_size=self.ana_pixel_size.value(),
            frame_interval=self.ana_frame_interval.value(),
            enable_geometric=self.ana_geometric_cb.isChecked(),
            enable_kinematic=self.ana_kinematic_cb.isChecked(),
            enable_spatial=self.ana_spatial_cb.isChecked())

        features_by_track = {}
        for i, track_indices in enumerate(tracks):
            positions = locs[track_indices, 1:3]
            try:
                features_by_track[i] = calc.compute_track_features(positions)
            except Exception:
                features_by_track[i] = {'n_points': len(positions)}

        # Save results
        if output_dir is None:
            output_dir = os.path.dirname(file_path)

        base = os.path.splitext(os.path.basename(file_path))[0]

        # Save localizations
        loc_path = os.path.join(output_dir, f"{base}_localizations.csv")
        header = "frame,x,y,intensity"
        if locs.shape[1] > 4:
            header += ",track_id"
        np.savetxt(loc_path, locs, delimiter=',', header=header,
                   comments='')

        # Save track features
        if features_by_track:
            import csv
            feat_path = os.path.join(output_dir, f"{base}_features.csv")
            all_names = self._get_feature_names(features_by_track)
            with open(feat_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=['track_id', 'n_points'] + all_names)
                writer.writeheader()
                for tid, feat in sorted(features_by_track.items()):
                    row = {'track_id': tid}
                    row.update(feat)
                    writer.writerow(row)

        return {
            'file': file_path,
            'n_localizations': len(locs),
            'n_tracks': len(tracks),
            'n_features': len(self._get_feature_names(features_by_track)),
        }

    # ------------------------------------------------------------------
    # Tab 6 callbacks -- Visualization
    # ------------------------------------------------------------------

    def _on_viz_track_overlay(self):
        """Show track overlay on the current image window."""
        from .. import global_vars as g

        win = self._require_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None or 'tracks' not in spt:
            QtWidgets.QMessageBox.warning(
                self, "No Tracks",
                "No track data found.  Run detection and linking first.")
            return

        try:
            from .track_overlay import show_track_overlay
            overlay, panel = show_track_overlay(win)

            # Load tracks from tracks_dict if available, else build it
            tracks_dict = spt.get('tracks_dict')
            if tracks_dict is None:
                from ..spt.linking.greedy_linker import tracks_to_dict
                locs = spt.get('localizations')
                tracks = spt.get('tracks')
                if locs is not None and tracks is not None:
                    tracks_dict = tracks_to_dict(locs, tracks)
                    spt['tracks_dict'] = tracks_dict

            if tracks_dict:
                overlay.load_tracks_from_dict(tracks_dict)

            logger.info("Track overlay displayed on %s", win.name)
        except Exception as exc:
            logger.exception("Failed to show track overlay: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Overlay Error",
                f"Failed to display track overlay:\n{exc}")

    def _on_viz_track_detail(self):
        """Open the track detail inspector window."""
        from .. import global_vars as g

        win = self._require_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None or 'tracks' not in spt:
            QtWidgets.QMessageBox.warning(
                self, "No Tracks",
                "No track data found.  Run detection and linking first.")
            return

        try:
            from .track_window import TrackDetailWindow
            detail_win = TrackDetailWindow(
                source_window=win, parent=g.m)
            detail_win.set_data(
                spt['localizations'], spt['tracks'],
                features_dict=spt.get('features_by_track'))
            detail_win.show()
            logger.info("Track detail window opened")
        except ImportError:
            logger.warning("TrackDetailWindow not available")
            QtWidgets.QMessageBox.information(
                self, "Not Available",
                "Track detail viewer is not yet available.")
        except Exception as exc:
            logger.exception("Failed to open track detail: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"Failed to open track detail window:\n{exc}")

    def _on_viz_all_tracks(self):
        """Open the all-tracks plot window."""
        from .. import global_vars as g

        win = self._require_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None or 'tracks' not in spt:
            QtWidgets.QMessageBox.warning(
                self, "No Tracks",
                "No track data found.  Run detection and linking first.")
            return

        try:
            from .all_tracks_plot import AllTracksPlotWindow
            plot_win = AllTracksPlotWindow(parent=g.m)

            # Build tracks_dict if not present
            tracks_dict = spt.get('tracks_dict')
            if tracks_dict is None:
                from ..spt.linking.greedy_linker import tracks_to_dict
                tracks_dict = tracks_to_dict(
                    spt['localizations'], spt['tracks'])
                spt['tracks_dict'] = tracks_dict

            plot_win.set_data(win, tracks_dict)
            plot_win.show()
            logger.info("All tracks plot window opened")
        except ImportError:
            logger.warning("AllTracksPlotWindow not available")
            QtWidgets.QMessageBox.information(
                self, "Not Available",
                "All tracks plot is not yet available.")
        except Exception as exc:
            logger.exception("Failed to open all tracks plot: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"Failed to open all tracks plot:\n{exc}")

    def _on_viz_diffusion(self):
        """Open the diffusion analysis window."""
        from .. import global_vars as g

        win = self._require_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None or 'tracks' not in spt:
            QtWidgets.QMessageBox.warning(
                self, "No Tracks",
                "No track data found.  Run detection and linking first.")
            return

        try:
            from .diffusion_plot import DiffusionAnalysisWindow
            diff_win = DiffusionAnalysisWindow(parent=g.m)

            # Build tracks_dict if not present
            tracks_dict = spt.get('tracks_dict')
            if tracks_dict is None:
                from ..spt.linking.greedy_linker import tracks_to_dict
                tracks_dict = tracks_to_dict(
                    spt['localizations'], spt['tracks'])
                spt['tracks_dict'] = tracks_dict

            diff_win.set_data(
                tracks_dict,
                pixel_size=self.ana_pixel_size.value(),
                frame_interval=self.ana_frame_interval.value())
            diff_win.show()
            logger.info("Diffusion analysis window opened")
        except ImportError:
            logger.warning("DiffusionAnalysisWindow not available")
            QtWidgets.QMessageBox.information(
                self, "Not Available",
                "Diffusion analysis window is not yet available.")
        except Exception as exc:
            logger.exception("Failed to open diffusion analysis: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"Failed to open diffusion analysis:\n{exc}")

    def _on_viz_flower(self):
        """Open the flower plot (origin-centered tracks) window."""
        from .. import global_vars as g

        win = self._require_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None or 'tracks' not in spt:
            QtWidgets.QMessageBox.warning(
                self, "No Tracks",
                "No track data found.  Run detection and linking first.")
            return

        try:
            from .flower_plot import FlowerPlotWindow
            flower_win = FlowerPlotWindow(parent=g.m)

            # Build tracks_dict if not present
            tracks_dict = spt.get('tracks_dict')
            if tracks_dict is None:
                from ..spt.linking.greedy_linker import tracks_to_dict
                tracks_dict = tracks_to_dict(
                    spt['localizations'], spt['tracks'])
                spt['tracks_dict'] = tracks_dict

            flower_win.set_tracks(
                tracks_dict,
                classification=spt.get('classification'),
                features=spt.get('features_by_track'))
            flower_win.show()
            logger.info("Flower plot window opened")
        except ImportError:
            logger.warning("FlowerPlotWindow not available")
            QtWidgets.QMessageBox.information(
                self, "Not Available",
                "Flower plot is not yet available.")
        except Exception as exc:
            logger.exception("Failed to open flower plot: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"Failed to open flower plot:\n{exc}")

    def _on_viz_chart_dock(self):
        """Open the chart dock for scatter/histogram plots."""
        from .. import global_vars as g

        win = self._require_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt is None or 'features_by_track' not in spt:
            QtWidgets.QMessageBox.warning(
                self, "No Features",
                "No feature data found.  Run feature computation first.")
            return

        try:
            import pandas as pd
            from .chart_dock import ChartDock
            chart = ChartDock(parent=g.m)

            # Build DataFrame from features_by_track
            features = spt['features_by_track']
            df = pd.DataFrame.from_dict(features, orient='index')
            df.index.name = 'track_id'

            # Add classification as a column if available
            classification = spt.get('classification')
            if classification:
                df['classification'] = df.index.map(
                    lambda tid: classification.get(tid, -1))

            chart.set_data(df)
            chart.show()
            logger.info("Chart dock opened")
        except ImportError:
            logger.warning("ChartDock not available")
            QtWidgets.QMessageBox.information(
                self, "Not Available",
                "Chart dock is not yet available.")
        except Exception as exc:
            logger.exception("Failed to open chart dock: %s", exc)
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"Failed to open chart dock:\n{exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_detection_params(self, psf_sigma=None, alpha=None,
                             min_intensity=None, method=None, **kwargs):
        """Programmatically set detection parameters.

        Useful for scripts and macros.  Extra keyword arguments are
        forwarded to the advanced U-Track detection controls
        (dog_ratio, mixture_separation, local_bg_inner, local_bg_outer).
        """
        if method is not None:
            idx = self.det_method_combo.findText(method)
            if idx >= 0:
                self.det_method_combo.setCurrentIndex(idx)
        if psf_sigma is not None:
            self.det_psf_sigma.setValue(psf_sigma)
        if alpha is not None:
            self.det_alpha.setValue(alpha)
        if min_intensity is not None:
            self.det_min_intensity.setValue(min_intensity)
        if kwargs:
            self.utrack_det_controls.set_params(kwargs)

    def set_linking_params(self, max_distance=None, max_gap=None,
                           min_track_length=None, method=None, **kwargs):
        """Programmatically set linking parameters.

        Extra keyword arguments are forwarded to the U-Track LAP
        controls (all TrackingConfig fields).
        """
        if method is not None:
            idx = self.link_method_combo.findText(method)
            if idx >= 0:
                self.link_method_combo.setCurrentIndex(idx)
        if max_distance is not None:
            self.link_max_distance.setValue(max_distance)
        if max_gap is not None:
            self.link_max_gap.setValue(max_gap)
        if min_track_length is not None:
            self.link_min_length.setValue(min_track_length)
        if kwargs:
            self.utrack_controls.set_config(kwargs)

    def set_analysis_params(self, pixel_size=None, frame_interval=None,
                            geometric=None, kinematic=None, spatial=None):
        """Programmatically set analysis parameters."""
        if pixel_size is not None:
            self.ana_pixel_size.setValue(pixel_size)
        if frame_interval is not None:
            self.ana_frame_interval.setValue(frame_interval)
        if geometric is not None:
            self.ana_geometric_cb.setChecked(geometric)
        if kinematic is not None:
            self.ana_kinematic_cb.setChecked(kinematic)
        if spatial is not None:
            self.ana_spatial_cb.setChecked(spatial)

    def run_full_pipeline(self):
        """Run detect -> link -> compute features sequentially.

        Convenience method for scripting.
        """
        self._on_detect()
        win = self._current_window()
        if win is None:
            return

        spt = _get_spt_data(win)
        if spt and 'localizations' in spt and len(spt['localizations']) > 0:
            self._on_link()

        if spt and 'tracks' in spt and len(spt['tracks']) > 0:
            self._on_compute_features()

    def get_results(self):
        """Return the current SPT results dict from the active window.

        Returns None if no window is active or no SPT data exists.
        """
        win = self._current_window()
        if win is None:
            return None
        return _get_spt_data(win)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Clear the singleton reference on close."""
        SPTControlPanel._instance = None
        super().closeEvent(event)
