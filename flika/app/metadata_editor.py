"""REMBI-compliant metadata editor for flika.

Provides a tabbed dialog for editing imaging metadata following the
Recommended Metadata for Biological Images (REMBI) standard.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from qtpy import QtCore, QtWidgets

from flika.logger import logger
import flika.global_vars as g


REMBI_SCHEMA = {
    'Study': {
        'title': {'type': 'text', 'label': 'Title'},
        'description': {'type': 'multiline', 'label': 'Description'},
        'authors': {'type': 'text', 'label': 'Authors'},
        'license': {'type': 'choice', 'label': 'License',
                    'options': ['CC-BY-4.0', 'CC-BY-SA-4.0', 'CC0-1.0', 'MIT', 'Other']},
        'keywords': {'type': 'text', 'label': 'Keywords'},
    },
    'Sample': {
        'organism': {'type': 'text', 'label': 'Organism'},
        'tissue': {'type': 'text', 'label': 'Tissue/Cell Type'},
        'preparation': {'type': 'multiline', 'label': 'Sample Preparation'},
        'staining': {'type': 'text', 'label': 'Staining/Labeling'},
    },
    'Acquisition': {
        'imaging_method': {'type': 'choice', 'label': 'Imaging Method',
                           'options': ['Widefield', 'Confocal', 'Light Sheet',
                                       'Two-Photon', 'TIRF', 'STORM', 'PALM', 'SIM', 'Other']},
        'microscope': {'type': 'text', 'label': 'Microscope'},
        'objective': {'type': 'text', 'label': 'Objective'},
        'magnification': {'type': 'number', 'label': 'Magnification'},
        'numerical_aperture': {'type': 'number', 'label': 'Numerical Aperture'},
        'excitation_wavelength': {'type': 'text', 'label': 'Excitation Wavelength (nm)'},
        'emission_wavelength': {'type': 'text', 'label': 'Emission Wavelength (nm)'},
        'detector': {'type': 'text', 'label': 'Detector'},
        'exposure_time': {'type': 'text', 'label': 'Exposure Time'},
    },
    'Image': {
        'pixel_size_x': {'type': 'number', 'label': 'Pixel Size X (um)'},
        'pixel_size_y': {'type': 'number', 'label': 'Pixel Size Y (um)'},
        'pixel_size_z': {'type': 'number', 'label': 'Pixel Size Z (um)'},
        'time_interval': {'type': 'text', 'label': 'Time Interval'},
        'dimensions': {'type': 'text', 'label': 'Dimensions (read-only)'},
        'bit_depth': {'type': 'text', 'label': 'Bit Depth (read-only)'},
    },
    'Analysis': {
        'software': {'type': 'text', 'label': 'Software'},
        'version': {'type': 'text', 'label': 'Software Version'},
        'processing_steps': {'type': 'multiline', 'label': 'Processing Steps'},
        'parameters': {'type': 'multiline', 'label': 'Analysis Parameters'},
    },
}


class MetadataEditorDialog(QtWidgets.QDialog):
    """REMBI-compliant metadata editor with tabbed interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Metadata Editor (REMBI)')
        self.resize(600, 500)
        self._widgets: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        self._setup_ui()
        self._auto_populate()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        for module_name, fields in REMBI_SCHEMA.items():
            tab = QtWidgets.QWidget()
            form_layout = QtWidgets.QFormLayout(tab)
            self._widgets[module_name] = {}

            for field_name, spec in fields.items():
                widget = self._create_field_widget(spec)
                form_layout.addRow(spec['label'] + ':', widget)
                self._widgets[module_name][field_name] = widget

            self.tabs.addTab(tab, module_name)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        load_btn = QtWidgets.QPushButton('Load from JSON')
        load_btn.clicked.connect(self._load_json)
        btn_layout.addWidget(load_btn)

        save_btn = QtWidgets.QPushButton('Save as JSON')
        save_btn.clicked.connect(self._save_json)
        btn_layout.addWidget(save_btn)

        apply_btn = QtWidgets.QPushButton('Apply to Window')
        apply_btn.clicked.connect(self._apply_to_window)
        btn_layout.addWidget(apply_btn)

        btn_layout.addStretch()

        close_btn = QtWidgets.QPushButton('Close')
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _create_field_widget(self, spec):
        field_type = spec['type']
        if field_type == 'text':
            return QtWidgets.QLineEdit()
        elif field_type == 'multiline':
            w = QtWidgets.QTextEdit()
            w.setMaximumHeight(80)
            return w
        elif field_type == 'choice':
            combo = QtWidgets.QComboBox()
            combo.addItems(spec.get('options', []))
            combo.setEditable(True)
            return combo
        elif field_type == 'number':
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0, 999999)
            spin.setSpecialValueText('')
            return spin
        return QtWidgets.QLineEdit()

    def _get_field_value(self, widget):
        if isinstance(widget, QtWidgets.QTextEdit):
            return widget.toPlainText()
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        return ''

    def _set_field_value(self, widget, value):
        if isinstance(widget, QtWidgets.QTextEdit):
            widget.setPlainText(str(value))
        elif isinstance(widget, QtWidgets.QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
            else:
                widget.setEditText(str(value))
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            try:
                widget.setValue(float(value))
            except (ValueError, TypeError):
                pass
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(str(value))

    def _auto_populate(self):
        """Auto-fill fields from current window metadata."""
        from flika.version import __version__

        # Software info
        self._set_field_value(self._widgets['Analysis']['software'], 'flika')
        self._set_field_value(self._widgets['Analysis']['version'], __version__)

        if g.win is None:
            return

        # Image dimensions
        shape_str = ' x '.join(str(s) for s in g.win.image.shape)
        self._set_field_value(self._widgets['Image']['dimensions'], shape_str)
        self._set_field_value(self._widgets['Image']['bit_depth'], str(g.win.image.dtype))

        # Processing steps from commands
        if g.win.commands:
            steps = '\n'.join(g.win.commands)
            self._set_field_value(self._widgets['Analysis']['processing_steps'], steps)

        # Pixel sizes from OME metadata
        meta = g.win.metadata
        if 'pixel_size_x' in meta:
            self._set_field_value(self._widgets['Image']['pixel_size_x'], meta['pixel_size_x'])
        if 'pixel_size_y' in meta:
            self._set_field_value(self._widgets['Image']['pixel_size_y'], meta['pixel_size_y'])
        if 'pixel_size_z' in meta:
            self._set_field_value(self._widgets['Image']['pixel_size_z'], meta['pixel_size_z'])

        # Load existing REMBI data if present
        if 'rembi' in meta:
            self._load_data(meta['rembi'])

    def _get_all_data(self) -> dict:
        """Collect all field values into a nested dict."""
        data = {}
        for module_name, fields in self._widgets.items():
            data[module_name] = {}
            for field_name, widget in fields.items():
                data[module_name][field_name] = self._get_field_value(widget)
        return data

    def _load_data(self, data: dict):
        """Populate fields from a nested dict."""
        for module_name, fields in data.items():
            if module_name in self._widgets:
                for field_name, value in fields.items():
                    if field_name in self._widgets[module_name]:
                        self._set_field_value(self._widgets[module_name][field_name], value)

    def _load_json(self):
        from flika.utils.misc import open_file_gui
        path = open_file_gui('Load Metadata', filetypes='JSON files (*.json)')
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                self._load_data(data)
                g.m.statusBar().showMessage(f'Metadata loaded from {os.path.basename(path)}')
            except Exception as e:
                g.alert(f'Error loading metadata: {e}')

    def _save_json(self):
        from flika.utils.misc import save_file_gui
        path = save_file_gui('Save Metadata', filetypes='JSON files (*.json)')
        if path:
            data = self._get_all_data()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            g.m.statusBar().showMessage(f'Metadata saved to {os.path.basename(path)}')

    def _apply_to_window(self):
        if g.win is None:
            g.alert('No window selected.')
            return
        g.win.metadata['rembi'] = self._get_all_data()
        g.m.statusBar().showMessage('REMBI metadata applied to current window')
