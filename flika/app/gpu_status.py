"""GPU / Acceleration Status dialog for flika's Help menu.

Shows detected compute devices, backend versions, and usability status.
Follows the same pattern as ``dependency_checker.py``.
"""
from __future__ import annotations

from qtpy import QtCore, QtWidgets


class GPUStatusDialog(QtWidgets.QDialog):
    """Dialog showing detected GPU/acceleration devices and their status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU / Acceleration Status")
        self.resize(700, 450)
        layout = QtWidgets.QVBoxLayout(self)

        self._text = QtWidgets.QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFontFamily("monospace")
        layout.addWidget(self._text)

        btn_layout = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._run_check)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._run_check()

    def _run_check(self):
        from ..utils.accel import detect_devices
        info = detect_devices(force_refresh=True)
        self._text.setPlainText(info.status_report())
