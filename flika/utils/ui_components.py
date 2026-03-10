# -*- coding: utf-8 -*-
"""
Shared UI Components
====================

Reusable Qt widgets and factory functions used across flika viewers and dialogs.

- :class:`CollapsibleSection` — a widget with a toggle button that shows/hides
  its child form layout.
- :func:`make_double_spin` — create a ``QDoubleSpinBox`` with sensible defaults.
- :func:`make_int_spin` — create a ``QSpinBox`` with sensible defaults.
- :func:`make_hline` — create a horizontal line separator.
"""
from qtpy import QtCore, QtWidgets


# ---------------------------------------------------------------------------
# Collapsible section
# ---------------------------------------------------------------------------

class CollapsibleSection(QtWidgets.QWidget):
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


# ---------------------------------------------------------------------------
# Spin-box factories
# ---------------------------------------------------------------------------

def make_double_spin(value, lo, hi, decimals=3, suffix='', step=None):
    """Create a ``QDoubleSpinBox`` with sensible defaults."""
    sb = QtWidgets.QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    if step is not None:
        sb.setSingleStep(step)
    return sb


def make_int_spin(value, lo, hi, suffix=''):
    """Create a ``QSpinBox`` with sensible defaults."""
    sb = QtWidgets.QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    return sb


# ---------------------------------------------------------------------------
# Separator
# ---------------------------------------------------------------------------

def make_hline():
    """Create a horizontal line separator."""
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return line
