# -*- coding: utf-8 -*-
"""
Dock Singleton Base Class
=========================

Provides :class:`DockSingleton`, a ``QDockWidget`` subclass that enforces a
single-instance pattern.  Subclasses obtain the instance via the
:meth:`instance` classmethod, which transparently handles creation and
recovery from C++ object deletion.

Usage::

    class MyPanel(DockSingleton):
        def __init__(self, parent=None):
            super().__init__('My Panel', parent)
            ...

    panel = MyPanel.instance(parent=main_window)
"""
from qtpy import QtWidgets


class DockSingleton(QtWidgets.QDockWidget):
    """Base class for singleton ``QDockWidget`` panels.

    Class Attributes
    ----------------
    _instance : DockSingleton or None
        The current singleton instance (per concrete subclass).

    Methods
    -------
    instance(parent=None)
        Return the existing instance if alive, or create a new one.
    cleanup()
        Override in subclasses to release resources before close.
    """

    _instance = None

    @classmethod
    def instance(cls, parent=None):
        """Return the singleton instance, creating it if necessary.

        Parameters
        ----------
        parent : QWidget or None
            Parent widget (typically the main application window).
        """
        if cls._instance is None or not cls._is_alive(cls._instance):
            cls._instance = cls(parent)
        return cls._instance

    @staticmethod
    def _is_alive(widget):
        """Check whether a Qt widget's underlying C++ object still exists."""
        try:
            widget.objectName()
            return True
        except RuntimeError:
            return False

    def cleanup(self):
        """Release resources before the panel is closed.

        Subclasses should override this to perform custom teardown.
        The default implementation does nothing.
        """
        pass

    def closeEvent(self, event):
        """Clean up and clear the singleton reference on close."""
        self.cleanup()
        type(self)._instance = None
        super().closeEvent(event)
