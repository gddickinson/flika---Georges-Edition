import os
import sys
from qtpy import QtCore, QtGui, QtWidgets, PYQT5

__all__ = ['get_qapp']

qapp = None


def get_qapp(icon_path=None, headless=False):
    """Get the QApplication instance currently in use. If no QApplication exists,
    one is created and the standard window icon is set to icon_path.

    Args:
        icon_path (str): location of icon to use as default window icon
        headless (bool): if True, use offscreen platform plugin (no display needed)

    Returns:
        QtGui.QApplication: the current application process
    """
    global qapp
    qapp = QtWidgets.QApplication.instance()
    if qapp is None:
        if headless and 'QT_QPA_PLATFORM' not in os.environ:
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        qapp = QtWidgets.QApplication(sys.argv[:1])
        qapp.setQuitOnLastWindowClosed(not headless)
        if icon_path is not None:
            qapp.setWindowIcon(QtGui.QIcon(icon_path))

    # Make sure we use high resolution icons with PyQt5 for HDPI
    # displays. TODO: check impact on non-HDPI displays.
    if PYQT5:
        qapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    return qapp