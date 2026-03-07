# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/compositing.py'")
import numpy as np
from qtpy import QtWidgets, QtCore
from .. import global_vars as g
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox

__all__ = ['channel_compositor']


class Channel_Compositor(BaseProcess):
    """channel_compositor(keepSourceWindow=True)

    Opens the Channel Compositor panel for multi-channel overlay compositing.

    Add multiple grayscale windows as channels, each with independent colormap,
    brightness/contrast, and opacity controls. Channels are blended additively
    for standard fluorescence microscopy composite viewing.

    Parameters:
        keepSourceWindow (bool): Not used (channels are non-destructive overlays).
    Returns:
        None
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        if g.win is None:
            g.alert("No window is currently selected.")
            return
        self.__call__()

    def __call__(self, keepSourceWindow=True):
        from ..viewers.channel_compositor import ChannelCompositor
        from ..viewers.channel_panel import ChannelPanel

        host = g.win
        if host is None:
            raise MissingWindowError("No window selected for compositing.")

        # If already compositing on this window, just show the panel
        if hasattr(host, '_compositor') and host._compositor is not None:
            panel = ChannelPanel.instance(g.m)
            panel.set_compositor(host._compositor)
            g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
            panel.show()
            panel.raise_()
            return

        # Create new compositor
        compositor = ChannelCompositor(host)
        host._compositor = compositor

        # Open control panel
        panel = ChannelPanel.instance(g.m)
        panel.set_compositor(compositor)
        g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
        panel.show()
        panel.raise_()

        g.m.statusBar().showMessage('Channel Compositor opened. Add channels from the panel.')
        return None


channel_compositor = Channel_Compositor()

logger.debug("Completed 'reading process/compositing.py'")
