# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/color.py'")
import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from .. import global_vars as g
from ..window import Window
from ..utils.BaseProcess import BaseProcess, WindowSelector, MissingWindowError, CheckBox, ComboBox, SliderLabel

__all__ = ['split_channels', 'Split_channels', 'blend_channels']


class Split_channels(BaseProcess):
    """ split_channels(keepSourceWindow=False)

    This splits the color channels in a Window

    Returns:
        list of new Windows
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        super().gui()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        newWindows = []
        if not self.oldwindow.metadata['is_rgb']:
            g.alert('Cannot split channels, no colors detected.')
            return None
        nChannels = self.tif.shape[-1]
        for i in range(nChannels):
            newtif = self.tif[..., i]
            name = self.oldname + ' - Channel ' + str(i)
            newWindow = Window(newtif, name, self.oldwindow.filename)
            newWindows.append(newWindow)
        if keepSourceWindow is False:
            self.oldwindow.close()
        g.m.statusBar().showMessage('Finished with {}.'.format(self.__name__))
        return newWindows


split_channels = Split_channels()


class Blend_Channels(BaseProcess):
    """blend_channels(window1, window2, mode, alpha, keepSourceWindow=False)

    Blends two single-channel windows into a composite image.

    Parameters:
        window1 (Window): First channel (displayed as green by default).
        window2 (Window): Second channel (displayed as magenta by default).
        mode (str): Blending mode — 'Additive', 'Screen', 'Multiply', or 'Alpha'.
        alpha (float): Mixing weight for window1 (0-1). window2 weight is 1-alpha.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'mode': 'Additive', 'alpha': 0.5}

    def gui(self):
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        mode = ComboBox()
        mode.addItems(['Additive', 'Screen', 'Multiply', 'Alpha'])
        alpha = SliderLabel(2)
        alpha.setRange(0, 1)
        alpha.setValue(0.5)
        self.items.append({'name': 'window1', 'string': 'Channel 1 (green)', 'object': window1})
        self.items.append({'name': 'window2', 'string': 'Channel 2 (magenta)', 'object': window2})
        self.items.append({'name': 'mode', 'string': 'Blend Mode', 'object': mode})
        self.items.append({'name': 'alpha', 'string': 'Alpha (Ch1 weight)', 'object': alpha})
        super().gui()

    def __call__(self, window1, window2, mode='Additive', alpha=0.5, keepSourceWindow=False):
        self.keepSourceWindow = keepSourceWindow
        g.m.statusBar().showMessage('Performing {}...'.format(self.__name__))
        if window1 is None or window2 is None:
            raise MissingWindowError("Select two windows to blend.")

        A = window1.image.astype(np.float64)
        B = window2.image.astype(np.float64)

        # Normalise both to 0-1
        a_min, a_max = np.nanmin(A), np.nanmax(A)
        b_min, b_max = np.nanmin(B), np.nanmax(B)
        if a_max - a_min > 0:
            A = (A - a_min) / (a_max - a_min)
        if b_max - b_min > 0:
            B = (B - b_min) / (b_max - b_min)

        # Ensure same spatial shape (trim time if needed)
        if A.ndim == 3 and B.ndim == 3:
            n = min(A.shape[0], B.shape[0])
            A, B = A[:n], B[:n]
            if A.shape[1:] != B.shape[1:]:
                g.alert('Windows have different spatial dimensions')
                return None
        elif A.ndim == 2 and B.ndim == 2:
            if A.shape != B.shape:
                g.alert('Windows have different spatial dimensions')
                return None
        else:
            g.alert('Windows must have the same number of dimensions')
            return None

        if mode == 'Additive':
            blended = alpha * A + (1 - alpha) * B
        elif mode == 'Screen':
            blended = 1.0 - (1.0 - alpha * A) * (1.0 - (1 - alpha) * B)
        elif mode == 'Multiply':
            blended = A * B
        elif mode == 'Alpha':
            blended = alpha * A + (1 - alpha) * B
        else:
            blended = alpha * A + (1 - alpha) * B

        # Create RGB composite: Ch1=green, Ch2=magenta
        if blended.ndim == 2:
            rgb = np.zeros(blended.shape + (3,), dtype=np.float64)
            rgb[..., 0] = (1 - alpha) * B  # Red from Ch2
            rgb[..., 1] = alpha * A         # Green from Ch1
            rgb[..., 2] = (1 - alpha) * B  # Blue from Ch2
        else:
            rgb = np.zeros(blended.shape + (3,), dtype=np.float64)
            rgb[..., 0] = (1 - alpha) * B
            rgb[..., 1] = alpha * A
            rgb[..., 2] = (1 - alpha) * B

        self.newtif = np.clip(rgb, 0, 1)
        self.oldwindow = window1
        self.oldname = window1.name
        self.newname = self.oldname + f' - Blended ({mode})'
        if keepSourceWindow is False:
            window1.close()
            window2.close()
        g.m.statusBar().showMessage('Finished with {}.'.format(self.__name__))
        metadata = {'is_rgb': True}
        newWindow = Window(self.newtif, str(self.newname), self.oldwindow.filename, metadata=metadata)
        del self.newtif
        return newWindow


blend_channels = Blend_Channels()


logger.debug("Completed 'reading process/color.py'")