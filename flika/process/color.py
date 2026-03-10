import numpy as np
from qtpy import QtWidgets, QtCore, QtGui

import flika.global_vars as g
from flika.utils.BaseProcess import (
    BaseProcess,
    WindowSelector,
    MissingWindowError,
    CheckBox,
    ComboBox,
    SliderLabel,
)
from flika.window import Window

__all__ = [
    "split_channels",
    "Split_channels",
    "blend_channels",
    "convert_color_space",
    "grayscale",
]


class Split_channels(BaseProcess):
    """split_channels(keepSourceWindow=False)

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
        if not self.oldwindow.metadata["is_rgb"]:
            g.alert("Cannot split channels, no colors detected.")
            return None
        nChannels = self.tif.shape[-1]
        for i in range(nChannels):
            newtif = self.tif[..., i]
            name = self.oldname + " - Channel " + str(i)
            newWindow = Window(newtif, name, self.oldwindow.filename)
            newWindows.append(newWindow)
        if keepSourceWindow is False:
            self.oldwindow.close()
        g.m.statusBar().showMessage("Finished with {}.".format(self.__name__))
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
        return {"mode": "Additive", "alpha": 0.5}

    def gui(self):
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        mode = ComboBox()
        mode.addItems(["Additive", "Screen", "Multiply", "Alpha"])
        alpha = SliderLabel(2)
        alpha.setRange(0, 1)
        alpha.setValue(0.5)
        self.items.append(
            {"name": "window1", "string": "Channel 1 (green)", "object": window1}
        )
        self.items.append(
            {"name": "window2", "string": "Channel 2 (magenta)", "object": window2}
        )
        self.items.append(
            {"name": "mode", "string": "Blend Mode", "object": mode}
        )
        self.items.append(
            {"name": "alpha", "string": "Alpha (Ch1 weight)", "object": alpha}
        )
        super().gui()

    def __call__(
        self, window1, window2, mode="Additive", alpha=0.5, keepSourceWindow=False
    ):
        self.keepSourceWindow = keepSourceWindow
        g.m.statusBar().showMessage("Performing {}...".format(self.__name__))
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
                g.alert("Windows have different spatial dimensions")
                return None
        elif A.ndim == 2 and B.ndim == 2:
            if A.shape != B.shape:
                g.alert("Windows have different spatial dimensions")
                return None
        else:
            g.alert("Windows must have the same number of dimensions")
            return None

        if mode == "Additive":
            blended = alpha * A + (1 - alpha) * B
        elif mode == "Screen":
            blended = 1.0 - (1.0 - alpha * A) * (1.0 - (1 - alpha) * B)
        elif mode == "Multiply":
            blended = A * B
        elif mode == "Alpha":
            blended = alpha * A + (1 - alpha) * B
        else:
            blended = alpha * A + (1 - alpha) * B

        # Create RGB composite: Ch1=green, Ch2=magenta
        if blended.ndim == 2:
            rgb = np.zeros(blended.shape + (3,), dtype=np.float64)
            rgb[..., 0] = (1 - alpha) * B  # Red from Ch2
            rgb[..., 1] = alpha * A  # Green from Ch1
            rgb[..., 2] = (1 - alpha) * B  # Blue from Ch2
        else:
            rgb = np.zeros(blended.shape + (3,), dtype=np.float64)
            rgb[..., 0] = (1 - alpha) * B
            rgb[..., 1] = alpha * A
            rgb[..., 2] = (1 - alpha) * B

        self.newtif = np.clip(rgb, 0, 1)
        self.oldwindow = window1
        self.oldname = window1.name
        self.newname = self.oldname + f" - Blended ({mode})"
        if keepSourceWindow is False:
            window1.close()
            window2.close()
        g.m.statusBar().showMessage("Finished with {}.".format(self.__name__))
        metadata = {"is_rgb": True}
        newWindow = Window(
            self.newtif, str(self.newname), self.oldwindow.filename, metadata=metadata
        )
        del self.newtif
        return newWindow


blend_channels = Blend_Channels()


class Convert_Color_Space(BaseProcess):
    """convert_color_space(conversion, keepSourceWindow=False)

    Converts an RGB image between color spaces.

    Parameters:
        conversion (str): Color space conversion.
            'RGB to HSV', 'HSV to RGB', 'RGB to LAB', 'LAB to RGB',
            'RGB to YCrCb', 'YCrCb to RGB'.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        conversion = ComboBox()
        conversion.addItems(
            [
                "RGB to HSV",
                "HSV to RGB",
                "RGB to LAB",
                "LAB to RGB",
                "RGB to YCrCb",
                "YCrCb to RGB",
            ]
        )
        self.items.append(
            {"name": "conversion", "string": "Conversion", "object": conversion}
        )
        super().gui()

    def __call__(self, conversion="RGB to HSV", keepSourceWindow=False):
        self.start(keepSourceWindow)
        from skimage import color as skcolor

        tif = self.tif.astype(np.float64)

        converters = {
            "RGB to HSV": skcolor.rgb2hsv,
            "HSV to RGB": skcolor.hsv2rgb,
            "RGB to LAB": skcolor.rgb2lab,
            "LAB to RGB": skcolor.lab2rgb,
            "RGB to YCrCb": lambda x: (
                skcolor.rgb2ycbcr(x) if hasattr(skcolor, "rgb2ycbcr") else x
            ),
            "YCrCb to RGB": lambda x: (
                skcolor.ycbcr2rgb(x) if hasattr(skcolor, "ycbcr2rgb") else x
            ),
        }

        convert_fn = converters.get(conversion)
        if convert_fn is None:
            g.alert(f"Unknown conversion: {conversion}")
            return None

        # Normalize to 0-1 for skimage color conversions that expect it
        if conversion.startswith("RGB"):
            vmin, vmax = tif.min(), tif.max()
            if vmax - vmin > 0:
                tif = (tif - vmin) / (vmax - vmin)

        if tif.ndim == 3 and tif.shape[-1] in (3, 4):
            self.newtif = convert_fn(tif[..., :3])
        elif tif.ndim == 4 and tif.shape[-1] in (3, 4):
            result = np.zeros(tif.shape[:3] + (3,), dtype=np.float64)
            for t in range(tif.shape[0]):
                result[t] = convert_fn(tif[t, ..., :3])
            self.newtif = result
        else:
            g.alert("Color conversion requires an RGB image (last dim = 3 or 4).")
            return None

        self.newname = self.oldname + f" - {conversion}"
        return self.end()


convert_color_space = Convert_Color_Space()


class Grayscale(BaseProcess):
    """grayscale(method, keepSourceWindow=False)

    Converts an RGB image to grayscale.

    Parameters:
        method (str): 'Luminance' (standard weighted), 'Average', or 'Lightness'.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        method = ComboBox()
        method.addItems(["Luminance", "Average", "Lightness"])
        self.items.append(
            {"name": "method", "string": "Method", "object": method}
        )
        super().gui()

    def __call__(self, method="Luminance", keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif.astype(np.float64)

        if tif.ndim < 3 or tif.shape[-1] not in (3, 4):
            g.alert("Grayscale conversion requires an RGB image.")
            return None

        rgb = tif[..., :3]
        if method == "Luminance":
            self.newtif = (
                0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            )
        elif method == "Average":
            self.newtif = rgb.mean(axis=-1)
        elif method == "Lightness":
            self.newtif = (rgb.max(axis=-1) + rgb.min(axis=-1)) / 2
        else:
            self.newtif = rgb.mean(axis=-1)

        self.newname = self.oldname + " - Grayscale"
        return self.end()


grayscale = Grayscale()
