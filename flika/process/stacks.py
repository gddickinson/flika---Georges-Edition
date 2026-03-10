import numpy as np
from qtpy import QtWidgets

import flika.global_vars as g
from flika.utils.BaseProcess import BaseProcess, BaseProcess_noPriorWindow
from flika.utils.custom_widgets import (
    CheckBox,
    ComboBox,
    MissingWindowError,
    SliderLabel,
    WindowSelector,
)
from flika.window import Window

__all__ = [
    "deinterleave",
    "trim",
    "zproject",
    "image_calculator",
    "pixel_binning",
    "frame_binning",
    "resize",
    "concatenate_stacks",
    "duplicate",
    "generate_random_image",
    "generate_phantom_volume",
    "change_datatype",
    "shear_transform",
    "frame_remover",
    "rotate_90",
    "rotate_custom",
    "flip_image",
]


def duplicate():
    old = g.win
    if old is None:
        g.alert("Select a window before trying to duplicate it.")
    else:
        return Window(old.image, old.name, old.filename, old.commands, old.metadata)


class Generate_Random_Image(BaseProcess_noPriorWindow):
    def __init__(self):
        super().__init__()
        self.puffs = None

    def get_init_settings_dict(self):
        s = dict()
        s["nFrames"] = 1000
        s["width"] = 128
        s["height"] = 128
        s["dimensions"] = "3D (T, X, Y)"
        return s

    def gui(self):
        self.gui_reset()
        dimensions = ComboBox()
        dimensions.addItems(["2D (X, Y)", "3D (T, X, Y)", "4D (T, X, Y, Z)", "4D (T, X, Y, C)"])
        nFrames = SliderLabel(0)
        nFrames.setRange(1, 10000)
        width = SliderLabel(0)
        width.setRange(2, 1024)
        height = SliderLabel(0)
        height.setRange(2, 1024)
        depth = SliderLabel(0)
        depth.setRange(2, 64)
        self.items.append(
            {"name": "dimensions", "string": "Dimensions", "object": dimensions}
        )
        self.items.append(
            {"name": "nFrames", "string": "Movie Duration (frames)", "object": nFrames}
        )
        self.items.append(
            {"name": "width", "string": "Width of image (pixels)", "object": width}
        )
        self.items.append(
            {"name": "height", "string": "Height of image (pixels)", "object": height}
        )
        self.items.append(
            {"name": "depth", "string": "Depth / Channels", "object": depth}
        )
        super().gui()

    def __call__(self, nFrames=10000, width=128, height=128, depth=16,
                 dimensions="3D (T, X, Y)"):
        self.start()
        if dimensions == "2D (X, Y)":
            self.newtif = np.random.randn(width, height)
            self.newname = " Random Noise 2D "
        elif dimensions == "3D (T, X, Y)":
            self.newtif = np.random.randn(nFrames, width, height)
            self.newname = " Random Noise 3D "
        elif dimensions == "4D (T, X, Y, Z)":
            self.newtif = np.random.randn(nFrames, width, height, depth)
            self.newname = " Random Noise 4D "
        elif dimensions == "4D (T, X, Y, C)":
            if depth > 4:
                depth = 3
            self.newtif = np.random.randn(nFrames, width, height, depth)
            self.newname = " Random Noise 4D-RGB "
        else:
            self.newtif = np.random.randn(nFrames, width, width)
            self.newname = " Random Noise "
        return self.end()


generate_random_image = Generate_Random_Image()


class Deinterleave(BaseProcess):
    """deinterleave(nChannels, keepSourceWindow=False)

    This deinterleaves a stack into nChannels

    Parameters:
        nChannels (int): The number of channels to deinterleave.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {"nChannels": 3}

    def gui(self):
        self.gui_reset()
        nChannels = QtWidgets.QSpinBox()
        nChannels.setMinimum(2)
        self.items.append(
            {"name": "nChannels", "string": "How many Channels?", "object": nChannels}
        )
        super().gui()

    def __call__(self, nChannels, keepSourceWindow=False):
        self.start(keepSourceWindow)

        newWindows = []
        for i in np.arange(nChannels):
            newtif = self.tif[i::nChannels]
            name = self.oldname + " - Channel " + str(i)
            newWindow = Window(newtif, name, self.oldwindow.filename)
            newWindows.append(newWindow)

        if keepSourceWindow is False:
            self.oldwindow.close()
        g.m.statusBar().showMessage("Finished with {}.".format(self.__name__))
        return newWindows


deinterleave = Deinterleave()


class Pixel_binning(BaseProcess):
    """pixel_binning(nPixels, keepSourceWindow=False)

    This bins the pixels to reduce the file size

    Parameters:
        nPixels (int): The number of pixels to bin.  Example: a value of 2 will reduce file size from 256x256->128x128.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        s = dict()
        s["nPixels"] = 2
        return s

    def gui(self):
        self.gui_reset()
        nPixels = QtWidgets.QSpinBox()
        nPixels.setMinimum(2)
        nPixels.setMaximum(2)
        self.items.append(
            {
                "name": "nPixels",
                "string": "How many adjacent pixels to bin?",
                "object": nPixels,
            }
        )
        super().gui()

    def __call__(self, nPixels, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif
        nDim = len(tif.shape)
        if nPixels == 2:
            if nDim == 4:
                image1 = tif[:, 0::2, 0::2, :]
                image2 = tif[:, 1::2, 0::2, :]
                image3 = tif[:, 0::2, 1::2, :]
                image4 = tif[:, 1::2, 1::2, :]
                self.newtif = (image1 + image2 + image3 + image4) / 4
                self.newname = self.oldname + " - Binned"
                return self.end()
            elif nDim == 3:
                mt, mx, my = tif.shape
                if mx % 2 == 1:
                    tif = tif[:, :-1, :]
                if my % 2 == 1:
                    tif = tif[:, :, :-1]
                image1 = tif[:, 0::2, 0::2]
                image2 = tif[:, 1::2, 0::2]
                image3 = tif[:, 0::2, 1::2]
                image4 = tif[:, 1::2, 1::2]
                self.newtif = (image1 + image2 + image3 + image4) / 4
                self.newname = self.oldname + " - Binned"
                return self.end()
            elif nDim == 2:
                image1 = tif[0::2, 0::2]
                image2 = tif[1::2, 0::2]
                image3 = tif[0::2, 1::2]
                image4 = tif[1::2, 1::2]
                self.newtif = (image1 + image2 + image3 + image4) / 4
                self.newname = self.oldname + " - Binned"
                return self.end()
        else:
            g.alert("2 is the only supported value for binning at the moment")


pixel_binning = Pixel_binning()


class Frame_binning(BaseProcess):
    """frame_binning(nFrames, keepSourceWindow=False)

    This bins the pixels to reduce the file size

    Parameters:
        nFrames (int): The number of frames to bin.  Example: a value of 2 will reduce number of frames from 1000 to 500.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {"nFrames": 2}

    def gui(self):
        self.gui_reset()
        nFrames = QtWidgets.QSpinBox()
        nFrames.setMinimum(2)
        nFrames.setMaximum(10000)
        self.items.append(
            {"name": "nFrames", "string": "How many frames to bin?", "object": nFrames}
        )
        super().gui()

    def __call__(self, nFrames, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif
        self.newtif = np.array(
            [np.mean(tif[i : i + nFrames], 0) for i in np.arange(0, len(tif), nFrames)]
        )
        self.newname = self.oldname + " - Binned {} frames".format(nFrames)
        return self.end()


frame_binning = Frame_binning()


class Resize(BaseProcess):
    """resize(factor, keepSourceWindow=False)

    Performs interpolation to up-size images

    Parameters:
        factor (int): The factor to scale the images by.  Example: a value of 2 will double the number of pixels wide the images are.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {"factor": 2}

    def gui(self):
        self.gui_reset()
        factor = QtWidgets.QSpinBox()
        factor.setMinimum(2)
        factor.setMaximum(100)
        self.items.append(
            {
                "name": "factor",
                "string": "By what factor to resize the image?",
                "object": factor,
            }
        )
        super().gui()

    def __call__(self, factor, keepSourceWindow=False):
        import skimage.transform

        self.start(keepSourceWindow)
        if self.tif.dtype in (np.uint64, np.int64):
            g.alert(
                "Resize fails on int64 and uint64 movie types, change the image type to resize."
            )
            return
        A = self.tif
        nDim = len(A.shape)
        is_rgb = (
            self.oldwindow.metadata["is_rgb"]
            or (nDim == 3 and A.shape[2] == 3)
            or nDim == 4
        )
        B = None
        if not is_rgb:
            if nDim == 3:
                mt, mx, my = A.shape
                B = np.zeros((mt, mx * factor, my * factor))
                for t in np.arange(mt):
                    B[t] = skimage.transform.resize(A[t], (mx * factor, my * factor))
            elif nDim == 2:
                mx, my = A.shape
                B = skimage.transform.resize(A, (mx * factor, my * factor))
        self.newtif = B
        self.newname = self.oldname + " - resized {}x ".format(factor)
        return self.end()


resize = Resize()


class Trim(BaseProcess):
    """trim(firstFrame, lastFrame, increment=1, delete=False, keepSourceWindow=False)

    This creates a new stack from the frames between the firstFrame and the lastFrame

    Parameters:
        firstFrame (int): The index of the first frame in the stack to be kept.
        lastFrame (int): The index of the last frame in the stack to be kept.
        increment (int): if increment equals i, then every ith frame is kept.
        delete (bool): if False, then the specified frames will be kept.  If True, they will be deleted.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        s = dict()
        s["firstFrame"] = 0
        s["lastFrame"] = g.win.image.shape[0]
        s["increment"] = 1
        s["delete"] = False
        return s

    def gui(self):
        self.gui_reset()
        nFrames = 1
        if g.win is not None:
            nFrames = g.win.image.shape[0]
        firstFrame = QtWidgets.QSpinBox()
        firstFrame.setMaximum(nFrames - 1)
        lastFrame = QtWidgets.QSpinBox()
        lastFrame.setRange(0, nFrames - 1)
        increment = QtWidgets.QSpinBox()
        increment.setMaximum(nFrames)
        increment.setMinimum(1)
        delete = CheckBox()

        self.items.append(
            {"name": "firstFrame", "string": "First Frame", "object": firstFrame}
        )
        self.items.append(
            {"name": "lastFrame", "string": "Last Frame", "object": lastFrame}
        )
        self.items.append(
            {"name": "increment", "string": "Increment", "object": increment}
        )
        self.items.append({"name": "delete", "string": "Delete", "object": delete})
        super().gui()

    def __call__(
        self, firstFrame, lastFrame, increment=1, delete=False, keepSourceWindow=False
    ):
        self.start(keepSourceWindow)
        if not delete:
            self.newtif = self.tif[firstFrame : lastFrame + 1 : increment]
        if delete:
            idxs_not = np.arange(firstFrame, lastFrame + 1, increment)
            idxs = np.ones(len(self.tif), dtype=bool)
            idxs[idxs_not] = False
            self.newtif = self.tif[idxs]
        self.newname = self.oldname + " - Kept Stack"
        return self.end()


trim = Trim()


class ZProject(BaseProcess):
    """zproject(firstFrame, lastFrame, projection_type, keepSourceWindow=False)

    This creates a new stack from the frames between the firstFrame and the lastFrame

    Parameters:
        firstFrame (int): The index of the first frame in the stack to be kept.
        lastFrame (int): The index of the last frame in the stack to be kept.
        projection_type (str): Method used to combine the frames.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {"firstFrame": 0, "lastFrame": 0, "projection_type": "Average"}

    def gui(self):
        self.gui_reset()
        nFrames = 1
        if g.win and len(g.win.image.shape) != 3:
            g.alert("zproject only works on 3 dimensional windows")
            return False
        if g.win is not None:
            nFrames = g.win.image.shape[0]
        firstFrame = QtWidgets.QSpinBox()
        firstFrame.setMaximum(nFrames)
        self.items.append(
            {"name": "firstFrame", "string": "First Frame", "object": firstFrame}
        )
        lastFrame = QtWidgets.QSpinBox()
        lastFrame.setRange(1, nFrames - 1)
        self.items.append(
            {"name": "lastFrame", "string": "Last Frame", "object": lastFrame}
        )
        projection_type = ComboBox()
        projection_type.addItem("Average")
        projection_type.addItem("Max Intensity")
        projection_type.addItem("Min Intensity")
        projection_type.addItem("Sum Slices")
        projection_type.addItem("Standard Deviation")
        projection_type.addItem("Median")
        self.items.append(
            {
                "name": "projection_type",
                "string": "Projection Type",
                "object": projection_type,
            }
        )
        super().gui()
        lastFrame.setValue(nFrames - 1)

    def __call__(self, firstFrame, lastFrame, projection_type, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.ndim != 3 or self.tif.shape[2] == 3:
            g.m.statusBar().showMessage(
                "zproject only works on 3 dimensional, non-color windows"
            )
            return False
        self.newtif = self.tif[firstFrame : lastFrame + 1]
        p = projection_type
        if p == "Average":
            self.newtif = np.mean(self.newtif, 0)
        elif p == "Max Intensity":
            self.newtif = np.max(self.newtif, 0)
        elif p == "Min Intensity":
            self.newtif = np.min(self.newtif, 0)
        elif p == "Sum Slices":
            self.newtif = np.sum(self.newtif, 0)
        elif p == "Standard Deviation":
            self.newtif = np.std(self.newtif, 0)
        elif p == "Median":
            self.newtif = np.median(self.newtif, 0)
        self.newname = self.oldname + " {} Projection".format(projection_type)
        return self.end()


zproject = ZProject()


class Image_calculator(BaseProcess):
    """###image_calculator(window1,window2,operation,keepSourceWindow=False)

    This creates a new stack by combining two windows in an operation

    Parameters:
        window1 (Window): The first window
        window2 (Window): The second window
        operation (str): Method used to combine the frames.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        operation = ComboBox()
        operation.addItems(
            [
                "Add",
                "Subtract",
                "Multiply",
                "Divide",
                "AND",
                "OR",
                "XOR",
                "Min",
                "Max",
                "Average",
            ]
        )
        self.items.append({"name": "window1", "string": "Window 1", "object": window1})
        self.items.append({"name": "window2", "string": "Window 2", "object": window2})
        self.items.append(
            {"name": "operation", "string": "Operation", "object": operation}
        )
        super().gui()

    def __call__(self, window1, window2, operation, keepSourceWindow=False):
        self.keepSourceWindow = keepSourceWindow
        g.m.statusBar().showMessage("Performing {}...".format(self.__name__))
        if window1 is None or window2 is None:
            g.alert(
                "You cannot execute '{}' without selecting a window first.".format(
                    self.__name__
                )
            )
            return None
        A = window1.image
        B = window2.image
        nDim1 = len(A.shape)
        nDim2 = len(B.shape)
        xyshape1 = A.shape
        xyshape2 = B.shape
        if nDim1 == 3:
            xyshape1 = xyshape1[1:]
        if nDim2 == 3:
            xyshape2 = xyshape2[1:]
        if xyshape1 != xyshape2:
            g.alert(
                "The two windows have images of different shapes. They could not be combined"
            )
            return None
        if nDim1 == 3 and nDim2 == 3:
            n1 = A.shape[0]
            n2 = B.shape[0]
            if n1 != n2:  # if the two movies have different # frames
                n = np.min([n1, n2])
                A = A[:n]  # shrink them so they have the same length
                B = B[:n]

        if operation == "Add":
            self.newtif = np.add(A, B)
        elif operation == "Subtract":
            self.newtif = np.subtract(A, B)
        elif operation == "Multiply":
            self.newtif = np.multiply(A, B)
        elif operation == "Divide":
            self.newtif = np.divide(A, B)
            self.newtif[np.isnan(self.newtif)] = 0
            self.newtif[np.isinf(self.newtif)] = 0
        if operation == "AND":
            self.newtif = np.logical_and(window1.image, window2.image).astype(np.uint8)
        elif operation == "OR":
            self.newtif = np.logical_or(window1.image, window2.image).astype(np.uint8)
        elif operation == "XOR":
            self.newtif = np.logical_xor(window1.image, window2.image).astype(np.uint8)
        elif operation == "Min":
            self.newtif = np.minimum(A, B)
        elif operation == "Max":
            self.newtif = np.maximum(A, B)
        elif operation == "Average":
            if nDim1 == 3 and nDim2 == 3:
                C = np.concatenate((np.expand_dims(A, 4), np.expand_dims(B, 4)), 3)
                self.newtif = np.mean(C, 3)
            elif nDim1 == 2 and nDim2 == 2:
                C = np.concatenate((np.expand_dims(A, 3), np.expand_dims(B, 3)), 2)
                self.newtif = np.mean(C, 2)
            else:
                if nDim1 == 3:
                    B = np.repeat(np.expand_dims(B, 0), len(A), 0)
                elif nDim2 == 3:
                    A = np.repeat(np.expand_dims(A, 0), len(B), 0)
                C = np.concatenate((np.expand_dims(A, 4), np.expand_dims(B, 4)), 3)
                self.newtif = np.mean(C, 3)
        #
        self.oldwindow = window1
        self.oldname = window1.name
        self.newname = self.oldname + " - {}".format(operation)
        if keepSourceWindow is False:
            print("closing both windows")
            window1.close()
            window2.close()
        g.m.statusBar().showMessage("Finished with {}.".format(self.__name__))
        newWindow = Window(self.newtif, str(self.newname), self.oldwindow.filename)
        del self.newtif
        return newWindow


image_calculator = Image_calculator()


class Concatenate_stacks(BaseProcess):
    """concatenate_stacks(window1,window2,keepSourceWindow=False)

    This creates a new stack by concatenating two stacks

    Parameters:
        window1 (Window): The first window
        window2 (Window): The second window
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        self.items.append({"name": "window1", "string": "Window 1", "object": window1})
        self.items.append({"name": "window2", "string": "Window 2", "object": window2})
        super().gui()

    def __call__(self, window1, window2, keepSourceWindow=False):
        self.keepSourceWindow = keepSourceWindow
        g.m.statusBar().showMessage("Performing {}...".format(self.__name__))
        if window1 is None or window2 is None:
            raise (
                MissingWindowError(
                    "You cannot execute '{}' without selecting a window first.".format(
                        self.__name__
                    )
                )
            )
        A = window1.image
        B = window2.image
        try:
            self.newtif = np.concatenate((A, B), 0)
        except ValueError:
            if len(A.shape) == 2 and len(B.shape) == 3:
                self.newtif = np.concatenate((np.expand_dims(A, 0), B), 0)
            elif len(A.shape) == 3 and len(B.shape) == 2:
                self.newtif = np.concatenate((A, np.expand_dims(B, 0)), 0)
        self.oldwindow = window1
        self.oldname = window1.name
        self.newname = self.oldname + " - {}".format("concatenated")
        if keepSourceWindow is False:
            print("closing both windows")
            window1.close()
            window2.close()
        g.m.statusBar().showMessage("Finished with {}.".format(self.__name__))
        newWindow = Window(self.newtif, str(self.newname), self.oldwindow.filename)
        del self.newtif
        return newWindow


concatenate_stacks = Concatenate_stacks()


class Change_datatype(BaseProcess):
    """change_datatype(datatype, keepSourceWindow=False)

    This bins the pixels to reduce the file size

    Parameters:
        datatype (string)
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        dtype = ComboBox()
        dtype_strs = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]
        for d in dtype_strs:
            dtype.addItem(d)
        self.items.append(
            {
                "name": "datatype",
                "string": "Convert image to which datatype?",
                "object": dtype,
            }
        )
        super().gui()

    def __call__(self, datatype, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = self.tif.astype(np.dtype(datatype))
        self.newname = self.oldname
        return self.end()


change_datatype = Change_datatype()


# ---------------------------------------------------------------------------
# Phantom volume helpers
# ---------------------------------------------------------------------------

def _make_sphere(vol, cx, cy, cz, r, value):
    """Draw a filled sphere into vol (X, Y, Z) at centre (cx, cy, cz)."""
    x0, x1 = max(0, int(cx - r)), min(vol.shape[0], int(cx + r + 1))
    y0, y1 = max(0, int(cy - r)), min(vol.shape[1], int(cy + r + 1))
    z0, z1 = max(0, int(cz - r)), min(vol.shape[2], int(cz + r + 1))
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    zs = np.arange(z0, z1)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    mask = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 <= r**2
    vol[x0:x1, y0:y1, z0:z1][mask] = value


def _make_ellipsoid(vol, cx, cy, cz, rx, ry, rz, value):
    """Draw a filled ellipsoid."""
    x0, x1 = max(0, int(cx - rx)), min(vol.shape[0], int(cx + rx + 1))
    y0, y1 = max(0, int(cy - ry)), min(vol.shape[1], int(cy + ry + 1))
    z0, z1 = max(0, int(cz - rz)), min(vol.shape[2], int(cz + rz + 1))
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    zs = np.arange(z0, z1)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    mask = ((xx - cx) / max(rx, 1))**2 + ((yy - cy) / max(ry, 1))**2 + ((zz - cz) / max(rz, 1))**2 <= 1
    vol[x0:x1, y0:y1, z0:z1][mask] = value


def _make_cube(vol, cx, cy, cz, half, value):
    """Draw a filled cube."""
    x0, x1 = max(0, int(cx - half)), min(vol.shape[0], int(cx + half + 1))
    y0, y1 = max(0, int(cy - half)), min(vol.shape[1], int(cy + half + 1))
    z0, z1 = max(0, int(cz - half)), min(vol.shape[2], int(cz + half + 1))
    vol[x0:x1, y0:y1, z0:z1] = value


def _make_cylinder(vol, cx, cy, cz, r, half_h, axis, value):
    """Draw a filled cylinder aligned to *axis* (0=X, 1=Y, 2=Z)."""
    centres = [cx, cy, cz]
    halfs = [r, r, r]
    halfs[axis] = half_h
    bbox = []
    for i in range(3):
        bbox.append((max(0, int(centres[i] - halfs[i])),
                      min(vol.shape[i], int(centres[i] + halfs[i] + 1))))
    xs = np.arange(bbox[0][0], bbox[0][1])
    ys = np.arange(bbox[1][0], bbox[1][1])
    zs = np.arange(bbox[2][0], bbox[2][1])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = [xx, yy, zz]
    radial = [i for i in range(3) if i != axis]
    dist_sq = (coords[radial[0]] - centres[radial[0]])**2 + (coords[radial[1]] - centres[radial[1]])**2
    mask = dist_sq <= r**2
    vol[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]][mask] = value


class Generate_Phantom_Volume(BaseProcess_noPriorWindow):
    """generate_phantom_volume(nFrames, width, height, depth, nObjects, content_type)

    Generates a 4D (T, X, Y, Z) volume populated with random 3D shapes
    for testing the volume viewer and orthogonal views.

    Parameters:
        nFrames (int): Number of time frames
        width (int): X dimension
        height (int): Y dimension
        depth (int): Z dimension
        nObjects (int): Number of shapes to place per frame
        content_type (str): 'Blobs' | 'Mixed shapes' | 'Moving blobs'
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {
            'nFrames': 10,
            'width': 64,
            'height': 64,
            'depth': 32,
            'nObjects': 12,
            'content_type': 'Mixed shapes',
        }

    def gui(self):
        self.gui_reset()
        nFrames = SliderLabel(0)
        nFrames.setRange(1, 200)
        nFrames.setValue(10)
        width = SliderLabel(0)
        width.setRange(16, 256)
        width.setValue(64)
        height = SliderLabel(0)
        height.setRange(16, 256)
        height.setValue(64)
        depth = SliderLabel(0)
        depth.setRange(4, 128)
        depth.setValue(32)
        nObjects = SliderLabel(0)
        nObjects.setRange(1, 100)
        nObjects.setValue(12)
        content_type = ComboBox()
        content_type.addItems(['Blobs', 'Mixed shapes', 'Moving blobs'])
        self.items.append({'name': 'nFrames', 'string': 'Frames (T)', 'object': nFrames})
        self.items.append({'name': 'width', 'string': 'Width (X)', 'object': width})
        self.items.append({'name': 'height', 'string': 'Height (Y)', 'object': height})
        self.items.append({'name': 'depth', 'string': 'Depth (Z)', 'object': depth})
        self.items.append({'name': 'nObjects', 'string': 'Number of objects', 'object': nObjects})
        self.items.append({'name': 'content_type', 'string': 'Content', 'object': content_type})
        super().gui()

    def __call__(self, nFrames=10, width=64, height=64, depth=32, nObjects=12,
                 content_type='Mixed shapes'):
        self.start()
        rng = np.random.default_rng()
        vol = np.zeros((nFrames, width, height, depth), dtype=np.float64)
        max_dim = min(width, height, depth)
        min_r = max(2, max_dim // 20)
        max_r = max(3, max_dim // 5)
        shape_kinds = ['sphere', 'ellipsoid', 'cube', 'cylinder']

        if content_type == 'Moving blobs':
            obj_params = []
            for _ in range(nObjects):
                cx0 = rng.uniform(max_r, width - max_r)
                cy0 = rng.uniform(max_r, height - max_r)
                cz0 = rng.uniform(max_r, depth - max_r)
                vx = rng.uniform(-1, 1)
                vy = rng.uniform(-1, 1)
                vz = rng.uniform(-0.5, 0.5)
                r = rng.uniform(min_r, max_r)
                intensity = rng.uniform(0.3, 1.0)
                obj_params.append((cx0, cy0, cz0, vx, vy, vz, r, intensity))
            for t in range(nFrames):
                frame = vol[t]
                frame += rng.normal(0, 0.02, frame.shape)
                for cx0, cy0, cz0, vx, vy, vz, r, intensity in obj_params:
                    cx = np.clip(cx0 + vx * t, r, width - r - 1)
                    cy = np.clip(cy0 + vy * t, r, height - r - 1)
                    cz = np.clip(cz0 + vz * t, r, depth - r - 1)
                    _make_sphere(frame, cx, cy, cz, r, intensity)
        else:
            template = np.zeros((width, height, depth), dtype=np.float64)
            for i in range(nObjects):
                cx = rng.uniform(max_r, width - max_r)
                cy = rng.uniform(max_r, height - max_r)
                cz = rng.uniform(max_r, depth - max_r)
                intensity = rng.uniform(0.3, 1.0)
                r = rng.uniform(min_r, max_r)
                if content_type == 'Blobs':
                    kind = 'sphere'
                else:
                    kind = rng.choice(shape_kinds)
                if kind == 'sphere':
                    _make_sphere(template, cx, cy, cz, r, intensity)
                elif kind == 'ellipsoid':
                    rx = rng.uniform(min_r, max_r)
                    ry = rng.uniform(min_r, max_r)
                    rz = rng.uniform(min_r, max_r)
                    _make_ellipsoid(template, cx, cy, cz, rx, ry, rz, intensity)
                elif kind == 'cube':
                    _make_cube(template, cx, cy, cz, r, intensity)
                elif kind == 'cylinder':
                    axis = rng.integers(0, 3)
                    half_h = rng.uniform(min_r, max_r * 1.5)
                    _make_cylinder(template, cx, cy, cz, r * 0.7, half_h, axis, intensity)
            for t in range(nFrames):
                vol[t] = template + rng.normal(0, 0.02, template.shape)

        self.newtif = vol
        self.newname = 'Phantom {} {}x{}x{}'.format(content_type, width, height, depth)
        return self.end()


generate_phantom_volume = Generate_Phantom_Volume()


# ---------------------------------------------------------------------------
# Shear Transform helpers
# ---------------------------------------------------------------------------

def _get_shear_transform_matrix(theta):
    """Compute the 2D shear matrix for a given angle in degrees."""
    theta_rad = np.radians(theta)
    hx = np.cos(theta_rad)
    sy = np.sin(theta_rad)
    return np.array([[1, hx, 0],
                     [0, sy, 0],
                     [0, 0,  1]])


def _get_shear_coordinates(image_2d, theta):
    """Compute old->new coordinate mapping for a 2D shear transform."""
    mx, my = image_2d.shape
    S = _get_shear_transform_matrix(theta)
    S_inv = np.linalg.inv(S)
    corners = np.array([[0, 0, 1], [mx - 1, 0, 1],
                        [0, my - 1, 1], [mx - 1, my - 1, 1]], dtype=np.float64)
    new_corners = (S @ corners.T).T
    min_r, min_c = new_corners[:, 0].min(), new_corners[:, 1].min()
    max_r, max_c = new_corners[:, 0].max(), new_corners[:, 1].max()
    new_rows = np.arange(int(np.floor(min_r)), int(np.ceil(max_r)) + 1)
    new_cols = np.arange(int(np.floor(min_c)), int(np.ceil(max_c)) + 1)
    nr, nc = np.meshgrid(new_rows, new_cols, indexing='ij')
    ones = np.ones_like(nr, dtype=np.float64)
    new_pts = np.stack([nr.ravel(), nc.ravel(), ones.ravel()], axis=0)
    old_pts = S_inv @ new_pts
    old_r = np.round(old_pts[0]).astype(int)
    old_c = np.round(old_pts[1]).astype(int)
    new_r = nr.ravel() - int(np.floor(min_r))
    new_c = nc.ravel() - int(np.floor(min_c))
    valid = (old_r >= 0) & (old_r < mx) & (old_c >= 0) & (old_c < my)
    return ([old_r[valid], old_c[valid]], [new_r[valid], new_c[valid]],
            int(np.ceil(max_r) - np.floor(min_r)) + 1,
            int(np.ceil(max_c) - np.floor(min_c)) + 1)


class Shear_Transform(BaseProcess):
    """shear_transform(theta, shift_factor, interpolate, keepSourceWindow=False)

    Applies a shear transformation to correct oblique-plane (e.g. light-sheet)
    imaging geometry.  Operates on 2-D, 3-D or 4-D data.

    Parameters:
        theta (float): Shear angle in degrees (typically 0-90).
        shift_factor (int): Up-sampling factor along the shear axis.
        interpolate (bool): Use smooth rescaling instead of pixel repetition.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'theta': 45.0, 'shift_factor': 1, 'interpolate': False}

    def gui(self):
        self.gui_reset()
        theta = SliderLabel(1)
        theta.setRange(1, 89)
        theta.setValue(45)
        shift_factor = SliderLabel(0)
        shift_factor.setRange(1, 8)
        shift_factor.setValue(1)
        interpolate = CheckBox()
        self.items.append({'name': 'theta', 'string': 'Angle (degrees)', 'object': theta})
        self.items.append({'name': 'shift_factor', 'string': 'Upsample factor', 'object': shift_factor})
        self.items.append({'name': 'interpolate', 'string': 'Smooth interpolation', 'object': interpolate})
        super().gui()

    def __call__(self, theta=45.0, shift_factor=1, interpolate=False, keepSourceWindow=False):
        self.start(keepSourceWindow)
        tif = self.tif
        dtype = tif.dtype
        if tif.ndim == 3:
            result = self._shear_stack(tif, theta, shift_factor, interpolate, dtype)
        elif tif.ndim == 4:
            result = self._shear_4d(tif, theta, shift_factor, interpolate, dtype)
        elif tif.ndim == 2:
            result = self._shear_2d(tif, theta, shift_factor, interpolate, dtype)
        else:
            g.alert('Shear transform requires 2D, 3D or 4D data')
            return None
        self.newtif = result
        self.newname = self.oldname + ' - Sheared {}deg'.format(theta)
        return self.end()

    @staticmethod
    def _upsample_axis(arr, axis, factor, interpolate):
        if factor <= 1:
            return arr
        if interpolate:
            try:
                from skimage.transform import rescale
                scales = [1.0] * arr.ndim
                scales[axis] = float(factor)
                return rescale(arr, tuple(scales), mode='constant',
                               preserve_range=True, anti_aliasing=False)
            except ImportError:
                pass
        return np.repeat(arr, int(factor), axis=axis)

    def _shear_2d(self, img, theta, shift_factor, interpolate, dtype):
        img = self._upsample_axis(img, 0, shift_factor, interpolate)
        old_c, new_c, nr, nc = _get_shear_coordinates(img, theta)
        out = np.zeros((nr, nc), dtype=dtype)
        out[new_c[0], new_c[1]] = img[old_c[0], old_c[1]]
        return out

    def _shear_stack(self, tif, theta, shift_factor, interpolate, dtype):
        tif = self._upsample_axis(tif, 1, shift_factor, interpolate)
        old_c, new_c, nr, nc = _get_shear_coordinates(tif[0], theta)
        mt = tif.shape[0]
        out = np.zeros((mt, nr, nc), dtype=dtype)
        for t in range(mt):
            out[t, new_c[0], new_c[1]] = tif[t, old_c[0], old_c[1]]
        return out

    def _shear_4d(self, tif, theta, shift_factor, interpolate, dtype):
        A = np.moveaxis(tif, 3, 1)
        mt, mz, my, mx = A.shape
        A = self._upsample_axis(A, 1, shift_factor, interpolate)
        old_c, new_c, nr, nc = _get_shear_coordinates(A[0, :, 0, :], theta)
        out = np.zeros((mt, nr, my, nc), dtype=dtype)
        for t in range(mt):
            for y in range(my):
                sl = A[t, :, y, :]
                out[t, new_c[0], y, new_c[1]] = sl[old_c[0], old_c[1]]
        out = np.moveaxis(out, 1, 3)
        return out


shear_transform = Shear_Transform()


class Frame_Remover(BaseProcess):
    """frame_remover(start, end, interval, length, keepSourceWindow=False)

    Removes frames from a stack at regular intervals.

    Parameters:
        start (int): First frame index to consider for removal.
        end (int): Last frame index to consider.
        interval (int): Spacing between removal blocks.
        length (int): Number of consecutive frames to remove per block.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'start': 0, 'end': 0, 'interval': 10, 'length': 1}

    def gui(self):
        self.gui_reset()
        nFrames = 1
        if g.win is not None and g.win.image.ndim >= 3:
            nFrames = g.win.image.shape[0]
        start = SliderLabel(0)
        start.setRange(0, max(0, nFrames - 1))
        start.setValue(0)
        end = SliderLabel(0)
        end.setRange(0, max(0, nFrames - 1))
        end.setValue(max(0, nFrames - 1))
        interval = SliderLabel(0)
        interval.setRange(1, max(1, nFrames))
        interval.setValue(10)
        length = SliderLabel(0)
        length.setRange(1, max(1, nFrames))
        length.setValue(1)
        self.items.append({'name': 'start', 'string': 'Start Frame', 'object': start})
        self.items.append({'name': 'end', 'string': 'End Frame', 'object': end})
        self.items.append({'name': 'interval', 'string': 'Interval', 'object': interval})
        self.items.append({'name': 'length', 'string': 'Frames to Remove', 'object': length})
        super().gui()

    def __call__(self, start, end, interval, length, keepSourceWindow=False):
        self.start(keepSourceWindow)
        if self.tif.ndim < 3:
            g.alert('Frame remover requires at least 3D data')
            return None
        start_indices = np.arange(start, end + 1, interval)
        to_delete = []
        for i in start_indices:
            to_delete.extend(range(i, min(i + length, self.tif.shape[0])))
        to_delete = sorted(set(to_delete))
        if len(to_delete) == 0:
            g.alert('No frames selected for removal')
            return None
        self.newtif = np.delete(self.tif, to_delete, axis=0)
        self.newname = self.oldname + ' - {} frames removed'.format(len(to_delete))
        return self.end()


frame_remover = Frame_Remover()


class Rotate_90(BaseProcess):
    """rotate_90(direction='clockwise', keepSourceWindow=False)

    Rotates the image by 90 degrees.

    Parameters:
        direction (str): 'Clockwise' or 'Counter-clockwise'.
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'direction': 'Clockwise'}

    def gui(self):
        self.gui_reset()
        direction = ComboBox()
        direction.addItems(['Clockwise', 'Counter-clockwise'])
        self.items.append({'name': 'direction', 'string': 'Direction', 'object': direction})
        super().gui()

    def __call__(self, direction='Clockwise', keepSourceWindow=False):
        self.start(keepSourceWindow)
        k = 1 if direction == 'Clockwise' else -1
        if self.tif.ndim == 3:
            self.newtif = np.array([np.rot90(frame, k=k) for frame in self.tif])
        elif self.tif.ndim == 4:
            self.newtif = np.array([[np.rot90(plane, k=k)
                                     for plane in vol] for vol in self.tif])
        else:
            self.newtif = np.rot90(self.tif, k=k)
        self.newname = self.oldname + ' - Rotated 90 {}'.format(direction)
        return self.end()


rotate_90 = Rotate_90()


class Rotate_Custom(BaseProcess):
    """rotate_custom(angle=0.0, reshape=True, order=1, keepSourceWindow=False)

    Rotates the image by an arbitrary angle using scipy.ndimage.rotate.

    Parameters:
        angle (float): Rotation angle in degrees (positive = counter-clockwise).
        reshape (bool): If True, the output is resized to contain the whole rotated image.
        order (int): Spline interpolation order (0=nearest, 1=bilinear, 3=cubic).
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'angle': 0.0, 'reshape': True, 'order': 1}

    def gui(self):
        self.gui_reset()
        angle = QtWidgets.QDoubleSpinBox()
        angle.setRange(-360, 360)
        angle.setDecimals(2)
        angle.setSingleStep(1.0)
        angle.setValue(0.0)
        reshape = CheckBox()
        reshape.setChecked(True)
        order = ComboBox()
        order.addItems(['0 - Nearest', '1 - Bilinear', '3 - Cubic'])
        order.setCurrentIndex(1)
        self.items.append({'name': 'angle', 'string': 'Angle (degrees)', 'object': angle})
        self.items.append({'name': 'reshape', 'string': 'Resize output', 'object': reshape})
        self.items.append({'name': 'order', 'string': 'Interpolation', 'object': order})
        super().gui()

    def __call__(self, angle=0.0, reshape=True, order=1, keepSourceWindow=False):
        from scipy.ndimage import rotate as ndi_rotate
        self.start(keepSourceWindow)
        if isinstance(order, str):
            order = int(order[0])
        if self.tif.ndim == 3:
            frames = []
            for t in range(self.tif.shape[0]):
                frames.append(ndi_rotate(self.tif[t], angle, reshape=reshape,
                                         order=order, mode='constant', cval=0))
            self.newtif = np.array(frames)
        elif self.tif.ndim == 4:
            result = []
            for t in range(self.tif.shape[0]):
                planes = []
                for z in range(self.tif.shape[1]):
                    planes.append(ndi_rotate(self.tif[t, z], angle, reshape=reshape,
                                             order=order, mode='constant', cval=0))
                result.append(np.array(planes))
            self.newtif = np.array(result)
        else:
            self.newtif = ndi_rotate(self.tif, angle, reshape=reshape,
                                     order=order, mode='constant', cval=0)
        self.newname = self.oldname + ' - Rotated {}deg'.format(angle)
        return self.end()


rotate_custom = Rotate_Custom()


class Flip_Image(BaseProcess):
    """flip_image(direction='Horizontal', keepSourceWindow=False)

    Flips (mirrors) the image along the chosen axis.

    Parameters:
        direction (str): 'Horizontal' (left-right) or 'Vertical' (up-down).
    Returns:
        newWindow
    """

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {'direction': 'Horizontal'}

    def gui(self):
        self.gui_reset()
        direction = ComboBox()
        direction.addItems(['Horizontal', 'Vertical'])
        self.items.append({'name': 'direction', 'string': 'Flip Direction', 'object': direction})
        super().gui()

    def __call__(self, direction='Horizontal', keepSourceWindow=False):
        self.start(keepSourceWindow)
        if direction == 'Horizontal':
            flip_axis = -1
        else:
            flip_axis = -2
        self.newtif = np.flip(self.tif, axis=flip_axis).copy()
        self.newname = self.oldname + ' - Flipped {}'.format(direction)
        return self.end()


flip_image = Flip_Image()
