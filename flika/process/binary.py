"""
Binary image processing functions for the flika package.
"""

import jaxtyping
import numpy as np
import scipy
import scipy.ndimage
from qtpy import QtGui, QtWidgets
from skimage import feature, measure
from skimage.filters.thresholding import threshold_local
from skimage.morphology import remove_small_objects

# Local application imports
import flika.window
from flika import global_vars as g
from flika.utils.ndim import per_plane
from flika.roi import ROI_Drawing, makeROI
from flika.utils.BaseProcess import BaseProcess
from flika.utils.custom_widgets import (
    CheckBox,
    ComboBox,
    MissingWindowError,
    SliderLabel,
    WindowSelector,
)

__all__ = [
    "threshold",
    "remove_small_blobs",
    "adaptive_threshold",
    "logically_combine",
    "binary_dilation",
    "binary_erosion",
    "generate_rois",
    "canny_edge_detector",
    "analyze_particles",
    "grayscale_opening",
    "grayscale_closing",
    "morphological_gradient",
    "h_maxima",
    "h_minima",
    "area_opening",
    "area_closing",
    "remove_small_holes",
    "flood_fill_process",
    "hysteresis_threshold",
    "multi_otsu_threshold",
]


def convert2uint8(
    tif: jaxtyping.Num[np.ndarray, "..."],
) -> jaxtyping.UInt8[np.ndarray, "..."]:
    """Convert any array to uint8 format with proper scaling.

    Args:
        tif: Input array to be converted

    Returns:
        Array in uint8 format
    """
    oldmin = np.min(tif)
    oldmax = np.max(tif)
    newmax = 2**8 - 1
    scaled_tif = ((tif - oldmin) * newmax) / (oldmax - oldmin)
    return scaled_tif.astype(np.uint8)


class Threshold(BaseProcess):
    """Applies a threshold to an image to create a binary mask.

    Parameters:
        value (float): The threshold to be applied
        darkBackground (bool): If True, pixels below the threshold will be True
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the thresholded image
    """

    def gui(self) -> None:
        self.gui_reset()
        valueSlider = SliderLabel(2)
        if g.win is not None:
            image = g.win.image
            valueSlider.setRange(np.min(image), np.max(image))
            valueSlider.setValue(np.mean(image))
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({"name": "value", "string": "Value", "object": valueSlider})
        self.items.append(
            {
                "name": "darkBackground",
                "string": "Dark Background",
                "object": CheckBox(),
            }
        )
        self.items.append({"name": "preview", "string": "Preview", "object": preview})
        super().gui()

    def __call__(
        self, value: float, darkBackground: bool = False, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        if self.oldwindow.nDims > 3:
            g.alert(
                "You cannot run this function on an image of dimension greater than 3. If your window has color, convert to a grayscale image before running this function"
            )
            return None

        if darkBackground:
            newtif = self.tif < value
        else:
            newtif = self.tif > value

        self.newtif = newtif.astype(np.uint8)
        self.newname = f"{self.oldname} - Thresholded {value}"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return

        win = g.win
        value = self.getValue("value")
        preview = self.getValue("preview")
        darkBackground = self.getValue("darkBackground")

        if win.nDims > 3:
            g.alert(
                "You cannot run this function on an image of dimension greater than 3. If your window has color, convert to a grayscale image before running this function"
            )
            return None

        if preview:
            if win.nDims == 3:  # if the image is 3d
                testimage = np.copy(win.image[win.currentIndex])
            elif win.nDims == 2:
                testimage = np.copy(win.image)

            if darkBackground:
                testimage = testimage < value
            else:
                testimage = testimage > value

            win.imageview.setImage(testimage, autoLevels=False)
            win.imageview.setLevels(-0.1, 1.1)
        else:
            win.reset()
            if win.nDims == 3:
                image = win.image[win.currentIndex]
            else:
                image = win.image
            win.imageview.setLevels(np.min(image), np.max(image))


threshold = Threshold()


class BlocksizeSlider(SliderLabel):
    """A slider that only allows odd values."""

    def __init__(self, decimals: int = 0):
        SliderLabel.__init__(self, decimals)

    def updateSlider(self, value: int) -> None:
        if value % 2 == 0:
            if value < self.slider.value():
                value -= 1
            else:
                value += 1
            self.label.setValue(value)
        self.slider.setValue(int(value * 10**self.decimals))

    def updateLabel(self, value: int | float) -> None:
        if value % 2 == 0:
            value -= 1
        self.label.setValue(value)


class Adaptive_threshold(BaseProcess):
    """Applies an adaptive threshold to an image using the scikit-image threshold_local function.

    Parameters:
        value (int): The threshold offset to be applied
        block_size (int): Size of pixel neighborhood used to calculate the threshold (must be odd)
        darkBackground (bool): If True, pixels below the threshold will be True
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the thresholded image
    """

    def gui(self) -> None:
        self.gui_reset()
        valueSlider = SliderLabel(2)
        valueSlider.setRange(-20, 20)
        valueSlider.setValue(0)

        block_size = BlocksizeSlider(0)
        if g.win is not None:
            max_block = int(max([g.win.image.shape[-1], g.win.image.shape[-2]]) / 2)
        else:
            max_block = 100
        block_size.setRange(3, max_block)

        preview = CheckBox()
        preview.setChecked(True)

        self.items.append({"name": "value", "string": "Value", "object": valueSlider})
        self.items.append(
            {"name": "block_size", "string": "Block Size", "object": block_size}
        )
        self.items.append(
            {
                "name": "darkBackground",
                "string": "Dark Background",
                "object": CheckBox(),
            }
        )
        self.items.append({"name": "preview", "string": "Preview", "object": preview})

        super().gui()
        self.preview()

    def __call__(
        self,
        value: float,
        block_size: int,
        darkBackground: bool = False,
        keepSourceWindow: bool = False,
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)

        if self.tif.dtype == np.float16:
            g.alert("Local Threshold does not support float16 type arrays")
            return None

        newtif = np.copy(self.tif)

        if self.oldwindow.nDims == 2:
            newtif = threshold_local(newtif, block_size, offset=value)
        elif self.oldwindow.nDims == 3:
            for i in range(len(newtif)):
                newtif[i] = threshold_local(newtif[i], block_size, offset=value)
        else:
            g.alert(
                "You cannot run this function on an image of dimension greater than 3. If your window has color, convert to a grayscale image before running this function"
            )
            return None

        if darkBackground:
            newtif = np.logical_not(newtif)

        self.newtif = newtif.astype(np.uint8)
        self.newname = f"{self.oldname} - Thresholded {value}"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return

        win = g.win
        value = self.getValue("value")
        block_size = self.getValue("block_size")
        preview = self.getValue("preview")
        darkBackground = self.getValue("darkBackground")

        nDim = len(win.image.shape)
        if nDim > 3:
            g.alert(
                "You cannot run this function on an image of dimension greater than 3. If your window has color, convert to a grayscale image before running this function"
            )
            return None

        if preview:
            if nDim == 3:  # if the image is 3d
                testimage = np.copy(win.image[win.currentIndex])
            elif nDim == 2:
                testimage = np.copy(win.image)

            testimage = threshold_local(testimage, block_size, offset=value)

            if darkBackground:
                testimage = np.logical_not(testimage)

            testimage = testimage.astype(np.uint8)
            win.imageview.setImage(testimage, autoLevels=False)
            win.imageview.setLevels(-0.1, 1.1)
        else:
            win.reset()
            if nDim == 3:
                image = win.image[win.currentIndex]
            else:
                image = win.image
            win.imageview.setLevels(np.min(image), np.max(image))


adaptive_threshold = Adaptive_threshold()


class Canny_edge_detector(BaseProcess):
    """Detects edges in an image using the Canny edge detection algorithm.

    Parameters:
        sigma (float): Standard deviation for Gaussian kernel
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the edge detection results
    """

    def gui(self) -> None:
        self.gui_reset()
        sigma = SliderLabel(2)
        if g.win is not None:
            sigma.setRange(0, 1000)
            sigma.setValue(1)

        preview = CheckBox()
        preview.setChecked(True)

        self.items.append({"name": "sigma", "string": "Sigma", "object": sigma})
        self.items.append({"name": "preview", "string": "Preview", "object": preview})

        super().gui()
        self.preview()

    def __call__(
        self, sigma: float, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        nDim = len(self.tif.shape)
        newtif = np.copy(self.tif)

        if self.tif.dtype == np.float16:
            g.alert(
                "Canny Edge Detection does not work on float16 images. Change the data type to use this function."
            )
            return None

        if nDim == 2:
            newtif = feature.canny(self.tif, sigma)
        else:
            for i in range(len(newtif)):
                newtif[i] = feature.canny(self.tif[i], sigma)

        self.newtif = newtif.astype(np.uint8)
        self.newname = f"{self.oldname} - Canny"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return

        win = g.win
        sigma = self.getValue("sigma")
        preview = self.getValue("preview")
        nDim = len(win.image.shape)

        if preview:
            if nDim == 3:  # if the image is 3d
                testimage = np.copy(win.image[win.currentIndex])
            elif nDim == 2:
                testimage = np.copy(win.image)

            testimage = feature.canny(testimage, sigma)
            win.imageview.setImage(testimage, autoLevels=False)
            win.imageview.setLevels(-0.1, 1.1)
        else:
            win.reset()
            if nDim == 3:
                image = win.image[win.currentIndex]
            else:
                image = win.image
            win.imageview.setLevels(np.min(image), np.max(image))


canny_edge_detector = Canny_edge_detector()


class Logically_combine(BaseProcess):
    """Combines two binary images with a logical operation.

    Parameters:
        window1: First binary image window
        window2: Second binary image window
        operator (str): Logical operation to perform ('AND', 'OR', 'XOR')
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the combined binary image
    """

    def gui(self) -> None:
        self.gui_reset()
        window1 = WindowSelector()
        window2 = WindowSelector()
        operator = ComboBox()
        operator.addItem("AND")
        operator.addItem("OR")
        operator.addItem("XOR")

        self.items.append({"name": "window1", "string": "Window 1", "object": window1})
        self.items.append({"name": "window2", "string": "Window 2", "object": window2})
        self.items.append(
            {"name": "operator", "string": "Operator", "object": operator}
        )

        super().gui()

    def __call__(
        self,
        window1: flika.window.Window,
        window2: flika.window.Window,
        operator: str,
        keepSourceWindow: bool = False,
    ) -> flika.window.Window | None:
        self.keepSourceWindow = keepSourceWindow
        g.m.statusBar().showMessage(f"Performing {self.__name__}...")

        if window1 is None or window2 is None:
            raise MissingWindowError(
                f"You cannot execute '{self.__name__}' without selecting a window first."
            )

        if window1.image.shape != window2.image.shape:
            g.m.statusBar().showMessage(
                "The two windows have images of different shapes. They could not be combined"
            )
            return None

        if operator == "AND":
            self.newtif = np.logical_and(window1.image, window2.image)
        elif operator == "OR":
            self.newtif = np.logical_or(window1.image, window2.image)
        elif operator == "XOR":
            self.newtif = np.logical_xor(window1.image, window2.image)

        self.oldwindow = window1
        self.oldname = window1.name
        self.newname = f"{self.oldname} - Logical {operator}"

        if not keepSourceWindow:
            window2.close()

        g.m.statusBar().showMessage(f"Finished with {self.__name__}.")
        return self.end()


logically_combine = Logically_combine()


class Remove_small_blobs(BaseProcess):
    """Removes small connected regions from binary images.

    Parameters:
        rank (int): Number of dimensions to consider for connectivity (2 or 3)
        value (int): Minimum size (in pixels) for a region to be kept
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with small regions removed
    """

    def gui(self) -> None:
        self.gui_reset()
        rank = QtWidgets.QSpinBox()
        rank.setRange(2, 3)

        value = QtWidgets.QSpinBox()
        value.setRange(1, 100000)

        self.items.append(
            {"name": "rank", "string": "Number of Dimensions", "object": rank}
        )
        self.items.append({"name": "value", "string": "Value", "object": value})

        super().gui()

    def __call__(
        self, rank: int, value: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)

        if self.tif.dtype == np.float16:
            g.alert("remove_small_blobs() does not support float16 type arrays")
            return None

        oldshape = self.tif.shape
        newtif = np.zeros_like(self.tif, dtype=bool)

        if self.oldwindow.nDims == 2:
            newtif = remove_small_objects(self.tif.astype(bool), value, connectivity=2)
        elif self.oldwindow.nDims == 3:
            if rank == 2:
                for i in range(len(self.tif)):
                    newtif[i] = remove_small_objects(
                        self.tif[i].astype(bool), value, connectivity=2
                    )
            elif rank == 3:
                newtif = remove_small_objects(
                    self.tif.astype(bool), value, connectivity=2
                )

        self.newtif = newtif
        self.newname = f"{self.oldname} - Removed Blobs {value}"
        return self.end()

    def get_init_settings_dict(self) -> dict[str, int]:
        s = {}
        s["rank"] = 2
        s["value"] = 1
        return s


remove_small_blobs = Remove_small_blobs()


class Binary_Dilation(BaseProcess):
    """Performs binary dilation on a binary image to expand regions.

    Parameters:
        rank (int): Number of dimensions to consider (2 or 3)
        connectivity (int): Connectivity pattern (1 to rank)
        iterations (int): Number of times to repeat the dilation
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the dilated binary image
    """

    def gui(self) -> None:
        self.gui_reset()
        rank = QtWidgets.QSpinBox()
        rank.setRange(2, 3)

        connectivity = QtWidgets.QSpinBox()
        connectivity.setRange(1, 3)

        iterations = QtWidgets.QSpinBox()
        iterations.setRange(1, 100)

        self.items.append(
            {"name": "rank", "string": "Number of Dimensions", "object": rank}
        )
        self.items.append(
            {"name": "connectivity", "string": "Connectivity", "object": connectivity}
        )
        self.items.append(
            {"name": "iterations", "string": "Iterations", "object": iterations}
        )

        super().gui()

    def __call__(
        self,
        rank: int,
        connectivity: int,
        iterations: int,
        keepSourceWindow: bool = False,
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)

        if self.tif.dtype == np.float16:
            g.alert("binary_dilation does not support float16 type arrays")
            return None

        if len(self.tif.shape) == 3 and rank == 2:
            s = scipy.ndimage.generate_binary_structure(3, connectivity)
            s[0] = False
            s[2] = False
        else:
            s = scipy.ndimage.generate_binary_structure(rank, connectivity)

        self.newtif = scipy.ndimage.binary_dilation(self.tif, s, iterations)
        self.newtif = self.newtif.astype(np.uint8)
        self.newname = f"{self.oldname} - Dilated"
        return self.end()


binary_dilation = Binary_Dilation()


class Binary_Erosion(BaseProcess):
    """Performs binary erosion on a binary image to shrink regions.

    Parameters:
        rank (int): Number of dimensions to consider (2 or 3)
        connectivity (int): Connectivity pattern (1 to rank)
        iterations (int): Number of times to repeat the erosion
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the eroded binary image
    """

    def gui(self) -> None:
        self.gui_reset()
        rank = QtWidgets.QSpinBox()
        rank.setRange(2, 3)

        connectivity = QtWidgets.QSpinBox()
        connectivity.setRange(1, 3)

        iterations = QtWidgets.QSpinBox()
        iterations.setRange(1, 100)

        self.items.append(
            {"name": "rank", "string": "Number of Dimensions", "object": rank}
        )
        self.items.append(
            {"name": "connectivity", "string": "Connectivity", "object": connectivity}
        )
        self.items.append(
            {"name": "iterations", "string": "Iterations", "object": iterations}
        )

        super().gui()

    def __call__(
        self,
        rank: int,
        connectivity: int,
        iterations: int,
        keepSourceWindow: bool = False,
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)

        if self.tif.dtype == np.float16:
            g.alert(
                "Binary Erosion does not work on float16 images. Change the data type to use this function."
            )
            return None

        if len(self.tif.shape) == 3 and rank == 2:
            s = scipy.ndimage.generate_binary_structure(3, connectivity)
            s[0] = False
            s[2] = False
        else:
            s = scipy.ndimage.generate_binary_structure(rank, connectivity)

        self.newtif = scipy.ndimage.binary_erosion(self.tif, s, iterations)
        self.newtif = self.newtif.astype(np.uint8)
        self.newname = f"{self.oldname} - Eroded"
        return self.end()


binary_erosion = Binary_Erosion()


class Generate_ROIs(BaseProcess):
    """Generates Region of Interest (ROI) objects from binary image clusters.

    Parameters:
        level (float): Contour level for finding boundaries (0-1)
        minDensity (int): Minimum pixel count for a region to be considered
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the generated ROIs
    """

    def __init__(self):
        super().__init__()
        self.ROIs: list = []

    def gui(self) -> None:
        self.gui_reset()
        level = SliderLabel(2)
        level.setRange(0, 1)
        level.setValue(0.5)

        minDensity = QtWidgets.QSpinBox()
        minDensity.setRange(4, 1000)

        preview = CheckBox()
        preview.setChecked(True)

        self.items.append({"name": "level", "string": "Contour Level", "object": level})
        self.items.append(
            {"name": "minDensity", "string": "Minimum Density", "object": minDensity}
        )
        self.items.append({"name": "preview", "string": "Preview", "object": preview})

        self.ROIs = []
        super().gui()
        self.ui.rejected.connect(self.removeROIs)

    def removeROIs(self) -> None:
        for roi in self.ROIs:
            roi.cancel()
        self.ROIs = []

    def __call__(
        self, level: float, minDensity: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)

        if self.tif.dtype == np.float16:
            g.alert("generate_rois does not support float16 type arrays")
            return None

        if not np.all((self.tif == 0) | (self.tif == 1)):
            g.alert("The current image is not a binary image. Threshold first")
            return None

        for roi in self.ROIs:
            roi.cancel()
        self.ROIs = []

        im = g.win.image if g.win.image.ndim == 2 else g.win.image[g.win.currentIndex]
        im = scipy.ndimage.binary_closing(im)
        thresholded_image = np.squeeze(im)
        labelled = measure.label(thresholded_image)
        ROIs = []

        for i in range(1, np.max(labelled) + 1):
            if np.sum(labelled == i) >= minDensity:
                im = scipy.ndimage.binary_dilation(
                    scipy.ndimage.binary_closing(labelled == i)
                )
                outline_coords = measure.find_contours(im, level)
                if len(outline_coords) == 0:
                    continue
                outline_coords = outline_coords[0]
                new_roi = makeROI("freehand", outline_coords)
                ROIs.append(new_roi)

        self.newtif = self.tif.copy()
        self.newname = f"{self.oldname} - ROIs Generated"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return

        win = g.win
        im = win.image if win.image.ndim == 2 else win.image[win.currentIndex]

        if not np.all((im == 0) | (im == 1)):
            g.alert("The current image is not a binary image. Threshold first")
            return None

        im = scipy.ndimage.binary_closing(im)
        level = self.getValue("level")
        minDensity = self.getValue("minDensity")
        thresholded_image = np.squeeze(im)
        labelled = measure.label(thresholded_image)

        for roi in self.ROIs:
            roi.cancel()
        self.ROIs = []

        for i in range(1, np.max(labelled) + 1):
            QtWidgets.QApplication.processEvents()
            if np.sum(labelled == i) >= minDensity:
                im = scipy.ndimage.binary_dilation(
                    scipy.ndimage.binary_closing(labelled == i)
                )
                outline_coords = measure.find_contours(im, level)
                if len(outline_coords) == 0:
                    continue
                outline_coords = outline_coords[0]
                self.ROIs.append(
                    ROI_Drawing(
                        win, outline_coords[0][0], outline_coords[0][1], "freehand"
                    )
                )
                for p in outline_coords[1:]:
                    self.ROIs[-1].extend(p[0], p[1])
                    QtWidgets.QApplication.processEvents()


generate_rois = Generate_ROIs()


class Analyze_Particles(BaseProcess):
    """Analyzes connected components in a binary image.

    Labels each connected region and computes properties such as area,
    centroid, bounding box, eccentricity, and mean intensity.

    Parameters:
        min_area (int): Minimum area (in pixels) for a particle to be kept
        max_area (int): Maximum area (0 means no limit)
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the labeled particle image
    """

    def gui(self) -> None:
        self.gui_reset()
        min_area = QtWidgets.QSpinBox()
        min_area.setRange(1, 100000)
        min_area.setValue(1)

        max_area = QtWidgets.QSpinBox()
        max_area.setRange(0, 10000000)
        max_area.setValue(0)

        self.items.append(
            {"name": "min_area", "string": "Min Area (pixels)", "object": min_area}
        )
        self.items.append(
            {"name": "max_area", "string": "Max Area (0=no limit)", "object": max_area}
        )
        super().gui()

    def __call__(
        self,
        min_area: int = 1,
        max_area: int = 0,
        keepSourceWindow: bool = False,
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        tif = self.tif

        if tif.ndim == 2:
            labelled, particles = self._analyze_frame(tif, min_area, max_area, None)
            self.newtif = labelled.astype(np.float64)
        elif tif.ndim == 3:
            use_parallel = g.settings.get("multiprocessing", True) and tif.shape[0] > 4
            if use_parallel:
                from concurrent.futures import ThreadPoolExecutor

                n_workers = g.settings.get("nCores", 4)

                def analyze_t(t):
                    return t, self._analyze_frame(
                        tif[t], min_area, max_area, tif[t]
                    )

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    frame_results = list(
                        executor.map(analyze_t, range(tif.shape[0]))
                    )
                all_particles = []
                result = np.zeros_like(tif, dtype=np.float64)
                for t, (labelled, particles) in frame_results:
                    result[t] = labelled.astype(np.float64)
                    for p in particles:
                        p["frame"] = t
                    all_particles.extend(particles)
            else:
                all_particles = []
                result = np.zeros_like(tif, dtype=np.float64)
                for t in range(tif.shape[0]):
                    labelled, particles = self._analyze_frame(
                        tif[t], min_area, max_area, tif[t]
                    )
                    result[t] = labelled.astype(np.float64)
                    for p in particles:
                        p["frame"] = t
                    all_particles.extend(particles)
            self.newtif = result
            particles = all_particles
        else:
            g.alert("Analyze particles requires 2D or 3D binary images")
            return None

        self.newname = f"{self.oldname} - Particles"
        w = self.end()
        if w is not None:
            w.metadata["particles"] = particles
            n = len(particles)
            g.m.statusBar().showMessage(f"Found {n} particles")
        return w

    @staticmethod
    def _analyze_frame(
        binary_frame: np.ndarray,
        min_area: int,
        max_area: int,
        intensity_frame: np.ndarray | None,
    ) -> tuple[np.ndarray, list[dict]]:
        labelled = measure.label(binary_frame.astype(bool))
        regions = measure.regionprops(labelled, intensity_image=intensity_frame)
        particles = []
        filtered_label = np.zeros_like(labelled)
        new_id = 1

        for props in regions:
            area = props.area
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue
            p = {
                "label": new_id,
                "area": int(area),
                "centroid": tuple(float(c) for c in props.centroid),
                "bbox": props.bbox,
                "eccentricity": (
                    float(props.eccentricity)
                    if hasattr(props, "eccentricity")
                    else 0.0
                ),
            }
            if intensity_frame is not None:
                p["mean_intensity"] = float(props.mean_intensity)
            filtered_label[labelled == props.label] = new_id
            particles.append(p)
            new_id += 1

        return filtered_label, particles


analyze_particles = Analyze_Particles()


@per_plane
def _grayscale_opening_impl(
    tif: np.ndarray, size: int
) -> np.ndarray:
    if tif.ndim == 2:
        return scipy.ndimage.grey_opening(tif, size=(size, size))
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = scipy.ndimage.grey_opening(tif[i], size=(size, size))
        return result


class Grayscale_Opening(BaseProcess):
    """Performs a grayscale morphological opening.

    Uses scipy.ndimage.grey_opening to remove bright structures smaller
    than the footprint.

    Parameters:
        size (int): Size of the 2D footprint (must be odd)
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the opened image
    """

    def gui(self) -> None:
        self.gui_reset()
        size = BlocksizeSlider(0)
        size.setRange(3, 51)
        size.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({"name": "size", "string": "Size", "object": size})
        self.items.append({"name": "preview", "string": "Preview", "object": preview})
        super().gui()
        self.preview()

    def __call__(
        self, size: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _grayscale_opening_impl(np.copy(self.tif), size)
        self.newname = f"{self.oldname} - Grayscale Opening {size}"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return
        win = g.win
        size = self.getValue("size")
        preview = self.getValue("preview")
        if preview:
            if win.nDims == 3:
                testimage = np.copy(win.image[win.currentIndex])
            elif win.nDims == 2:
                testimage = np.copy(win.image)
            testimage = scipy.ndimage.grey_opening(testimage, size=(size, size))
            win.imageview.setImage(testimage, autoLevels=False)
        else:
            win.reset()


grayscale_opening = Grayscale_Opening()


@per_plane
def _grayscale_closing_impl(
    tif: np.ndarray, size: int
) -> np.ndarray:
    if tif.ndim == 2:
        return scipy.ndimage.grey_closing(tif, size=(size, size))
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = scipy.ndimage.grey_closing(tif[i], size=(size, size))
        return result


class Grayscale_Closing(BaseProcess):
    """Performs a grayscale morphological closing.

    Uses scipy.ndimage.grey_closing to fill dark structures smaller
    than the footprint.

    Parameters:
        size (int): Size of the 2D footprint (must be odd)
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the closed image
    """

    def gui(self) -> None:
        self.gui_reset()
        size = BlocksizeSlider(0)
        size.setRange(3, 51)
        size.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({"name": "size", "string": "Size", "object": size})
        self.items.append({"name": "preview", "string": "Preview", "object": preview})
        super().gui()
        self.preview()

    def __call__(
        self, size: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _grayscale_closing_impl(np.copy(self.tif), size)
        self.newname = f"{self.oldname} - Grayscale Closing {size}"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return
        win = g.win
        size = self.getValue("size")
        preview = self.getValue("preview")
        if preview:
            if win.nDims == 3:
                testimage = np.copy(win.image[win.currentIndex])
            elif win.nDims == 2:
                testimage = np.copy(win.image)
            testimage = scipy.ndimage.grey_closing(testimage, size=(size, size))
            win.imageview.setImage(testimage, autoLevels=False)
        else:
            win.reset()


grayscale_closing = Grayscale_Closing()


@per_plane
def _morphological_gradient_impl(
    tif: np.ndarray, size: int
) -> np.ndarray:
    if tif.ndim == 2:
        return scipy.ndimage.morphological_gradient(tif, size=(size, size))
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = scipy.ndimage.morphological_gradient(
                tif[i], size=(size, size)
            )
        return result


class Morphological_Gradient(BaseProcess):
    """Computes the morphological gradient (dilation minus erosion).

    Uses scipy.ndimage.morphological_gradient to highlight edges.

    Parameters:
        size (int): Size of the 2D footprint (must be odd)
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the gradient image
    """

    def gui(self) -> None:
        self.gui_reset()
        size = BlocksizeSlider(0)
        size.setRange(3, 51)
        size.setValue(3)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({"name": "size", "string": "Size", "object": size})
        self.items.append({"name": "preview", "string": "Preview", "object": preview})
        super().gui()
        self.preview()

    def __call__(
        self, size: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _morphological_gradient_impl(np.copy(self.tif), size)
        self.newname = f"{self.oldname} - Morphological Gradient {size}"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return
        win = g.win
        size = self.getValue("size")
        preview = self.getValue("preview")
        if preview:
            if win.nDims == 3:
                testimage = np.copy(win.image[win.currentIndex])
            elif win.nDims == 2:
                testimage = np.copy(win.image)
            testimage = scipy.ndimage.morphological_gradient(
                testimage, size=(size, size)
            )
            win.imageview.setImage(testimage, autoLevels=False)
        else:
            win.reset()


morphological_gradient = Morphological_Gradient()


@per_plane
def _h_maxima_impl(tif: np.ndarray, h: float) -> np.ndarray:
    from skimage.morphology import h_maxima as _h_maxima

    if tif.ndim == 2:
        return _h_maxima(tif, h)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = _h_maxima(tif[i], h)
        return result


class H_Maxima(BaseProcess):
    """Determines all maxima of the image with depth >= h.

    Uses skimage.morphology.h_maxima.

    Parameters:
        h (float): Minimum depth of maxima
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the h-maxima image
    """

    def gui(self) -> None:
        self.gui_reset()
        h = SliderLabel(2)
        h.setRange(0.1, 1000)
        h.setValue(1)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append({"name": "h", "string": "H", "object": h})
        self.items.append({"name": "preview", "string": "Preview", "object": preview})
        super().gui()
        self.preview()

    def __call__(
        self, h: float, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _h_maxima_impl(np.copy(self.tif), h)
        self.newname = f"{self.oldname} - H Maxima {h}"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return
        win = g.win
        from skimage.morphology import h_maxima as _h_maxima

        h = self.getValue("h")
        preview = self.getValue("preview")
        if preview:
            if win.nDims == 3:
                testimage = np.copy(win.image[win.currentIndex])
            elif win.nDims == 2:
                testimage = np.copy(win.image)
            testimage = _h_maxima(testimage, h)
            win.imageview.setImage(testimage, autoLevels=False)
        else:
            win.reset()


h_maxima = H_Maxima()


@per_plane
def _h_minima_impl(tif: np.ndarray, h: float) -> np.ndarray:
    from skimage.morphology import h_minima as _h_minima

    if tif.ndim == 2:
        return _h_minima(tif, h)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = _h_minima(tif[i], h)
        return result


class H_Minima(BaseProcess):
    """Determines all minima of the image with depth >= h.

    Uses skimage.morphology.h_minima.

    Parameters:
        h (float): Minimum depth of minima
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the h-minima image
    """

    def gui(self) -> None:
        self.gui_reset()
        h = SliderLabel(2)
        h.setRange(0.1, 1000)
        h.setValue(1)
        self.items.append({"name": "h", "string": "H", "object": h})
        super().gui()

    def __call__(
        self, h: float, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _h_minima_impl(np.copy(self.tif), h)
        self.newname = f"{self.oldname} - H Minima {h}"
        return self.end()


h_minima = H_Minima()


@per_plane
def _area_opening_impl(
    tif: np.ndarray, area_threshold: int
) -> np.ndarray:
    from skimage.morphology import area_opening as _area_opening

    if tif.ndim == 2:
        return _area_opening(tif, area_threshold)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = _area_opening(tif[i], area_threshold)
        return result


class Area_Opening(BaseProcess):
    """Performs an area opening on a grayscale image.

    Removes all bright structures smaller than area_threshold using
    skimage.morphology.area_opening.

    Parameters:
        area_threshold (int): Minimum area (in pixels) of structures to keep
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the area-opened image
    """

    def gui(self) -> None:
        self.gui_reset()
        area_threshold = SliderLabel(0)
        area_threshold.setRange(1, 100000)
        area_threshold.setValue(64)
        self.items.append(
            {
                "name": "area_threshold",
                "string": "Area Threshold",
                "object": area_threshold,
            }
        )
        super().gui()

    def __call__(
        self, area_threshold: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _area_opening_impl(np.copy(self.tif), area_threshold)
        self.newname = f"{self.oldname} - Area Opening {area_threshold}"
        return self.end()


area_opening = Area_Opening()


@per_plane
def _area_closing_impl(
    tif: np.ndarray, area_threshold: int
) -> np.ndarray:
    from skimage.morphology import area_closing as _area_closing

    if tif.ndim == 2:
        return _area_closing(tif, area_threshold)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape)
        for i in range(len(tif)):
            result[i] = _area_closing(tif[i], area_threshold)
        return result


class Area_Closing(BaseProcess):
    """Performs an area closing on a grayscale image.

    Removes all dark structures smaller than area_threshold using
    skimage.morphology.area_closing.

    Parameters:
        area_threshold (int): Minimum area (in pixels) of structures to fill
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the area-closed image
    """

    def gui(self) -> None:
        self.gui_reset()
        area_threshold = SliderLabel(0)
        area_threshold.setRange(1, 100000)
        area_threshold.setValue(64)
        self.items.append(
            {
                "name": "area_threshold",
                "string": "Area Threshold",
                "object": area_threshold,
            }
        )
        super().gui()

    def __call__(
        self, area_threshold: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _area_closing_impl(np.copy(self.tif), area_threshold)
        self.newname = f"{self.oldname} - Area Closing {area_threshold}"
        return self.end()


area_closing = Area_Closing()


@per_plane
def _remove_small_holes_impl(
    tif: np.ndarray, max_size: int
) -> np.ndarray:
    from skimage.morphology import remove_small_holes as _remove_small_holes

    if tif.ndim == 2:
        return _remove_small_holes(tif.astype(bool), max_size).astype(np.uint8)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.uint8)
        for i in range(len(tif)):
            result[i] = _remove_small_holes(tif[i].astype(bool), max_size).astype(
                np.uint8
            )
        return result


class Remove_Small_Holes(BaseProcess):
    """Removes small holes from a binary image.

    Uses skimage.morphology.remove_small_holes.

    Parameters:
        max_size (int): Maximum area (in pixels) of holes to remove
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with small holes removed
    """

    def gui(self) -> None:
        self.gui_reset()
        max_size = SliderLabel(0)
        max_size.setRange(1, 100000)
        max_size.setValue(64)
        self.items.append(
            {"name": "max_size", "string": "Max Hole Size", "object": max_size}
        )
        super().gui()

    def __call__(
        self, max_size: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _remove_small_holes_impl(np.copy(self.tif), max_size)
        self.newname = f"{self.oldname} - Remove Small Holes {max_size}"
        return self.end()


remove_small_holes = Remove_Small_Holes()


class Flood_Fill_Process(BaseProcess):
    """Performs a flood fill starting from the current cursor position.

    Uses skimage.morphology.flood_fill.

    Parameters:
        tolerance (float): Tolerance for the fill (pixels within this range
            of the seed value are filled)
        new_value (float): The value to fill with
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the flood-filled image
    """

    def gui(self) -> None:
        self.gui_reset()
        tolerance = SliderLabel(2)
        tolerance.setRange(0, 1000)
        tolerance.setValue(10)
        new_value = SliderLabel(2)
        new_value.setRange(-10000, 10000)
        new_value.setValue(0)
        self.items.append(
            {"name": "tolerance", "string": "Tolerance", "object": tolerance}
        )
        self.items.append(
            {"name": "new_value", "string": "New Value", "object": new_value}
        )
        super().gui()

    def __call__(
        self,
        tolerance: float,
        new_value: float,
        keepSourceWindow: bool = False,
    ) -> flika.window.Window | None:
        from skimage.morphology import flood_fill as _flood_fill

        self.start(keepSourceWindow)
        win = self.oldwindow
        # Get seed point from the current cursor position
        view = win.imageview.getView()
        mouse_point = view.mapSceneToView(
            view.mapFromGlobal(QtGui.QCursor.pos())
        )
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        tif = np.copy(self.tif)
        if tif.ndim == 2:
            y = np.clip(y, 0, tif.shape[0] - 1)
            x = np.clip(x, 0, tif.shape[1] - 1)
            tif = _flood_fill(tif, (y, x), new_value, tolerance=tolerance)
        elif tif.ndim == 3:
            idx = win.currentIndex
            y = np.clip(y, 0, tif.shape[1] - 1)
            x = np.clip(x, 0, tif.shape[2] - 1)
            tif[idx] = _flood_fill(
                tif[idx], (y, x), new_value, tolerance=tolerance
            )
        self.newtif = tif
        self.newname = f"{self.oldname} - Flood Fill"
        return self.end()


flood_fill_process = Flood_Fill_Process()


@per_plane
def _hysteresis_threshold_impl(
    tif: np.ndarray, low: float, high: float
) -> np.ndarray:
    from skimage.filters import apply_hysteresis_threshold

    if tif.ndim == 2:
        return apply_hysteresis_threshold(tif, low, high).astype(np.uint8)
    elif tif.ndim == 3:
        result = np.zeros(tif.shape, dtype=np.uint8)
        for i in range(len(tif)):
            result[i] = apply_hysteresis_threshold(tif[i], low, high).astype(
                np.uint8
            )
        return result


class Hysteresis_Threshold(BaseProcess):
    """Applies hysteresis thresholding.

    Uses skimage.filters.apply_hysteresis_threshold. Pixels above ``high``
    are marked True; pixels between ``low`` and ``high`` are marked True
    only if connected to pixels above ``high``.

    Parameters:
        low (float): Lower threshold
        high (float): Upper threshold
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the hysteresis-thresholded image
    """

    def gui(self) -> None:
        self.gui_reset()
        low = SliderLabel(2)
        high = SliderLabel(2)
        if g.win is not None:
            image = g.win.image
            low.setRange(np.min(image), np.max(image))
            low.setValue(np.min(image))
            high.setRange(np.min(image), np.max(image))
            high.setValue(np.mean(image))
        else:
            low.setRange(0, 10000)
            low.setValue(0)
            high.setRange(0, 10000)
            high.setValue(100)
        preview = CheckBox()
        preview.setChecked(True)
        self.items.append(
            {"name": "low", "string": "Low Threshold", "object": low}
        )
        self.items.append(
            {"name": "high", "string": "High Threshold", "object": high}
        )
        self.items.append({"name": "preview", "string": "Preview", "object": preview})
        super().gui()
        self.preview()

    def __call__(
        self, low: float, high: float, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        self.start(keepSourceWindow)
        self.newtif = _hysteresis_threshold_impl(np.copy(self.tif), low, high)
        self.newname = f"{self.oldname} - Hysteresis Threshold"
        return self.end()

    def preview(self) -> None:
        if g.win is None or g.win.closed:
            return
        win = g.win
        from skimage.filters import apply_hysteresis_threshold

        low = self.getValue("low")
        high = self.getValue("high")
        preview = self.getValue("preview")
        if preview:
            if win.nDims == 3:
                testimage = np.copy(win.image[win.currentIndex])
            elif win.nDims == 2:
                testimage = np.copy(win.image)
            testimage = apply_hysteresis_threshold(testimage, low, high).astype(
                np.uint8
            )
            win.imageview.setImage(testimage, autoLevels=False)
            win.imageview.setLevels(-0.1, 1.1)
        else:
            win.reset()


hysteresis_threshold = Hysteresis_Threshold()


class Multi_Otsu_Threshold(BaseProcess):
    """Computes multi-Otsu thresholds and returns a labeled image.

    Each pixel is assigned to one of the classes using
    skimage.filters.threshold_multiotsu.

    Parameters:
        classes (int): Number of classes to threshold into (2-5)
        keepSourceWindow (bool): If True, don't close the source window

    Returns:
        newWindow: A new window with the class-labeled image
    """

    def gui(self) -> None:
        self.gui_reset()
        classes = QtWidgets.QSpinBox()
        classes.setRange(2, 5)
        classes.setValue(3)
        self.items.append(
            {"name": "classes", "string": "Number of Classes", "object": classes}
        )
        super().gui()

    def __call__(
        self, classes: int, keepSourceWindow: bool = False
    ) -> flika.window.Window | None:
        from skimage.filters import threshold_multiotsu

        self.start(keepSourceWindow)
        tif = np.copy(self.tif)
        if tif.ndim == 2:
            thresholds = threshold_multiotsu(tif, classes=classes)
            self.newtif = np.digitize(tif, bins=thresholds).astype(np.float64)
        elif tif.ndim == 3:
            result = np.zeros(tif.shape, dtype=np.float64)
            for i in range(len(tif)):
                thresholds = threshold_multiotsu(tif[i], classes=classes)
                result[i] = np.digitize(tif[i], bins=thresholds).astype(np.float64)
            self.newtif = result
        self.newname = f"{self.oldname} - Multi Otsu {classes} classes"
        return self.end()


multi_otsu_threshold = Multi_Otsu_Threshold()
