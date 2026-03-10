# -*- coding: utf-8 -*-
"""Channel alignment for dual-channel microscopy (e.g. beam splitter setups).

Provides geometric transforms, background subtraction, photobleach
correction, and auto-alignment via phase cross-correlation.
"""
import numpy as np
import scipy.ndimage
from skimage.registration import phase_cross_correlation
from qtpy import QtWidgets, QtCore, QtGui
import flika.global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess_noPriorWindow
from flika.utils.custom_widgets import SliderLabel, CheckBox, ComboBox, WindowSelector, MissingWindowError

__all__ = ['channel_alignment']


class Channel_Alignment(BaseProcess_noPriorWindow):
    """channel_alignment(red_window, green_window, x_shift=0, y_shift=0, rotation=0.0, scale_factor=1.0, background_method='None', background_radius=50, photobleach_correction='None', normalize_intensity=False)

    Align two image channels (e.g. from a beam splitter) and produce an
    RGB overlay composite.  The red channel is held fixed while geometric
    transforms (translation, rotation, scale) are applied to the green
    channel.

    Parameters:
        red_window (Window): Window containing the red channel image.
        green_window (Window): Window containing the green channel image.
        x_shift (int): Horizontal translation of green channel in pixels.
        y_shift (int): Vertical translation of green channel in pixels.
        rotation (float): Rotation of green channel in degrees.
        scale_factor (float): Scale factor for green channel.
        background_method (str): 'None', 'Rolling Ball', 'Gaussian', or 'Percentile'.
        background_radius (int): Radius / sigma for background subtraction.
        photobleach_correction (str): 'None', 'Exponential', or 'Histogram'.
        normalize_intensity (bool): Scale both channels to [0, 1] range.
    Returns:
        Window: New window containing the RGB composite overlay.
    """

    def __init__(self):
        super().__init__()
        self._preview_window = None

    def gui(self):
        self.gui_reset()
        red_window = WindowSelector()
        green_window = WindowSelector()
        x_shift = SliderLabel(decimals=0)
        x_shift.setRange(-1000, 1000)
        x_shift.setValue(0)
        y_shift = SliderLabel(decimals=0)
        y_shift.setRange(-1000, 1000)
        y_shift.setValue(0)
        rotation = SliderLabel(decimals=2)
        rotation.setRange(-180.0, 180.0)
        rotation.setValue(0.0)
        scale_factor = SliderLabel(decimals=3)
        scale_factor.setRange(0.5, 2.0)
        scale_factor.setValue(1.0)
        background_method = ComboBox()
        background_method.addItems(['None', 'Rolling Ball', 'Gaussian', 'Percentile'])
        background_radius = SliderLabel(decimals=0)
        background_radius.setRange(1, 200)
        background_radius.setValue(50)
        photobleach_correction = ComboBox()
        photobleach_correction.addItems(['None', 'Exponential', 'Histogram'])
        normalize_intensity = CheckBox()
        normalize_intensity.setChecked(False)

        self.items.append({'name': 'red_window', 'string': 'Red Channel', 'object': red_window})
        self.items.append({'name': 'green_window', 'string': 'Green Channel', 'object': green_window})
        self.items.append({'name': 'x_shift', 'string': 'X Shift', 'object': x_shift})
        self.items.append({'name': 'y_shift', 'string': 'Y Shift', 'object': y_shift})
        self.items.append({'name': 'rotation', 'string': 'Rotation (deg)', 'object': rotation})
        self.items.append({'name': 'scale_factor', 'string': 'Scale Factor', 'object': scale_factor})
        self.items.append({'name': 'background_method', 'string': 'Background Method', 'object': background_method})
        self.items.append({'name': 'background_radius', 'string': 'Background Radius', 'object': background_radius})
        self.items.append({'name': 'photobleach_correction', 'string': 'Photobleach Correction', 'object': photobleach_correction})
        self.items.append({'name': 'normalize_intensity', 'string': 'Normalize Intensity', 'object': normalize_intensity})

        super().gui()

        self.auto_align_button = QtWidgets.QPushButton('Auto Align')
        self.auto_align_button.clicked.connect(self.auto_align)
        self.ui.formlayout.addRow('', self.auto_align_button)

        self.preview_checkbox = CheckBox()
        self.preview_checkbox.setChecked(False)
        self.preview_checkbox.stateChanged.connect(lambda: self.preview())
        self.ui.formlayout.addRow('Preview', self.preview_checkbox)

        self._x_shift_widget = x_shift
        self._y_shift_widget = y_shift
        self._rotation_widget = rotation
        self._scale_factor_widget = scale_factor

        self.ui.keyPressEvent = self.keyPressed

    def __call__(self, red_window, green_window, x_shift=0, y_shift=0,
                 rotation=0.0, scale_factor=1.0, background_method='None',
                 background_radius=50, photobleach_correction='None',
                 normalize_intensity=False):
        if red_window is None or green_window is None:
            raise MissingWindowError("Two windows must be selected for channel alignment.")

        self.start()

        img_red = red_window.image.astype(np.float64)
        img_green = green_window.image.astype(np.float64)

        if background_method != 'None':
            img_red = self.subtract_background(img_red, background_method, background_radius)
            img_green = self.subtract_background(img_green, background_method, background_radius)

        if photobleach_correction != 'None':
            img_red = self.correct_photobleaching(img_red, photobleach_correction)
            img_green = self.correct_photobleaching(img_green, photobleach_correction)

        img_green = self.apply_transforms(img_green, x_shift, y_shift, rotation, scale_factor)

        if img_red.ndim == 2:
            min_h = min(img_red.shape[0], img_green.shape[0])
            min_w = min(img_red.shape[1], img_green.shape[1])
            img_red = img_red[:min_h, :min_w]
            img_green = img_green[:min_h, :min_w]
        elif img_red.ndim == 3:
            min_t = min(img_red.shape[0], img_green.shape[0])
            min_h = min(img_red.shape[1], img_green.shape[1])
            min_w = min(img_red.shape[2], img_green.shape[2])
            img_red = img_red[:min_t, :min_h, :min_w]
            img_green = img_green[:min_t, :min_h, :min_w]

        if normalize_intensity:
            img_red, img_green = self.normalize_intensities(img_red, img_green)

        zeros = np.zeros_like(img_red)
        rgb = np.stack([img_red, img_green, zeros], axis=-1)
        rgb = np.clip(rgb, 0, None)
        rgb_max = rgb.max()
        if rgb_max > 0:
            rgb = (rgb / rgb_max * 255).astype(np.float32)

        self.newtif = rgb
        self.newname = 'Aligned: {} + {}'.format(red_window.name, green_window.name)
        return self.end()

    @staticmethod
    def apply_transforms(image, x_shift, y_shift, rotation, scale_factor):
        """Apply geometric transforms (scale, rotate, translate) to an image."""
        is_3d = image.ndim == 3

        def _transform_2d(img):
            result = img.copy()
            if scale_factor != 1.0:
                result = scipy.ndimage.zoom(result, scale_factor, order=1)
                h_orig, w_orig = img.shape
                h_new, w_new = result.shape
                out = np.zeros_like(img)
                y_start_src = max(0, (h_new - h_orig) // 2)
                x_start_src = max(0, (w_new - w_orig) // 2)
                y_start_dst = max(0, (h_orig - h_new) // 2)
                x_start_dst = max(0, (w_orig - w_new) // 2)
                copy_h = min(h_orig - y_start_dst, h_new - y_start_src)
                copy_w = min(w_orig - x_start_dst, w_new - x_start_src)
                out[y_start_dst:y_start_dst + copy_h,
                    x_start_dst:x_start_dst + copy_w] = \
                    result[y_start_src:y_start_src + copy_h,
                           x_start_src:x_start_src + copy_w]
                result = out
            if rotation != 0.0:
                result = scipy.ndimage.rotate(result, rotation, reshape=False, order=1)
            if x_shift != 0 or y_shift != 0:
                result = scipy.ndimage.shift(result, [y_shift, x_shift], order=1)
            return result

        if is_3d:
            transformed = np.empty_like(image)
            for t in range(image.shape[0]):
                transformed[t] = _transform_2d(image[t])
            return transformed
        else:
            return _transform_2d(image)

    def auto_align(self):
        """Use phase cross-correlation to detect the optimal shift."""
        try:
            red_win = self.getValue('red_window')
            green_win = self.getValue('green_window')
        except (IndexError, AttributeError):
            if g.m is not None:
                g.m.statusBar().showMessage('Select both windows before auto-aligning.')
            return

        if red_win is None or green_win is None:
            if g.m is not None:
                g.m.statusBar().showMessage('Select both windows before auto-aligning.')
            return

        img_red = red_win.image.astype(np.float64)
        img_green = green_win.image.astype(np.float64)
        if img_red.ndim == 3:
            img_red = img_red[red_win.currentIndex]
        if img_green.ndim == 3:
            img_green = img_green[green_win.currentIndex]

        min_h = min(img_red.shape[0], img_green.shape[0])
        min_w = min(img_red.shape[1], img_green.shape[1])
        img_red = img_red[:min_h, :min_w]
        img_green = img_green[:min_h, :min_w]

        try:
            shift, _error, _phasediff = phase_cross_correlation(
                img_red, img_green, upsample_factor=10
            )
        except Exception as e:
            if g.m is not None:
                g.m.statusBar().showMessage('Auto-align failed: {}'.format(str(e)))
            return

        detected_y = int(round(shift[0]))
        detected_x = int(round(shift[1]))

        self._x_shift_widget.setValue(detected_x)
        self._y_shift_widget.setValue(detected_y)

        if g.m is not None:
            g.m.statusBar().showMessage(
                'Auto-align detected shift: x={}, y={}'.format(detected_x, detected_y))
        self.preview()

    @staticmethod
    def subtract_background(image, method, radius):
        """Subtract background from a 2-D or 3-D image."""
        def _subtract_2d(img):
            if method == 'Rolling Ball':
                bg = scipy.ndimage.grey_opening(img, size=(radius, radius))
                return img - bg
            elif method == 'Gaussian':
                bg = scipy.ndimage.gaussian_filter(img, sigma=radius)
                return img - bg
            elif method == 'Percentile':
                bg = np.percentile(img, radius)
                return img - bg
            return img

        if image.ndim == 3:
            result = np.empty_like(image)
            for t in range(image.shape[0]):
                result[t] = _subtract_2d(image[t])
            return result
        else:
            return _subtract_2d(image)

    @staticmethod
    def correct_photobleaching(image, method):
        """Correct photobleaching in a 3-D stack (T, H, W)."""
        if image.ndim != 3:
            return image
        n_frames = image.shape[0]
        if n_frames < 2:
            return image

        if method == 'Exponential':
            means = np.array([np.mean(image[t]) for t in range(n_frames)])
            t_axis = np.arange(n_frames, dtype=np.float64)
            valid = means > 0
            if np.sum(valid) < 2:
                return image
            log_means = np.log(means[valid])
            t_valid = t_axis[valid]
            coeffs = np.polyfit(t_valid, log_means, 1)
            decay_rate = -coeffs[0]
            corrected = image.copy()
            for t in range(n_frames):
                correction = np.exp(decay_rate * t)
                corrected[t] = image[t] * correction
            return corrected

        elif method == 'Histogram':
            try:
                from skimage.exposure import match_histograms
            except ImportError:
                return image
            reference = image[0]
            corrected = image.copy()
            for t in range(1, n_frames):
                corrected[t] = match_histograms(image[t], reference)
            return corrected

        return image

    @staticmethod
    def normalize_intensities(img1, img2):
        """Scale both images to [0, 1] using the 1st--99th percentile range."""
        def _norm(img):
            p_low = np.percentile(img, 1)
            p_high = np.percentile(img, 99)
            rng = p_high - p_low
            if rng == 0:
                rng = 1.0
            out = (img - p_low) / rng
            return np.clip(out, 0.0, 1.0)
        return _norm(img1), _norm(img2)

    def preview(self, *args):
        """Show an RGB overlay preview."""
        if not hasattr(self, 'preview_checkbox') or not self.preview_checkbox.isChecked():
            return
        try:
            red_win = self.getValue('red_window')
            green_win = self.getValue('green_window')
        except (IndexError, AttributeError):
            return
        if red_win is None or green_win is None:
            return

        x_shift = self.getValue('x_shift')
        y_shift = self.getValue('y_shift')
        rotation = self.getValue('rotation')
        scale_factor = self.getValue('scale_factor')
        background_method = self.getValue('background_method')
        background_radius = self.getValue('background_radius')
        normalize_intensity = self.getValue('normalize_intensity')

        img_red = red_win.image.astype(np.float64)
        img_green = green_win.image.astype(np.float64)
        if img_red.ndim == 3:
            img_red = img_red[red_win.currentIndex]
        if img_green.ndim == 3:
            img_green = img_green[green_win.currentIndex]

        if background_method != 'None':
            img_red = self.subtract_background(img_red, background_method, background_radius)
            img_green = self.subtract_background(img_green, background_method, background_radius)

        img_green = self.apply_transforms(img_green, x_shift, y_shift, rotation, scale_factor)

        min_h = min(img_red.shape[0], img_green.shape[0])
        min_w = min(img_red.shape[1], img_green.shape[1])
        img_red = img_red[:min_h, :min_w]
        img_green = img_green[:min_h, :min_w]

        if normalize_intensity:
            img_red, img_green = self.normalize_intensities(img_red, img_green)

        zeros = np.zeros_like(img_red)
        rgb = np.stack([img_red, img_green, zeros], axis=-1)
        rgb = np.clip(rgb, 0, None)
        rgb_max = rgb.max()
        if rgb_max > 0:
            rgb = (rgb / rgb_max * 255).astype(np.float32)

        try:
            if self._preview_window is not None and not self._preview_window.closed:
                self._preview_window.imageview.setImage(rgb)
                return
        except (AttributeError, RuntimeError):
            pass

        self._preview_window = Window(rgb, 'Alignment Preview')

    def keyPressed(self, event):
        """Handle keyboard shortcuts for fine adjustment."""
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self._x_shift_widget.setValue(self._x_shift_widget.value() - 1)
        elif key == QtCore.Qt.Key_Right:
            self._x_shift_widget.setValue(self._x_shift_widget.value() + 1)
        elif key == QtCore.Qt.Key_Up:
            self._y_shift_widget.setValue(self._y_shift_widget.value() - 1)
        elif key == QtCore.Qt.Key_Down:
            self._y_shift_widget.setValue(self._y_shift_widget.value() + 1)
        elif key == QtCore.Qt.Key_R:
            self._rotation_widget.setValue(self._rotation_widget.value() - 0.5)
        elif key == QtCore.Qt.Key_T:
            self._rotation_widget.setValue(self._rotation_widget.value() + 0.5)
        elif key == QtCore.Qt.Key_Plus or key == QtCore.Qt.Key_Equal:
            self._scale_factor_widget.setValue(self._scale_factor_widget.value() + 0.01)
        elif key == QtCore.Qt.Key_Minus:
            self._scale_factor_widget.setValue(self._scale_factor_widget.value() - 0.01)
        elif key == QtCore.Qt.Key_A:
            self.auto_align()
            return
        else:
            return
        self.preview()

    def get_init_settings_dict(self):
        return {
            'x_shift': 0,
            'y_shift': 0,
            'rotation': 0.0,
            'scale_factor': 1.0,
            'background_method': 'None',
            'background_radius': 50,
            'photobleach_correction': 'None',
            'normalize_intensity': False,
        }


channel_alignment = Channel_Alignment()
