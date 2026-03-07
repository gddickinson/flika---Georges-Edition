# -*- coding: utf-8 -*-
from ..logger import logger
logger.debug("Started 'reading process/overlay.py'")
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore, QtGui
from .. import global_vars as g
from ..utils.BaseProcess import (BaseProcess, SliderLabel, WindowSelector,
                                 MissingWindowError, CheckBox, ComboBox)

__all__ = ['time_stamp', 'background', 'scale_bar', 'grid_overlay',
           'counting_tool', 'bake_overlays_process']


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LOCATIONS = ['Lower Right', 'Lower Left', 'Upper Right', 'Upper Left']
_PRESET_COLORS = ['White', 'Black', 'Yellow', 'Green', 'Red', 'Cyan', 'Magenta', 'Custom...']
_BG_COLORS = ['None', 'Black', 'White', 'Semi-transparent Black',
              'Semi-transparent White', 'Custom...']


def _color_to_rgba(name, alpha=255):
    """Convert a preset name or hex string to an RGBA list."""
    presets = {
        'White': [255, 255, 255, alpha],
        'Black': [0, 0, 0, alpha],
        'Yellow': [255, 255, 0, alpha],
        'Green': [0, 255, 0, alpha],
        'Red': [255, 0, 0, alpha],
        'Cyan': [0, 255, 255, alpha],
        'Magenta': [255, 0, 255, alpha],
        'None': [0, 0, 0, 0],
        'Semi-transparent Black': [0, 0, 0, 128],
        'Semi-transparent White': [255, 255, 255, 128],
    }
    if name in presets:
        return presets[name]
    # Hex string
    try:
        c = QtGui.QColor(name)
        return [c.red(), c.green(), c.blue(), alpha]
    except Exception:
        return [255, 255, 255, alpha]


def _color_to_css(name):
    """Convert preset name to CSS color string."""
    mapping = {
        'White': 'white', 'Black': 'black', 'Yellow': 'yellow',
        'Green': '#00ff00', 'Red': 'red', 'Cyan': 'cyan',
        'Magenta': 'magenta', 'None': 'transparent',
        'Semi-transparent Black': 'rgba(0,0,0,0.5)',
        'Semi-transparent White': 'rgba(255,255,255,0.5)',
    }
    return mapping.get(name, name)


def _make_color_combo(items, current='White'):
    """Create a color combo with a Custom option that opens a color picker."""
    combo = ComboBox()
    for item in items:
        combo.addItem(item)
    idx = combo.findText(current)
    if idx >= 0:
        combo.setCurrentIndex(idx)

    def _on_change(text):
        if text == 'Custom...':
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor('white'), None, "Choose Color")
            if color.isValid():
                hex_str = color.name()
                # Insert before "Custom..."
                combo.blockSignals(True)
                combo.insertItem(combo.count() - 1, hex_str)
                combo.setCurrentIndex(combo.count() - 2)
                combo.blockSignals(False)

    combo.currentTextChanged.connect(_on_change)
    return combo


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

class Time_Stamp(BaseProcess):
    """time_stamp(framerate, show, font_size, color, bg_color, location,
                  bold, show_frame_number, format_string)

    Adds a time stamp overlay to a movie.

    Parameters:
        framerate (float): Frame rate in Hz (default from settings).
        show (bool): Show or hide the timestamp.
        font_size (int): Font size in points.
        color (str): Text color.
        bg_color (str): Background color behind text.
        location (str): Corner placement.
        bold (bool): Bold text.
        show_frame_number (bool): Also show frame number.
        format_string (str): Custom format (use {time} and {frame} placeholders).
    Returns:
        None
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        w = g.win
        if w is None:
            g.alert("No window is currently selected.")
            return

        # --- Auto-populate from settings ---
        default_interval = g.settings.get('frame_interval', 0.05) or 0.05
        default_framerate = 1.0 / default_interval if default_interval > 0 else 20.0
        if hasattr(w, 'framerate') and w.framerate is not None:
            default_framerate = w.framerate
        elif 'framerate' in g.settings.d and g.settings['framerate'] is not None:
            default_framerate = g.settings['framerate']

        framerate = QtWidgets.QDoubleSpinBox()
        framerate.setRange(0.0001, 1000000)
        framerate.setDecimals(4)
        framerate.setValue(default_framerate)

        font_size = QtWidgets.QSpinBox()
        font_size.setRange(4, 120)
        font_size.setValue(12)

        color = _make_color_combo(_PRESET_COLORS, 'White')
        bg_color = _make_color_combo(_BG_COLORS, 'None')

        location = ComboBox()
        for loc in _LOCATIONS:
            location.addItem(loc)
        location.setCurrentIndex(0)  # Upper Left default for timestamp

        bold = CheckBox()
        bold.setChecked(False)

        show_frame = CheckBox()
        show_frame.setChecked(False)

        format_string = QtWidgets.QLineEdit()
        format_string.setPlaceholderText("{time}")
        format_string.setToolTip(
            "Custom format. Placeholders: {time}, {frame}, {ms}, {s}, {min}, {hr}\n"
            "Leave empty for automatic formatting."
        )

        show = CheckBox()
        show.setChecked(True)

        # Restore existing properties if timestamp already on window
        if hasattr(w, '_ts_props') and w._ts_props:
            p = w._ts_props
            framerate.setValue(p.get('framerate', default_framerate))
            font_size.setValue(p.get('font_size', 12))
            idx = color.findText(p.get('color', 'White'))
            if idx >= 0:
                color.setCurrentIndex(idx)
            idx = bg_color.findText(p.get('bg_color', 'None'))
            if idx >= 0:
                bg_color.setCurrentIndex(idx)
            idx = location.findText(p.get('location', 'Upper Left'))
            if idx >= 0:
                location.setCurrentIndex(idx)
            bold.setChecked(p.get('bold', False))
            show_frame.setChecked(p.get('show_frame_number', False))
            format_string.setText(p.get('format_string', ''))

        self.items.append({'name': 'framerate', 'string': 'Frame Rate (Hz)', 'object': framerate})
        self.items.append({'name': 'font_size', 'string': 'Font Size (pt)', 'object': font_size})
        self.items.append({'name': 'color', 'string': 'Text Color', 'object': color})
        self.items.append({'name': 'bg_color', 'string': 'Background Color', 'object': bg_color})
        self.items.append({'name': 'location', 'string': 'Location', 'object': location})
        self.items.append({'name': 'bold', 'string': 'Bold', 'object': bold})
        self.items.append({'name': 'show_frame_number', 'string': 'Show Frame Number', 'object': show_frame})
        self.items.append({'name': 'format_string', 'string': 'Custom Format', 'object': format_string})
        self.items.append({'name': 'show', 'string': 'Show', 'object': show})
        super().gui()

    def __call__(self, framerate=20.0, show=True, font_size=12, color='White',
                 bg_color='None', location='Upper Left', bold=False,
                 show_frame_number=False, format_string='',
                 keepSourceWindow=None):
        w = g.win
        if w is None:
            return

        # Remove existing timestamp
        if hasattr(w, 'timeStampLabel') and w.timeStampLabel is not None:
            w.imageview.view.removeItem(w.timeStampLabel)
            w.timeStampLabel = None
            try:
                w.sigTimeChanged.disconnect(w.updateTimeStampLabel)
            except (TypeError, RuntimeError):
                pass

        if not show:
            w._ts_props = None
            return None

        # Store properties for later recall and for updateTimeStampLabel
        w.framerate = framerate
        g.settings['framerate'] = framerate
        w._ts_props = {
            'framerate': framerate, 'font_size': font_size,
            'color': color, 'bg_color': bg_color, 'location': location,
            'bold': bold, 'show_frame_number': show_frame_number,
            'format_string': format_string,
        }

        # Create initial label
        html = self._build_html(0, w._ts_props)
        label = pg.TextItem(html=html, anchor=(0, 0))
        w.timeStampLabel = label
        w.imageview.view.addItem(label)
        self._position_label(w, label, location)

        # Patch the window's update method to use our enhanced formatter
        def _update(frame, props=w._ts_props, lbl=label, win=w, stamp=self):
            lbl.setHtml(stamp._build_html(frame, props))
            stamp._position_label(win, lbl, props['location'])

        w.updateTimeStampLabel = _update
        w.sigTimeChanged.connect(w.updateTimeStampLabel)
        return None

    def _build_html(self, frame, props):
        """Build HTML string for the timestamp at the given frame."""
        framerate = props['framerate']
        font_size = props['font_size']
        color_css = _color_to_css(props['color'])
        bg_css = _color_to_css(props['bg_color'])
        bold_tag = 'font-weight:bold;' if props['bold'] else ''
        show_frame = props['show_frame_number']
        fmt = props.get('format_string', '').strip()

        if framerate == 0:
            time_str = "0 Hz"
        else:
            ttime = frame / framerate
            ms = ttime * 1000
            s = ttime
            minutes = int(np.floor(ttime / 60))
            hours = int(np.floor(ttime / 3600))

            if fmt:
                # User-provided format
                try:
                    time_str = fmt.format(
                        time=self._auto_format_time(ttime),
                        frame=int(frame), ms=ms, s=s,
                        min=minutes, hr=hours,
                    )
                except (KeyError, ValueError):
                    time_str = self._auto_format_time(ttime)
            else:
                time_str = self._auto_format_time(ttime)

        if show_frame:
            time_str += f"  [F{int(frame)}]"

        return (f"<span style='font-size:{font_size}pt; color:{color_css}; "
                f"background-color:{bg_css}; {bold_tag} padding:2px;'>"
                f"{time_str}</span>")

    @staticmethod
    def _auto_format_time(ttime):
        """Auto-format time value with appropriate units."""
        if ttime < 1:
            return f"{ttime * 1000:.0f} ms"
        elif ttime < 60:
            return f"{ttime:.3f} s"
        elif ttime < 3600:
            m = int(np.floor(ttime / 60))
            s = ttime % 60
            return f"{m}m {s:.1f}s"
        else:
            h = int(np.floor(ttime / 3600))
            rem = ttime - h * 3600
            m = int(np.floor(rem / 60))
            s = rem - m * 60
            return f"{h}h {m}m {s:.1f}s"

    @staticmethod
    def _position_label(win, label, location):
        """Position the label in the specified corner."""
        view = win.imageview.view
        vps = view.viewPixelSize()
        text_rect = label.boundingRect()
        tw = text_rect.width() * vps[0]
        th = text_rect.height() * vps[1]
        margin_x = 4
        margin_y = 4

        if location == 'Upper Left':
            label.setPos(margin_x, margin_y)
        elif location == 'Upper Right':
            label.setPos(win.mx - tw - margin_x, margin_y)
        elif location == 'Lower Left':
            label.setPos(margin_x, win.my - th - margin_y)
        elif location == 'Lower Right':
            label.setPos(win.mx - tw - margin_x, win.my - th - margin_y)

    def preview(self):
        vals = {}
        for item in self.items:
            obj = item['object']
            if hasattr(obj, 'value') and callable(obj.value):
                vals[item['name']] = obj.value()
            elif hasattr(obj, 'currentText') and callable(obj.currentText):
                vals[item['name']] = obj.currentText()
            elif hasattr(obj, 'isChecked') and callable(obj.isChecked):
                vals[item['name']] = obj.isChecked()
            elif hasattr(obj, 'text') and callable(obj.text):
                vals[item['name']] = obj.text()
        self.__call__(**vals)


time_stamp = Time_Stamp()


# ---------------------------------------------------------------------------
# Background overlay (unchanged)
# ---------------------------------------------------------------------------

class ShowCheckbox(CheckBox):

    def __init__(self, opacity_slider, parent=None):
        super().__init__(parent)
        self.stateChanged.connect(self.changed)
        self.opacity_slider = opacity_slider

    def changed(self, state):
        if state == 0:  # unchecked
            self.opacity_slider.setEnabled(True)
        if state == 2:  # checked
            self.opacity_slider.setEnabled(False)


class Background(BaseProcess):
    """ background(background_window, data_window)

    Overlays the background_window onto the data_window

    Parameters:
        background_window (Window)
        data_window (Window)
    Returns:
        None
    """
    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        background_window = WindowSelector()
        data_window = WindowSelector()
        opacity = SliderLabel(3)
        opacity.setRange(0, 1)
        opacity.setValue(.5)
        show = ShowCheckbox(opacity)
        show.setChecked(True)
        self.items.append({'name': 'background_window', 'string': 'Background window', 'object': background_window})
        self.items.append({'name': 'data_window', 'string': 'Data window', 'object': data_window})
        self.items.append({'name': 'opacity', 'string': 'Opacity', 'object': opacity})
        self.items.append({'name': 'show', 'string': 'Show', 'object': show})
        super().gui()

    def __call__(self, background_window, data_window, opacity, show, keepSourceWindow=False):
        if background_window is None or data_window is None:
            return
        w = data_window
        if show:
            if hasattr(w, 'bgItem') and w.bgItem is not None:
                w.bgItem.hist_luttt.hide()
                w.imageview.ui.gridLayout.removeWidget(w.bgItem.hist_luttt)
                w.imageview.view.removeItem(w.bgItem)
            bgItem = pg.ImageItem(background_window.imageview.imageItem.image)
            bgItem.setOpacity(opacity)
            w.imageview.view.addItem(bgItem)
            bgItem.hist_luttt = pg.HistogramLUTWidget()
            bgItem.hist_luttt.setMinimumWidth(110)
            bgItem.hist_luttt.setImageItem(bgItem)
            w.imageview.ui.gridLayout.addWidget(bgItem.hist_luttt, 0, 4, 1, 4)
            w.bgItem = bgItem
        else:
            if hasattr(w, 'bgItem') and w.bgItem is not None:
                w.bgItem.hist_luttt.hide()
                w.imageview.ui.gridLayout.removeWidget(w.bgItem.hist_luttt)
                w.imageview.view.removeItem(w.bgItem)
                w.bgItem.hist_luttt = None
                w.bgItem = None
            return None

    def preview(self):
        background_window = self.getValue('background_window')
        data_window = self.getValue('data_window')
        opacity = self.getValue('opacity')
        show = self.getValue('show')
        self.__call__(background_window, data_window, opacity, show)


background = Background()


# ---------------------------------------------------------------------------
# Scale Bar
# ---------------------------------------------------------------------------

class Scale_Bar(BaseProcess):
    """scale_bar(width_um, width_pixels, font_size, color, bg_color, location,
                 bar_color, bar_height, show_label, show, label_text, offset_x, offset_y)

    Adds a scale bar overlay to an image window.

    Parameters:
        width_um (float): Physical width the bar represents (in the unit).
        width_pixels (int): Width of the bar in pixels.
        font_size (int): Label font size in points.
        color (str): Label text color.
        bg_color (str): Label background color.
        bar_color (str): Bar fill color.
        location (str): Corner placement.
        bar_height (int): Bar thickness in pixels.
        show_label (bool): Show the text label above the bar.
        show (bool): Show or hide the scale bar.
        unit (str): Physical unit string (um, nm, mm, etc.).
        label_text (str): Override label text (empty = auto).
        bold (bool): Bold label text.
        offset_x (int): Extra horizontal offset from corner (pixels).
        offset_y (int): Extra vertical offset from corner (pixels).
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        w = g.win
        if w is None:
            g.alert("No window is currently selected.")
            return

        # --- Auto-calculate from pixel_size setting ---
        pixel_size_nm = g.settings.get('pixel_size', 108.0) or 108.0
        pixel_size_um = pixel_size_nm / 1000.0
        # Default: bar ~1/6 of image width, round to nice number
        default_bar_px = max(10, int(w.mx / 6))
        default_width_um = default_bar_px * pixel_size_um
        # Round to nearest nice number
        default_width_um = _round_to_nice(default_width_um)
        default_bar_px = max(1, int(round(default_width_um / pixel_size_um)))

        width_um = QtWidgets.QDoubleSpinBox()
        width_um.setRange(0.001, 100000.0)
        width_um.setDecimals(3)
        width_um.setValue(default_width_um)

        width_pixels = QtWidgets.QSpinBox()
        width_pixels.setRange(1, max(10, int(w.mx)))
        width_pixels.setValue(default_bar_px)

        unit = ComboBox()
        for u in ['\u00b5m', 'nm', 'mm', 'px']:
            unit.addItem(u)

        # Link width_um and width_pixels via pixel_size
        def _um_changed(val):
            if unit.currentText() == 'px':
                return
            px = max(1, int(round(val / pixel_size_um)))
            width_pixels.blockSignals(True)
            width_pixels.setValue(px)
            width_pixels.blockSignals(False)

        def _px_changed(val):
            if unit.currentText() == 'px':
                return
            um = val * pixel_size_um
            width_um.blockSignals(True)
            width_um.setValue(um)
            width_um.blockSignals(False)

        width_um.valueChanged.connect(_um_changed)
        width_pixels.valueChanged.connect(_px_changed)

        font_size = QtWidgets.QSpinBox()
        font_size.setRange(4, 120)
        font_size.setValue(12)

        color = _make_color_combo(_PRESET_COLORS, 'White')
        bg_color = _make_color_combo(_BG_COLORS, 'None')
        bar_color = _make_color_combo(_PRESET_COLORS, 'White')

        location = ComboBox()
        for loc in _LOCATIONS:
            location.addItem(loc)
        location.setCurrentIndex(0)  # Lower Right

        bar_height = QtWidgets.QSpinBox()
        bar_height.setRange(1, 100)
        bar_height.setValue(4)

        show_label = CheckBox()
        show_label.setChecked(True)

        bold = CheckBox()
        bold.setChecked(False)

        label_text = QtWidgets.QLineEdit()
        label_text.setPlaceholderText("auto (e.g. 10 \u00b5m)")
        label_text.setToolTip("Leave empty for auto label, or type custom text")

        offset_x = QtWidgets.QSpinBox()
        offset_x.setRange(0, 500)
        offset_x.setValue(10)

        offset_y = QtWidgets.QSpinBox()
        offset_y.setRange(0, 500)
        offset_y.setValue(10)

        show = CheckBox()
        show.setChecked(True)

        # Restore existing properties
        if hasattr(w, '_sb_props') and w._sb_props:
            p = w._sb_props
            width_um.setValue(p.get('width_um', default_width_um))
            width_pixels.setValue(p.get('width_pixels', default_bar_px))
            font_size.setValue(p.get('font_size', 12))
            for combo, key, default in [
                (color, 'color', 'White'), (bg_color, 'bg_color', 'None'),
                (bar_color, 'bar_color', 'White')
            ]:
                idx = combo.findText(p.get(key, default))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            idx = location.findText(p.get('location', 'Lower Right'))
            if idx >= 0:
                location.setCurrentIndex(idx)
            bar_height.setValue(p.get('bar_height', 4))
            show_label.setChecked(p.get('show_label', True))
            bold.setChecked(p.get('bold', False))
            label_text.setText(p.get('label_text', ''))
            offset_x.setValue(p.get('offset_x', 10))
            offset_y.setValue(p.get('offset_y', 10))
            idx = unit.findText(p.get('unit', '\u00b5m'))
            if idx >= 0:
                unit.setCurrentIndex(idx)

        self.items.append({'name': 'width_um', 'string': 'Physical Width', 'object': width_um})
        self.items.append({'name': 'unit', 'string': 'Unit', 'object': unit})
        self.items.append({'name': 'width_pixels', 'string': 'Width (pixels)', 'object': width_pixels})
        self.items.append({'name': 'bar_height', 'string': 'Bar Thickness (px)', 'object': bar_height})
        self.items.append({'name': 'font_size', 'string': 'Font Size (pt)', 'object': font_size})
        self.items.append({'name': 'bold', 'string': 'Bold Label', 'object': bold})
        self.items.append({'name': 'color', 'string': 'Label Color', 'object': color})
        self.items.append({'name': 'bar_color', 'string': 'Bar Color', 'object': bar_color})
        self.items.append({'name': 'bg_color', 'string': 'Background', 'object': bg_color})
        self.items.append({'name': 'location', 'string': 'Location', 'object': location})
        self.items.append({'name': 'show_label', 'string': 'Show Label', 'object': show_label})
        self.items.append({'name': 'label_text', 'string': 'Custom Label', 'object': label_text})
        self.items.append({'name': 'offset_x', 'string': 'Horizontal Offset', 'object': offset_x})
        self.items.append({'name': 'offset_y', 'string': 'Vertical Offset', 'object': offset_y})
        self.items.append({'name': 'show', 'string': 'Show', 'object': show})

        super().gui()
        self.preview()

    def __call__(self, width_um=10.0, width_pixels=50, font_size=12,
                 color='White', bg_color='None', location='Lower Right',
                 bar_color='White', bar_height=4, show_label=True,
                 show=True, unit='\u00b5m', label_text='', bold=False,
                 offset_x=10, offset_y=10, keepSourceWindow=None):
        w = g.win
        if w is None:
            return

        # Remove existing scale bar
        self._remove_scale_bar(w)

        if not show:
            w._sb_props = None
            return None

        # Store properties
        w._sb_props = {
            'width_um': width_um, 'width_pixels': width_pixels,
            'font_size': font_size, 'color': color, 'bg_color': bg_color,
            'bar_color': bar_color, 'location': location,
            'bar_height': bar_height, 'show_label': show_label,
            'unit': unit, 'label_text': label_text, 'bold': bold,
            'offset_x': offset_x, 'offset_y': offset_y,
        }

        view = w.imageview.view
        bar_rgba = _color_to_rgba(bar_color)

        # Auto-generate label text
        if label_text.strip():
            display_text = label_text
        else:
            # Format nicely
            if width_um == int(width_um):
                display_text = f"{int(width_um)} {unit}"
            else:
                display_text = f"{width_um:.3g} {unit}"

        # Create bar rectangle
        bar_item = QtWidgets.QGraphicsRectItem(
            QtCore.QRectF(0, 0, width_pixels, bar_height))
        bar_item.setPen(pg.mkPen(bar_rgba))
        bar_item.setBrush(pg.mkBrush(bar_rgba))
        view.addItem(bar_item)

        # Create label
        text_item = None
        if show_label:
            color_css = _color_to_css(color)
            bg_css = _color_to_css(bg_color)
            bold_css = 'font-weight:bold;' if bold else ''
            html = (f"<span style='font-size:{font_size}pt; color:{color_css}; "
                    f"background-color:{bg_css}; {bold_css}'>{display_text}</span>")
            text_item = pg.TextItem(html=html, anchor=(0.5, 1))
            view.addItem(text_item)

        # Store references on the window
        w.scaleBarLabel = text_item
        w._sb_bar = bar_item
        w._sb_text = text_item

        # Position
        self._position_scale_bar(w)

        # Connect resize to reposition
        try:
            view.sigResized.connect(lambda: self._position_scale_bar(w))
            w._sb_resize_connected = True
        except Exception:
            w._sb_resize_connected = False

        return None

    def _remove_scale_bar(self, w):
        """Remove existing scale bar items from the window."""
        view = w.imageview.view
        if hasattr(w, '_sb_bar') and w._sb_bar is not None:
            view.removeItem(w._sb_bar)
            w._sb_bar = None
        if hasattr(w, '_sb_text') and w._sb_text is not None:
            view.removeItem(w._sb_text)
            w._sb_text = None
        # Legacy cleanup
        if hasattr(w, 'scaleBarLabel') and w.scaleBarLabel is not None:
            try:
                if hasattr(w.scaleBarLabel, 'bar'):
                    view.removeItem(w.scaleBarLabel.bar)
                view.removeItem(w.scaleBarLabel)
            except Exception:
                pass
            w.scaleBarLabel = None
        # Disconnect resize
        if hasattr(w, '_sb_resize_connected') and w._sb_resize_connected:
            try:
                view.sigResized.disconnect()
            except (TypeError, RuntimeError):
                pass
            w._sb_resize_connected = False

    def _position_scale_bar(self, w):
        """Position the bar and label in the specified corner."""
        props = getattr(w, '_sb_props', None)
        if props is None:
            return
        bar = getattr(w, '_sb_bar', None)
        text = getattr(w, '_sb_text', None)
        if bar is None:
            return

        location = props['location']
        width_pixels = props['width_pixels']
        bar_height = props['bar_height']
        ox = props['offset_x']
        oy = props['offset_y']

        view = w.imageview.view
        vps = view.viewPixelSize()

        # Compute text dimensions in image coords
        text_h = 0
        if text is not None:
            tr = text.boundingRect()
            text_h = tr.height() * vps[1]

        if location == 'Lower Right':
            bx = w.mx - width_pixels - ox
            by = w.my - bar_height - oy
        elif location == 'Lower Left':
            bx = ox
            by = w.my - bar_height - oy
        elif location == 'Upper Right':
            bx = w.mx - width_pixels - ox
            by = oy + text_h
        elif location == 'Upper Left':
            bx = ox
            by = oy + text_h
        else:
            bx, by = ox, w.my - bar_height - oy

        bar.setRect(QtCore.QRectF(bx, by, width_pixels, bar_height))

        if text is not None:
            # Center text above bar
            text.setPos(bx + width_pixels / 2, by)

    def preview(self):
        vals = {}
        for item in self.items:
            obj = item['object']
            if hasattr(obj, 'value') and callable(obj.value):
                vals[item['name']] = obj.value()
            elif hasattr(obj, 'currentText') and callable(obj.currentText):
                vals[item['name']] = obj.currentText()
            elif hasattr(obj, 'isChecked') and callable(obj.isChecked):
                vals[item['name']] = obj.isChecked()
            elif hasattr(obj, 'text') and callable(obj.text):
                vals[item['name']] = obj.text()
        self.__call__(**vals)

    # Keep legacy updateBar for compatibility
    def updateBar(self):
        w = g.win
        if w is not None:
            self._position_scale_bar(w)


scale_bar = Scale_Bar()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_to_nice(value):
    """Round a value to a 'nice' number for scale bar labels (1, 2, 5, 10, ...)."""
    if value <= 0:
        return 1.0
    import math
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction < 1.5:
        nice = 1
    elif fraction < 3.5:
        nice = 2
    elif fraction < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exponent)


# ---------------------------------------------------------------------------
# Grid Overlay
# ---------------------------------------------------------------------------

class Grid_Overlay(BaseProcess):
    """grid_overlay(grid_type, spacing_x, spacing_y, divisions_x, divisions_y,
                    color, width, opacity, style, use_divisions, show)

    Adds a configurable grid overlay to the current window.

    Parameters:
        grid_type (str): 'Rectangular', 'Crosshair', 'Rule of Thirds',
                         'Dot Grid', or 'Polar'.
        spacing_x (int): Pixel spacing between vertical lines.
        spacing_y (int): Pixel spacing between horizontal lines.
        divisions_x (int): Number of horizontal divisions.
        divisions_y (int): Number of vertical divisions.
        color (str): Grid color.
        width (float): Line width.
        opacity (float): Opacity (0-1).
        style (str): 'solid', 'dash', 'dot', 'dash-dot'.
        use_divisions (bool): Use divisions instead of spacing.
        show (bool): Show or hide the grid.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        if g.win is None:
            g.alert("No window is currently selected.")
            return

        grid_type = ComboBox()
        grid_type.addItems(['Rectangular', 'Crosshair', 'Rule of Thirds',
                            'Dot Grid', 'Polar'])
        spacing_x = QtWidgets.QSpinBox()
        spacing_x.setRange(5, 5000)
        spacing_x.setValue(50)
        spacing_y = QtWidgets.QSpinBox()
        spacing_y.setRange(5, 5000)
        spacing_y.setValue(50)
        divisions_x = QtWidgets.QSpinBox()
        divisions_x.setRange(1, 100)
        divisions_x.setValue(4)
        divisions_y = QtWidgets.QSpinBox()
        divisions_y.setRange(1, 100)
        divisions_y.setValue(4)
        color = _make_color_combo(_PRESET_COLORS, 'Yellow')
        width_spin = QtWidgets.QDoubleSpinBox()
        width_spin.setRange(0.5, 10)
        width_spin.setValue(1.0)
        width_spin.setSingleStep(0.5)
        opacity = SliderLabel(2)
        opacity.setRange(0, 1)
        opacity.setValue(0.4)
        style = ComboBox()
        style.addItems(['solid', 'dash', 'dot', 'dash-dot'])
        use_divisions = CheckBox()
        show = CheckBox()
        show.setChecked(True)

        self.items.append({'name': 'grid_type', 'string': 'Grid Type', 'object': grid_type})
        self.items.append({'name': 'spacing_x', 'string': 'Spacing X', 'object': spacing_x})
        self.items.append({'name': 'spacing_y', 'string': 'Spacing Y', 'object': spacing_y})
        self.items.append({'name': 'divisions_x', 'string': 'Divisions X', 'object': divisions_x})
        self.items.append({'name': 'divisions_y', 'string': 'Divisions Y', 'object': divisions_y})
        self.items.append({'name': 'color', 'string': 'Color', 'object': color})
        self.items.append({'name': 'width', 'string': 'Line Width', 'object': width_spin})
        self.items.append({'name': 'opacity', 'string': 'Opacity', 'object': opacity})
        self.items.append({'name': 'style', 'string': 'Line Style', 'object': style})
        self.items.append({'name': 'use_divisions', 'string': 'Use Divisions', 'object': use_divisions})
        self.items.append({'name': 'show', 'string': 'Show', 'object': show})
        super().gui()

    def __call__(self, grid_type='Rectangular', spacing_x=50, spacing_y=50,
                 divisions_x=4, divisions_y=4, color='Yellow', width=1.0,
                 opacity=0.4, style='solid', use_divisions=False, show=True,
                 keepSourceWindow=None):
        w = g.win
        if w is None:
            return

        from ..viewers.grid_overlay import GridOverlay
        if not hasattr(w, '_grid_overlay') or w._grid_overlay is None:
            w._grid_overlay = GridOverlay(w)

        if not show:
            w._grid_overlay.clear()
            return None

        if grid_type == 'Rectangular':
            w._grid_overlay.draw_rectangular(
                spacing_x=spacing_x, spacing_y=spacing_y,
                color=color, width=width, opacity=opacity, style=style,
                divisions_mode=use_divisions,
                divisions_x=divisions_x, divisions_y=divisions_y)
        elif grid_type == 'Crosshair':
            w._grid_overlay.draw_crosshair(
                color=color, width=width, opacity=opacity, style=style)
        elif grid_type == 'Rule of Thirds':
            w._grid_overlay.draw_thirds(
                color=color, width=width, opacity=opacity, style=style)
        elif grid_type == 'Dot Grid':
            w._grid_overlay.draw_dot_grid(
                spacing_x=spacing_x, spacing_y=spacing_y,
                color=color, dot_size=3, opacity=opacity)
        elif grid_type == 'Polar':
            w._grid_overlay.draw_polar(
                color=color, width=width, opacity=opacity, style=style)
        return None

    def preview(self):
        vals = {}
        for item in self.items:
            obj = item['object']
            if hasattr(obj, 'value') and callable(obj.value):
                vals[item['name']] = obj.value()
            elif hasattr(obj, 'currentText') and callable(obj.currentText):
                vals[item['name']] = obj.currentText()
            elif hasattr(obj, 'isChecked') and callable(obj.isChecked):
                vals[item['name']] = obj.isChecked()
        self.__call__(**vals)


grid_overlay = Grid_Overlay()


# ---------------------------------------------------------------------------
# Counting Tool (launcher)
# ---------------------------------------------------------------------------

class Counting_Tool(BaseProcess):
    """counting_tool()

    Opens the counting tool panel for manual object counting.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        from ..viewers.counting_overlay import CountingPanel
        panel = CountingPanel.instance(g.m)
        if g.m is not None:
            g.m.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
        panel.show()
        panel.raise_()

    def __call__(self, keepSourceWindow=None):
        self.gui()


counting_tool = Counting_Tool()


# ---------------------------------------------------------------------------
# Bake Overlays
# ---------------------------------------------------------------------------

class Bake_Overlays(BaseProcess):
    """bake_overlays_process(include_rois, include_grid, include_scale_bar,
                             include_timestamp, create_new_window)

    Burns visible overlays into image pixel data.

    Parameters:
        include_rois (bool): Include ROIs.
        include_grid (bool): Include grid overlay.
        include_scale_bar (bool): Include scale bar.
        include_timestamp (bool): Include timestamp.
        create_new_window (bool): Create new window with baked result.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        self.gui_reset()
        if g.win is None:
            g.alert("No window is currently selected.")
            return

        include_rois = CheckBox()
        include_rois.setChecked(True)
        include_grid = CheckBox()
        include_grid.setChecked(True)
        include_scale_bar = CheckBox()
        include_scale_bar.setChecked(True)
        include_timestamp = CheckBox()
        include_timestamp.setChecked(True)
        create_new_window = CheckBox()
        create_new_window.setChecked(True)

        self.items.append({'name': 'include_rois', 'string': 'Include ROIs', 'object': include_rois})
        self.items.append({'name': 'include_grid', 'string': 'Include Grid', 'object': include_grid})
        self.items.append({'name': 'include_scale_bar', 'string': 'Include Scale Bar', 'object': include_scale_bar})
        self.items.append({'name': 'include_timestamp', 'string': 'Include Timestamp', 'object': include_timestamp})
        self.items.append({'name': 'create_new_window', 'string': 'New Window', 'object': create_new_window})
        super().gui()

    def __call__(self, include_rois=True, include_grid=True,
                 include_scale_bar=True, include_timestamp=True,
                 create_new_window=True, keepSourceWindow=None):
        w = g.win
        if w is None:
            return
        from ..viewers.overlay_manager import bake_overlays
        return bake_overlays(
            w,
            include_rois=include_rois,
            include_grid=include_grid,
            include_scale_bar=include_scale_bar,
            include_timestamp=include_timestamp,
            create_new_window=create_new_window,
        )


bake_overlays_process = Bake_Overlays()


logger.debug("Completed 'reading process/overlay.py'")
