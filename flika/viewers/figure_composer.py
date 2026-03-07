# -*- coding: utf-8 -*-
"""
Figure Composer
===============

Publication-quality figure composition dialog.  Arranges one or more flika
window images into a grid layout with titles, scale bars, and colorbars, and
exports the result as PNG, SVG, or PDF.

Usage::

    from flika.viewers.figure_composer import FigureComposerDialog
    dlg = FigureComposerDialog()
    dlg.show()

"""
from ..logger import logger
logger.debug("Started 'reading viewers/figure_composer.py'")

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Signal

# ---------------------------------------------------------------------------
# FigurePanel — data model for one cell in the grid
# ---------------------------------------------------------------------------

class FigurePanel:
    """Data model for a single panel in the composed figure.

    Each panel maps to one cell (or span of cells) in the grid layout and
    stores the information needed to render its image, title, scale bar, and
    colorbar.
    """

    def __init__(self, row=0, col=0):
        self.source_window = None   # flika Window object or None ("Empty")
        self.frame_index = 0        # which frame for movies
        self.title = ''             # text label above the panel
        self.show_scale_bar = False
        self.scale_bar_um = 10.0    # physical length in microns
        self.scale_bar_px = 50.0    # length in pixels
        self.show_colorbar = False
        self.row = row
        self.col = col
        self.row_span = 1
        self.col_span = 1

    def __repr__(self):
        src = getattr(self.source_window, 'name', 'Empty')
        return (
            f"<FigurePanel ({self.row},{self.col}) src={src!r} "
            f"frame={self.frame_index}>"
        )


# ---------------------------------------------------------------------------
# Helper: extract a QImage from a Window
# ---------------------------------------------------------------------------

def _window_to_qimage(window, frame_index=0):
    """Extract a 2-D frame from *window* and return it as a ``QImage``.

    Parameters
    ----------
    window : Window
        A flika Window object.
    frame_index : int
        Frame index to use for time-series data.

    Returns
    -------
    QtGui.QImage or None
        The rendered image, or *None* if no data is available.
    """
    if window is None or window.image is None:
        return None

    # --- compositor path ---
    compositor = getattr(window, '_compositor', None)
    if compositor is not None:
        try:
            rgb = compositor.export_composite_rgb()  # (H, W, 3) uint8
            return _rgb_array_to_qimage(rgb)
        except Exception as exc:
            logger.warning("Compositor export failed: %s", exc)
            # fall through to normal path

    img = np.asarray(window.image)
    ndim = img.ndim

    # --- determine the 2-D or RGB frame ---
    is_rgb = False
    if ndim == 2:
        frame = img
    elif ndim == 3:
        if img.shape[2] in (3, 4):
            # Single RGB(A) image
            frame = img[:, :, :3]
            is_rgb = True
        else:
            # (t, y, x)
            idx = min(frame_index, img.shape[0] - 1)
            frame = img[idx]
    elif ndim == 4:
        # (t, y, x, 3/4)
        idx = min(frame_index, img.shape[0] - 1)
        frame = img[idx]
        if frame.ndim == 3 and frame.shape[2] in (3, 4):
            frame = frame[:, :, :3]
            is_rgb = True
    else:
        frame = img

    if is_rgb:
        rgb = np.clip(frame, 0, 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
        return _rgb_array_to_qimage(np.ascontiguousarray(rgb))

    # --- grayscale: apply the window's current LUT ---
    frame = np.asarray(frame, dtype=np.float64)
    rgb = _apply_window_lut(window, frame)
    if rgb is not None:
        return _rgb_array_to_qimage(rgb)

    # Fallback: simple normalisation to grayscale
    gray = _normalize_to_uint8(frame)
    return _gray_array_to_qimage(gray)


def _apply_window_lut(window, frame):
    """Apply the window's current LUT/colormap to a grayscale *frame*.

    Returns an (H, W, 3) uint8 array, or *None* if no LUT is available.
    """
    try:
        iv = window.imageview
        if iv is None:
            return None

        # Get current display levels from pyqtgraph ImageView
        levels = iv.getLevels()
        if levels is None:
            return None
        lo, hi = float(levels[0]), float(levels[1])
        span = hi - lo
        if span <= 0:
            span = 1.0

        # Normalize to [0, 255]
        normed = (frame - lo) / span * 255.0
        np.clip(normed, 0.0, 255.0, out=normed)
        indices = normed.astype(np.intp)
        np.clip(indices, 0, 255, out=indices)

        # Try to get the LUT from the imageItem
        lut = iv.imageItem.lut
        if lut is not None and hasattr(lut, '__len__') and len(lut) >= 256:
            lut = np.asarray(lut)
            if lut.ndim == 2 and lut.shape[1] >= 3:
                mapped = lut[indices][:, :, :3].astype(np.uint8)
                return np.ascontiguousarray(mapped)

        # If the LUT is a callable (e.g. from a gradient), try to call it
        if callable(lut):
            try:
                lut_arr = lut(np.arange(256))
                if lut_arr is not None:
                    lut_arr = np.asarray(lut_arr)
                    if lut_arr.ndim == 2 and lut_arr.shape[1] >= 3:
                        mapped = lut_arr[indices][:, :, :3].astype(np.uint8)
                        return np.ascontiguousarray(mapped)
            except Exception:
                pass

        # Try to get LUT from the histogram widget gradient
        try:
            hist = iv.getHistogramWidget()
            if hist is not None:
                gradient = hist.gradient
                lut_arr = gradient.getLookupTable(256)
                if lut_arr is not None:
                    lut_arr = np.asarray(lut_arr)
                    if lut_arr.ndim == 2 and lut_arr.shape[1] >= 3:
                        mapped = lut_arr[indices][:, :, :3].astype(np.uint8)
                        return np.ascontiguousarray(mapped)
        except Exception:
            pass

    except Exception as exc:
        logger.debug("Could not apply window LUT: %s", exc)

    return None


def _normalize_to_uint8(frame):
    """Normalize a 2-D float array to uint8 [0, 255]."""
    frame = np.asarray(frame, dtype=np.float64)
    lo = float(np.nanmin(frame))
    hi = float(np.nanmax(frame))
    span = hi - lo
    if span <= 0:
        return np.zeros(frame.shape, dtype=np.uint8)
    normed = (frame - lo) / span * 255.0
    np.clip(normed, 0.0, 255.0, out=normed)
    return normed.astype(np.uint8)


def _rgb_array_to_qimage(rgb):
    """Convert an (H, W, 3) uint8 array to a ``QImage`` (Format_RGB888).

    Returns a *copy* of the QImage so the numpy buffer can be freed safely.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    bytes_per_line = 3 * w
    qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line,
                        QtGui.QImage.Format.Format_RGB888)
    return qimg.copy()  # detach from numpy buffer


def _gray_array_to_qimage(gray):
    """Convert an (H, W) uint8 array to a ``QImage`` (Format_Grayscale8).

    Returns a *copy* of the QImage so the numpy buffer can be freed safely.
    """
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape[:2]
    qimg = QtGui.QImage(gray.data, w, h, w,
                        QtGui.QImage.Format.Format_Grayscale8)
    return qimg.copy()


# ---------------------------------------------------------------------------
# FigureScene — QGraphicsScene that renders the composed figure
# ---------------------------------------------------------------------------

class FigureScene(QtWidgets.QGraphicsScene):
    """A ``QGraphicsScene`` that renders a grid of image panels for figure
    composition.

    Call :meth:`set_layout` to configure grid dimensions and styling, then
    :meth:`render_panels` to (re)draw all panels.  Use the ``export_*``
    methods to save the result to disk.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = 1
        self._cols = 1
        self._panels = []
        self._dpi = 300
        self._bg_color = QtCore.Qt.GlobalColor.white
        self._padding = 10
        self._font_name = 'Arial'
        self._font_size = 12
        # Default panel pixel size (before DPI scaling).
        self._panel_width = 256
        self._panel_height = 256

    # ----- configuration -----

    def set_layout(self, rows, cols, panels, dpi=300, bg_color=None,
                   padding=10, font_name='Arial', font_size=12):
        """Configure the scene layout.

        Parameters
        ----------
        rows, cols : int
            Grid dimensions.
        panels : list[FigurePanel]
            The panels to render (one per occupied cell).
        dpi : int
            Target DPI for raster export.
        bg_color : QColor or Qt.GlobalColor or None
            Background fill colour.  *None* means white.
        padding : int
            Spacing between panels in pixels.
        font_name : str
            Font family for titles and labels.
        font_size : int
            Font point size.
        """
        self._rows = max(1, rows)
        self._cols = max(1, cols)
        self._panels = list(panels)
        self._dpi = max(72, dpi)
        self._bg_color = bg_color if bg_color is not None else QtCore.Qt.GlobalColor.white
        self._padding = max(0, padding)
        self._font_name = font_name
        self._font_size = max(6, font_size)

    # ----- rendering -----

    def render_panels(self):
        """Clear the scene and draw all panels according to the current layout.
        """
        self.clear()

        font = QtGui.QFont(self._font_name, self._font_size)
        font_metrics = QtGui.QFontMetrics(font)
        title_height = font_metrics.height() + 4  # a little extra space

        pad = self._padding
        pw = self._panel_width
        ph = self._panel_height

        # Total cell size including title space above and padding
        cell_w = pw + pad
        cell_h = ph + title_height + pad

        # Total scene size
        total_w = self._cols * cell_w + pad
        total_h = self._rows * cell_h + pad

        # Background
        bg_brush = QtGui.QBrush(QtGui.QColor(self._bg_color))
        self.setBackgroundBrush(bg_brush)
        self.setSceneRect(0, 0, total_w, total_h)

        for panel in self._panels:
            if panel.source_window is None:
                continue

            r, c = panel.row, panel.col
            if r >= self._rows or c >= self._cols:
                continue

            x0 = pad + c * cell_w
            y0 = pad + r * cell_h

            # ---- title ----
            if panel.title:
                text_item = self.addText(panel.title, font)
                text_item.setDefaultTextColor(self._title_color())
                # Centre the title above the panel
                tw = font_metrics.horizontalAdvance(panel.title)
                tx = x0 + (pw - tw) / 2.0
                text_item.setPos(tx, y0)

            img_y = y0 + title_height

            # ---- image ----
            qimg = _window_to_qimage(panel.source_window, panel.frame_index)
            if qimg is None or qimg.isNull():
                # Draw a placeholder rectangle
                self.addRect(x0, img_y, pw, ph,
                             QtGui.QPen(QtGui.QColor(128, 128, 128)),
                             QtGui.QBrush(QtGui.QColor(40, 40, 40)))
                continue

            # Scale image to fit the panel cell while preserving aspect ratio
            scaled = qimg.scaled(pw, ph,
                                 QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                 QtCore.Qt.TransformationMode.SmoothTransformation)
            pixmap = QtGui.QPixmap.fromImage(scaled)
            pix_item = self.addPixmap(pixmap)

            # Centre the (possibly smaller) pixmap in the cell
            ox = x0 + (pw - scaled.width()) / 2.0
            oy = img_y + (ph - scaled.height()) / 2.0
            pix_item.setPos(ox, oy)

            # ---- scale bar ----
            if panel.show_scale_bar and panel.scale_bar_px > 0:
                self._draw_scale_bar(panel, ox, oy, scaled.width(),
                                     scaled.height(), qimg.width(), font)

            # ---- colorbar ----
            if panel.show_colorbar:
                self._draw_colorbar(panel, ox + scaled.width() + 4, oy,
                                    scaled.height(), font)

    def _title_color(self):
        """Return an appropriate title text colour contrasting the background.
        """
        bg = QtGui.QColor(self._bg_color)
        lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        if lum < 128:
            return QtGui.QColor(QtCore.Qt.GlobalColor.white)
        return QtGui.QColor(QtCore.Qt.GlobalColor.black)

    def _bar_color(self):
        """Return a colour for scale bars that contrasts with the background.
        """
        bg = QtGui.QColor(self._bg_color)
        lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        if lum < 128:
            return QtGui.QColor(255, 255, 255)
        return QtGui.QColor(0, 0, 0)

    def _draw_scale_bar(self, panel, img_x, img_y, display_w, display_h,
                        original_w, font):
        """Draw a scale bar at the bottom-right of the panel image."""
        # Scale factor from original image pixels to display pixels
        if original_w > 0:
            scale = display_w / original_w
        else:
            scale = 1.0

        bar_display_px = panel.scale_bar_px * scale
        bar_h = max(3, int(display_h * 0.02))

        bar_x = img_x + display_w - bar_display_px - 8
        bar_y = img_y + display_h - bar_h - 12

        color = self._bar_color()
        pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        brush = QtGui.QBrush(color)
        self.addRect(bar_x, bar_y, bar_display_px, bar_h, pen, brush)

        # Label
        label = f"{panel.scale_bar_um:.4g} um"
        small_font = QtGui.QFont(self._font_name, max(6, self._font_size - 2))
        text_item = self.addText(label, small_font)
        text_item.setDefaultTextColor(color)
        fm = QtGui.QFontMetrics(small_font)
        tw = fm.horizontalAdvance(label)
        text_item.setPos(bar_x + (bar_display_px - tw) / 2.0,
                         bar_y - fm.height() - 2)

    def _draw_colorbar(self, panel, x, y, height, font):
        """Draw a vertical colorbar gradient to the right of the panel image.
        """
        bar_w = 16
        bar_h = height

        # Build a gradient from the window's current LUT
        gradient = QtGui.QLinearGradient(x, y + bar_h, x, y)

        # Try to extract colormap from the window
        lut = self._get_lut_for_window(panel.source_window)
        if lut is not None and len(lut) >= 2:
            n = len(lut)
            for i in range(0, n, max(1, n // 16)):
                frac = i / (n - 1)
                r, g, b = int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])
                gradient.setColorAt(frac, QtGui.QColor(r, g, b))
        else:
            # Fallback: simple grayscale gradient
            gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0))
            gradient.setColorAt(1.0, QtGui.QColor(255, 255, 255))

        pen = QtGui.QPen(self._bar_color(), 1)
        brush = QtGui.QBrush(gradient)
        self.addRect(x, y, bar_w, bar_h, pen, brush)

        # Min / max labels
        color = self._title_color()
        small_font = QtGui.QFont(self._font_name, max(6, self._font_size - 3))

        try:
            iv = panel.source_window.imageview
            levels = iv.getLevels() if iv is not None else None
        except Exception:
            levels = None

        if levels is not None:
            lo_text = f"{levels[0]:.3g}"
            hi_text = f"{levels[1]:.3g}"
        else:
            lo_text = "0"
            hi_text = "255"

        lo_item = self.addText(lo_text, small_font)
        lo_item.setDefaultTextColor(color)
        lo_item.setPos(x + bar_w + 2, y + bar_h - 12)

        hi_item = self.addText(hi_text, small_font)
        hi_item.setDefaultTextColor(color)
        hi_item.setPos(x + bar_w + 2, y - 2)

    @staticmethod
    def _get_lut_for_window(window):
        """Try to extract a (N, 3+) uint8 LUT array from the window.

        Returns *None* if no LUT can be determined.
        """
        try:
            iv = window.imageview
            if iv is None:
                return None

            # Direct LUT on imageItem
            lut = iv.imageItem.lut
            if lut is not None and hasattr(lut, '__len__') and len(lut) >= 256:
                arr = np.asarray(lut)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr

            # From gradient
            try:
                hist = iv.getHistogramWidget()
                if hist is not None:
                    lut_arr = hist.gradient.getLookupTable(256)
                    if lut_arr is not None:
                        lut_arr = np.asarray(lut_arr)
                        if lut_arr.ndim == 2 and lut_arr.shape[1] >= 3:
                            return lut_arr
            except Exception:
                pass
        except Exception:
            pass
        return None

    # ----- export methods -----

    def export_png(self, path, dpi=None):
        """Render the scene to a PNG file at the specified *dpi*.

        Parameters
        ----------
        path : str
            Output file path.
        dpi : int or None
            Override DPI (uses the configured value if *None*).
        """
        if dpi is None:
            dpi = self._dpi

        rect = self.sceneRect()
        if rect.isEmpty():
            logger.warning("FigureScene: nothing to export (empty scene)")
            return

        # Scale factor: scene coordinates are at ~96 dpi (screen).
        scale = dpi / 96.0
        w = int(rect.width() * scale)
        h = int(rect.height() * scale)

        image = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(self._bg_color))

        # Set DPI metadata
        dots_per_m = int(dpi / 0.0254)
        image.setDotsPerMeterX(dots_per_m)
        image.setDotsPerMeterY(dots_per_m)

        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform,
                              True)
        self.render(painter, QtCore.QRectF(0, 0, w, h), rect)
        painter.end()

        image.save(path, 'PNG')
        logger.info("Exported figure to PNG: %s (%dx%d @ %d dpi)",
                     path, w, h, dpi)

    def export_svg(self, path):
        """Render the scene to an SVG file.

        Parameters
        ----------
        path : str
            Output file path (should end in ``.svg``).
        """
        from qtpy.QtSvg import QSvgGenerator

        rect = self.sceneRect()
        if rect.isEmpty():
            logger.warning("FigureScene: nothing to export (empty scene)")
            return

        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(QtCore.QSize(int(rect.width()), int(rect.height())))
        generator.setViewBox(rect)
        generator.setTitle("Flika Figure")
        generator.setDescription("Generated by flika Figure Composer")

        painter = QtGui.QPainter(generator)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.render(painter, QtCore.QRectF(), rect)
        painter.end()

        logger.info("Exported figure to SVG: %s", path)

    def export_pdf(self, path):
        """Render the scene to a PDF file.

        Parameters
        ----------
        path : str
            Output file path (should end in ``.pdf``).
        """
        from qtpy.QtPrintSupport import QPrinter

        rect = self.sceneRect()
        if rect.isEmpty():
            logger.warning("FigureScene: nothing to export (empty scene)")
            return

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setResolution(self._dpi)

        # Set page size to match the scene
        page_size = QtCore.QSizeF(rect.width(), rect.height())
        from qtpy.QtGui import QPageSize
        printer.setPageSize(QPageSize(page_size, QPageSize.Unit.Point))

        painter = QtGui.QPainter(printer)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.render(painter, QtCore.QRectF(), rect)
        painter.end()

        logger.info("Exported figure to PDF: %s", path)


# ---------------------------------------------------------------------------
# Annotation items
# ---------------------------------------------------------------------------

class _TextAnnotation(QtWidgets.QGraphicsTextItem):
    """A movable, editable text annotation on the figure scene."""

    def __init__(self, text='Text', font=None, parent=None):
        super().__init__(text, parent)
        if font is not None:
            self.setFont(font)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                     True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                     True)
        self.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        self.setDefaultTextColor(QtGui.QColor(QtCore.Qt.GlobalColor.black))


class _ArrowAnnotation(QtWidgets.QGraphicsLineItem):
    """A simple movable arrow annotation (line with arrowhead)."""

    def __init__(self, x1=0, y1=0, x2=60, y2=60, parent=None):
        super().__init__(x1, y1, x2, y2, parent)
        pen = QtGui.QPen(QtGui.QColor(QtCore.Qt.GlobalColor.red), 2)
        self.setPen(pen)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                     True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
                     True)
        self._arrowhead = _ArrowHead(self)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        # Draw arrowhead at the end of the line
        line = self.line()
        if line.length() < 1:
            return
        angle = np.arctan2(-line.dy(), line.dx())
        arrow_size = 10
        p1 = line.p2()
        ax1 = p1.x() - arrow_size * np.cos(angle - np.pi / 6)
        ay1 = p1.y() + arrow_size * np.sin(angle - np.pi / 6)
        ax2 = p1.x() - arrow_size * np.cos(angle + np.pi / 6)
        ay2 = p1.y() + arrow_size * np.sin(angle + np.pi / 6)

        painter.setPen(self.pen())
        painter.setBrush(QtGui.QBrush(self.pen().color()))
        head = QtGui.QPolygonF([
            QtCore.QPointF(p1.x(), p1.y()),
            QtCore.QPointF(ax1, ay1),
            QtCore.QPointF(ax2, ay2),
        ])
        painter.drawPolygon(head)


class _ArrowHead:
    """Placeholder so _ArrowAnnotation can reference arrow state."""
    def __init__(self, parent_line):
        self.parent_line = parent_line


# ---------------------------------------------------------------------------
# PanelControlWidget — per-cell controls in the right sidebar
# ---------------------------------------------------------------------------

class PanelControlWidget(QtWidgets.QGroupBox):
    """Controls for a single grid cell: window selection, frame, title, etc.
    """

    changed = Signal()

    def __init__(self, row, col, parent=None):
        super().__init__(f"Panel ({row+1}, {col+1})", parent)
        self.row = row
        self.col = col
        self._panel = FigurePanel(row, col)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(4)

        # Window selector (combo box)
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.setMinimumWidth(120)
        layout.addRow("Window:", self.window_combo)

        # Frame index
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 99999)
        self.frame_spin.setValue(0)
        layout.addRow("Frame:", self.frame_spin)

        # Title
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("Panel title...")
        layout.addRow("Title:", self.title_edit)

        # Scale bar
        sb_layout = QtWidgets.QHBoxLayout()
        self.scale_bar_check = QtWidgets.QCheckBox("Scale bar")
        sb_layout.addWidget(self.scale_bar_check)
        self.scale_um_spin = QtWidgets.QDoubleSpinBox()
        self.scale_um_spin.setRange(0.01, 100000.0)
        self.scale_um_spin.setValue(10.0)
        self.scale_um_spin.setSuffix(" um")
        self.scale_um_spin.setDecimals(2)
        self.scale_um_spin.setEnabled(False)
        sb_layout.addWidget(self.scale_um_spin)
        self.scale_px_spin = QtWidgets.QDoubleSpinBox()
        self.scale_px_spin.setRange(1.0, 10000.0)
        self.scale_px_spin.setValue(50.0)
        self.scale_px_spin.setSuffix(" px")
        self.scale_px_spin.setDecimals(1)
        self.scale_px_spin.setEnabled(False)
        sb_layout.addWidget(self.scale_px_spin)
        layout.addRow(sb_layout)

        # Colorbar
        self.colorbar_check = QtWidgets.QCheckBox("Show colorbar")
        layout.addRow(self.colorbar_check)

        self.setLayout(layout)

    def _connect_signals(self):
        self.window_combo.currentIndexChanged.connect(self._on_changed)
        self.frame_spin.valueChanged.connect(self._on_changed)
        self.title_edit.textChanged.connect(self._on_changed)
        self.scale_bar_check.toggled.connect(self._on_scale_bar_toggled)
        self.scale_um_spin.valueChanged.connect(self._on_changed)
        self.scale_px_spin.valueChanged.connect(self._on_changed)
        self.colorbar_check.toggled.connect(self._on_changed)

    def _on_scale_bar_toggled(self, checked):
        self.scale_um_spin.setEnabled(checked)
        self.scale_px_spin.setEnabled(checked)
        self._on_changed()

    def _on_changed(self):
        self.changed.emit()

    def populate_windows(self):
        """Refresh the window combo box from ``g.windows``."""
        from .. import global_vars as g

        current_text = self.window_combo.currentText()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        self.window_combo.addItem("Empty", None)

        windows = list(g.windows) if hasattr(g, 'windows') else []
        for win in windows:
            name = getattr(win, 'name', str(win))
            self.window_combo.addItem(name, win)

        # Try to restore previous selection
        idx = self.window_combo.findText(current_text)
        if idx >= 0:
            self.window_combo.setCurrentIndex(idx)

        self.window_combo.blockSignals(False)

    def get_panel(self):
        """Return a :class:`FigurePanel` populated from the current controls.
        """
        panel = FigurePanel(self.row, self.col)
        panel.source_window = self.window_combo.currentData()
        panel.frame_index = self.frame_spin.value()
        panel.title = self.title_edit.text()
        panel.show_scale_bar = self.scale_bar_check.isChecked()
        panel.scale_bar_um = self.scale_um_spin.value()
        panel.scale_bar_px = self.scale_px_spin.value()
        panel.show_colorbar = self.colorbar_check.isChecked()
        return panel


# ---------------------------------------------------------------------------
# FigureComposerDialog — main dialog
# ---------------------------------------------------------------------------

class FigureComposerDialog(QtWidgets.QDialog):
    """Publication-quality figure composition dialog.

    Arranges flika window images into a configurable grid with titles, scale
    bars, colorbars, and annotation tools.  Exports to PNG, SVG, or PDF.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Composer")
        self.resize(1200, 800)
        self.setMinimumSize(800, 500)

        self._scene = FigureScene(self)
        self._panel_controls = []  # list[PanelControlWidget]

        self._build_ui()
        self._connect_signals()
        self._rebuild_panel_controls()

    # ----- UI construction -----

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # ---- Left: preview ----
        self._view = QtWidgets.QGraphicsView(self._scene)
        self._view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self._view.setRenderHint(
            QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        self._view.setDragMode(
            QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setMinimumWidth(500)
        main_layout.addWidget(self._view, stretch=3)

        # ---- Right: controls ----
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(450)

        scroll_content = QtWidgets.QWidget()
        self._controls_layout = QtWidgets.QVBoxLayout(scroll_content)
        self._controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # --- Grid size ---
        grid_group = QtWidgets.QGroupBox("Grid Size")
        grid_layout = QtWidgets.QHBoxLayout(grid_group)
        grid_layout.addWidget(QtWidgets.QLabel("Rows:"))
        self._rows_spin = QtWidgets.QSpinBox()
        self._rows_spin.setRange(1, 10)
        self._rows_spin.setValue(1)
        grid_layout.addWidget(self._rows_spin)
        grid_layout.addWidget(QtWidgets.QLabel("Cols:"))
        self._cols_spin = QtWidgets.QSpinBox()
        self._cols_spin.setRange(1, 10)
        self._cols_spin.setValue(2)
        grid_layout.addWidget(self._cols_spin)
        self._controls_layout.addWidget(grid_group)

        # --- Panel controls container ---
        self._panels_container = QtWidgets.QWidget()
        self._panels_layout = QtWidgets.QVBoxLayout(self._panels_container)
        self._panels_layout.setContentsMargins(0, 0, 0, 0)
        self._panels_layout.setSpacing(4)
        self._controls_layout.addWidget(self._panels_container)

        # --- Global settings ---
        global_group = QtWidgets.QGroupBox("Global Settings")
        global_layout = QtWidgets.QFormLayout(global_group)
        global_layout.setSpacing(4)

        self._dpi_spin = QtWidgets.QSpinBox()
        self._dpi_spin.setRange(72, 1200)
        self._dpi_spin.setValue(300)
        self._dpi_spin.setSuffix(" dpi")
        global_layout.addRow("DPI:", self._dpi_spin)

        self._bg_combo = QtWidgets.QComboBox()
        self._bg_combo.addItems(["White", "Black", "Gray"])
        global_layout.addRow("Background:", self._bg_combo)

        self._padding_spin = QtWidgets.QSpinBox()
        self._padding_spin.setRange(0, 100)
        self._padding_spin.setValue(10)
        self._padding_spin.setSuffix(" px")
        global_layout.addRow("Padding:", self._padding_spin)

        self._font_combo = QtWidgets.QComboBox()
        self._font_combo.addItems(["Arial", "Helvetica", "Times"])
        global_layout.addRow("Font:", self._font_combo)

        self._fontsize_spin = QtWidgets.QSpinBox()
        self._fontsize_spin.setRange(6, 72)
        self._fontsize_spin.setValue(12)
        self._fontsize_spin.setSuffix(" pt")
        global_layout.addRow("Font size:", self._fontsize_spin)

        self._controls_layout.addWidget(global_group)

        # --- Annotations ---
        annot_group = QtWidgets.QGroupBox("Annotations")
        annot_layout = QtWidgets.QHBoxLayout(annot_group)
        self._add_text_btn = QtWidgets.QPushButton("Add Text")
        self._add_arrow_btn = QtWidgets.QPushButton("Add Arrow")
        annot_layout.addWidget(self._add_text_btn)
        annot_layout.addWidget(self._add_arrow_btn)
        self._controls_layout.addWidget(annot_group)

        # --- Export ---
        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)

        self._update_btn = QtWidgets.QPushButton("Update Preview")
        self._update_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }")
        export_layout.addWidget(self._update_btn)

        btn_row = QtWidgets.QHBoxLayout()
        self._export_png_btn = QtWidgets.QPushButton("Export PNG")
        self._export_svg_btn = QtWidgets.QPushButton("Export SVG")
        self._export_pdf_btn = QtWidgets.QPushButton("Export PDF")
        btn_row.addWidget(self._export_png_btn)
        btn_row.addWidget(self._export_svg_btn)
        btn_row.addWidget(self._export_pdf_btn)
        export_layout.addLayout(btn_row)

        self._controls_layout.addWidget(export_group)

        # Spacer at bottom
        self._controls_layout.addStretch()

        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)

        main_layout.addWidget(right_widget, stretch=1)

    # ----- signal connections -----

    def _connect_signals(self):
        self._rows_spin.valueChanged.connect(self._rebuild_panel_controls)
        self._cols_spin.valueChanged.connect(self._rebuild_panel_controls)
        self._update_btn.clicked.connect(self._update_preview)
        self._export_png_btn.clicked.connect(self._on_export_png)
        self._export_svg_btn.clicked.connect(self._on_export_svg)
        self._export_pdf_btn.clicked.connect(self._on_export_pdf)
        self._add_text_btn.clicked.connect(self._on_add_text)
        self._add_arrow_btn.clicked.connect(self._on_add_arrow)

    # ----- panel controls -----

    def _rebuild_panel_controls(self):
        """Regenerate the per-cell panel controls based on the grid size."""
        # Remove existing controls
        for ctrl in self._panel_controls:
            ctrl.setParent(None)
            ctrl.deleteLater()
        self._panel_controls.clear()

        rows = self._rows_spin.value()
        cols = self._cols_spin.value()

        for r in range(rows):
            for c in range(cols):
                ctrl = PanelControlWidget(r, c, self._panels_container)
                ctrl.populate_windows()
                ctrl.changed.connect(self._on_panel_changed)
                self._panels_layout.addWidget(ctrl)
                self._panel_controls.append(ctrl)

    def _on_panel_changed(self):
        """Slot called when any panel control value changes.

        Currently a no-op; preview is updated only via the Update button.
        """
        pass

    # ----- preview -----

    def _get_bg_color(self):
        """Return a ``QColor`` for the current background selection."""
        text = self._bg_combo.currentText()
        if text == "Black":
            return QtGui.QColor(QtCore.Qt.GlobalColor.black)
        elif text == "Gray":
            return QtGui.QColor(128, 128, 128)
        return QtGui.QColor(QtCore.Qt.GlobalColor.white)

    def _collect_panels(self):
        """Collect :class:`FigurePanel` objects from the current controls."""
        panels = []
        for ctrl in self._panel_controls:
            panel = ctrl.get_panel()
            if panel.source_window is not None:
                panels.append(panel)
        return panels

    def _update_preview(self):
        """Re-render the figure preview from the current settings."""
        # Refresh window lists in each panel control
        for ctrl in self._panel_controls:
            ctrl.populate_windows()

        panels = self._collect_panels()

        self._scene.set_layout(
            rows=self._rows_spin.value(),
            cols=self._cols_spin.value(),
            panels=panels,
            dpi=self._dpi_spin.value(),
            bg_color=self._get_bg_color(),
            padding=self._padding_spin.value(),
            font_name=self._font_combo.currentText(),
            font_size=self._fontsize_spin.value(),
        )
        self._scene.render_panels()

        # Fit the view to the scene contents
        self._view.fitInView(self._scene.sceneRect(),
                             QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # ----- annotations -----

    def _on_add_text(self):
        """Add a movable text annotation at the centre of the current view."""
        font = QtGui.QFont(self._font_combo.currentText(),
                           self._fontsize_spin.value())
        annotation = _TextAnnotation("Text", font)
        self._scene.addItem(annotation)

        # Place near centre of visible area
        center = self._view.mapToScene(
            self._view.viewport().rect().center())
        annotation.setPos(center)

    def _on_add_arrow(self):
        """Add a movable arrow annotation at the centre of the current view."""
        center = self._view.mapToScene(
            self._view.viewport().rect().center())
        arrow = _ArrowAnnotation(0, 0, 60, 40)
        self._scene.addItem(arrow)
        arrow.setPos(center)

    # ----- export handlers -----

    def _on_export_png(self):
        """Export the figure as PNG via a file dialog."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG Images (*.png);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith('.png'):
            path += '.png'
        self._ensure_rendered()
        self._scene.export_png(path, self._dpi_spin.value())

    def _on_export_svg(self):
        """Export the figure as SVG via a file dialog."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export SVG", "", "SVG Files (*.svg);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith('.svg'):
            path += '.svg'
        self._ensure_rendered()
        self._scene.export_svg(path)

    def _on_export_pdf(self):
        """Export the figure as PDF via a file dialog."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", "", "PDF Files (*.pdf);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith('.pdf'):
            path += '.pdf'
        self._ensure_rendered()
        self._scene.export_pdf(path)

    def _ensure_rendered(self):
        """Make sure the scene has been rendered before exporting."""
        if self._scene.sceneRect().isEmpty():
            self._update_preview()

    # ----- overrides -----

    def showEvent(self, event):
        """Refresh window lists when the dialog is shown."""
        super().showEvent(event)
        for ctrl in self._panel_controls:
            ctrl.populate_windows()

    def resizeEvent(self, event):
        """Refit the view when the dialog is resized."""
        super().resizeEvent(event)
        if not self._scene.sceneRect().isEmpty():
            self._view.fitInView(self._scene.sceneRect(),
                                 QtCore.Qt.AspectRatioMode.KeepAspectRatio)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def show_figure_composer(parent=None):
    """Open the Figure Composer dialog.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.  If *None*, uses ``g.m`` (the main application window).

    Returns
    -------
    FigureComposerDialog
        The dialog instance (shown non-modally).
    """
    if parent is None:
        try:
            from .. import global_vars as g
            parent = g.m
        except Exception:
            pass

    dlg = FigureComposerDialog(parent)
    dlg.show()
    return dlg


logger.debug("Completed 'reading viewers/figure_composer.py'")
