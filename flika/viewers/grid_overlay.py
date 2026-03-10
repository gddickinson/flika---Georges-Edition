"""Grid and guide overlays for flika windows.

Provides configurable grid overlays for microscopy:
  - Rectangular grids (fixed spacing or divisions)
  - Crosshair / center marker
  - Rule-of-thirds
  - Dot grid (intersection points only)
  - Polar / radial grid
  - Custom ruler lines with distance readout

All overlays are non-destructive graphics items added to the ViewBox.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import flika.global_vars as g
from flika.logger import logger


# ---------------------------------------------------------------------------
# Preset colors
# ---------------------------------------------------------------------------

_GRID_COLORS = {
    'White': (255, 255, 255),
    'Yellow': (255, 255, 0),
    'Cyan': (0, 255, 255),
    'Green': (0, 255, 0),
    'Red': (255, 0, 0),
    'Magenta': (255, 0, 255),
    'Gray': (128, 128, 128),
    'Light Gray': (192, 192, 192),
    'Blue': (0, 100, 255),
}


def _make_pen(color_name: str, width: float = 1.0, opacity: float = 0.5,
              style: str = 'solid') -> QtGui.QPen:
    """Create a QPen from preset name or hex."""
    if color_name in _GRID_COLORS:
        r, g_, b = _GRID_COLORS[color_name]
    else:
        c = QtGui.QColor(color_name)
        r, g_, b = c.red(), c.green(), c.blue()
    color = QtGui.QColor(r, g_, b, int(opacity * 255))
    pen = QtGui.QPen(color)
    pen.setWidthF(width)
    pen.setCosmetic(True)
    styles = {
        'solid': QtCore.Qt.SolidLine,
        'dash': QtCore.Qt.DashLine,
        'dot': QtCore.Qt.DotLine,
        'dash-dot': QtCore.Qt.DashDotLine,
    }
    pen.setStyle(styles.get(style, QtCore.Qt.SolidLine))
    return pen


# ---------------------------------------------------------------------------
# Grid Overlay
# ---------------------------------------------------------------------------

class GridOverlay:
    """Manages grid graphics items on a flika Window.

    Parameters
    ----------
    window : flika.window.Window
        Target window.
    """

    def __init__(self, window):
        self.window = window
        self._items: list[QtWidgets.QGraphicsItem] = []
        self._props: dict = {}

    def clear(self):
        """Remove all grid items from the view."""
        view = self.window.imageview.view
        for item in self._items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._items.clear()

    @property
    def visible(self) -> bool:
        return bool(self._items) and all(item.isVisible() for item in self._items)

    def set_visible(self, vis: bool):
        for item in self._items:
            item.setVisible(vis)

    def draw_rectangular(self, spacing_x: float = 50, spacing_y: float = 50,
                         color: str = 'Yellow', width: float = 1.0,
                         opacity: float = 0.4, style: str = 'solid',
                         divisions_mode: bool = False,
                         divisions_x: int = 4, divisions_y: int = 4):
        """Draw a rectangular grid.

        Parameters
        ----------
        spacing_x, spacing_y : float
            Pixel spacing between grid lines (ignored if divisions_mode).
        divisions_mode : bool
            If True, use divisions_x/y instead of spacing.
        divisions_x, divisions_y : int
            Number of divisions (used if divisions_mode).
        """
        self.clear()
        w, h = self.window.mx, self.window.my
        pen = _make_pen(color, width, opacity, style)
        view = self.window.imageview.view

        if divisions_mode:
            spacing_x = w / divisions_x if divisions_x > 0 else w
            spacing_y = h / divisions_y if divisions_y > 0 else h

        # Vertical lines
        x = spacing_x
        while x < w:
            line = QtWidgets.QGraphicsLineItem(x, 0, x, h)
            line.setPen(pen)
            line.setZValue(15)
            view.addItem(line)
            self._items.append(line)
            x += spacing_x

        # Horizontal lines
        y = spacing_y
        while y < h:
            line = QtWidgets.QGraphicsLineItem(0, y, w, y)
            line.setPen(pen)
            line.setZValue(15)
            view.addItem(line)
            self._items.append(line)
            y += spacing_y

        self._props = {
            'type': 'rectangular', 'spacing_x': spacing_x, 'spacing_y': spacing_y,
            'color': color, 'width': width, 'opacity': opacity, 'style': style,
        }

    def draw_crosshair(self, color: str = 'Yellow', width: float = 1.5,
                       opacity: float = 0.6, style: str = 'solid',
                       show_circle: bool = False, circle_radius: float = 20):
        """Draw center crosshair lines spanning the full image."""
        self.clear()
        w, h = self.window.mx, self.window.my
        cx, cy = w / 2, h / 2
        pen = _make_pen(color, width, opacity, style)
        view = self.window.imageview.view

        hline = QtWidgets.QGraphicsLineItem(0, cy, w, cy)
        hline.setPen(pen)
        hline.setZValue(15)
        view.addItem(hline)
        self._items.append(hline)

        vline = QtWidgets.QGraphicsLineItem(cx, 0, cx, h)
        vline.setPen(pen)
        vline.setZValue(15)
        view.addItem(vline)
        self._items.append(vline)

        if show_circle:
            ellipse = QtWidgets.QGraphicsEllipseItem(
                cx - circle_radius, cy - circle_radius,
                2 * circle_radius, 2 * circle_radius)
            ellipse.setPen(pen)
            ellipse.setZValue(15)
            view.addItem(ellipse)
            self._items.append(ellipse)

        self._props = {
            'type': 'crosshair', 'color': color, 'width': width,
            'opacity': opacity, 'style': style,
            'show_circle': show_circle, 'circle_radius': circle_radius,
        }

    def draw_thirds(self, color: str = 'Yellow', width: float = 1.0,
                    opacity: float = 0.5, style: str = 'solid'):
        """Draw rule-of-thirds grid (3x3 divisions)."""
        self.draw_rectangular(
            divisions_mode=True, divisions_x=3, divisions_y=3,
            color=color, width=width, opacity=opacity, style=style)
        self._props['type'] = 'thirds'

    def draw_dot_grid(self, spacing_x: float = 50, spacing_y: float = 50,
                      color: str = 'Yellow', dot_size: float = 3,
                      opacity: float = 0.6):
        """Draw dots at grid intersection points."""
        self.clear()
        w, h = self.window.mx, self.window.my
        view = self.window.imageview.view

        if color in _GRID_COLORS:
            r, g_, b = _GRID_COLORS[color]
        else:
            c = QtGui.QColor(color)
            r, g_, b = c.red(), c.green(), c.blue()

        xs = np.arange(spacing_x, w, spacing_x)
        ys = np.arange(spacing_y, h, spacing_y)
        if len(xs) == 0 or len(ys) == 0:
            return

        xg, yg = np.meshgrid(xs, ys)
        spots = [{'pos': (float(xg[i, j]), float(yg[i, j])), 'size': dot_size}
                 for i in range(yg.shape[0]) for j in range(xg.shape[1])]

        scatter = pg.ScatterPlotItem(
            spots=spots,
            brush=pg.mkBrush(r, g_, b, int(opacity * 255)),
            pen=pg.mkPen(None),
            pxMode=True,
        )
        scatter.setZValue(15)
        view.addItem(scatter)
        self._items.append(scatter)

        self._props = {
            'type': 'dot', 'spacing_x': spacing_x, 'spacing_y': spacing_y,
            'color': color, 'dot_size': dot_size, 'opacity': opacity,
        }

    def draw_polar(self, n_rings: int = 5, n_spokes: int = 8,
                   color: str = 'Cyan', width: float = 1.0,
                   opacity: float = 0.4, style: str = 'solid'):
        """Draw a polar/radial grid centered on the image."""
        self.clear()
        w, h = self.window.mx, self.window.my
        cx, cy = w / 2, h / 2
        max_r = min(w, h) / 2
        pen = _make_pen(color, width, opacity, style)
        view = self.window.imageview.view

        # Concentric rings
        for i in range(1, n_rings + 1):
            r = max_r * i / n_rings
            ellipse = QtWidgets.QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
            ellipse.setPen(pen)
            ellipse.setZValue(15)
            view.addItem(ellipse)
            self._items.append(ellipse)

        # Radial spokes
        for i in range(n_spokes):
            angle = 2 * np.pi * i / n_spokes
            x1 = cx + max_r * np.cos(angle)
            y1 = cy + max_r * np.sin(angle)
            line = QtWidgets.QGraphicsLineItem(cx, cy, x1, y1)
            line.setPen(pen)
            line.setZValue(15)
            view.addItem(line)
            self._items.append(line)

        self._props = {
            'type': 'polar', 'n_rings': n_rings, 'n_spokes': n_spokes,
            'color': color, 'width': width, 'opacity': opacity, 'style': style,
        }

    def draw_ruler(self, x1: float, y1: float, x2: float, y2: float,
                   color: str = 'Yellow', width: float = 2.0,
                   opacity: float = 0.8, show_distance: bool = True,
                   pixel_size: Optional[float] = None,
                   unit: str = '\u00b5m'):
        """Draw a measurement ruler line with distance label.

        Parameters
        ----------
        pixel_size : float, optional
            nm per pixel. If None, uses g.settings['pixel_size'].
        """
        view = self.window.imageview.view
        pen = _make_pen(color, width, opacity, 'solid')

        line = QtWidgets.QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(pen)
        line.setZValue(16)
        view.addItem(line)
        self._items.append(line)

        # Endpoint markers
        for (px, py) in [(x1, y1), (x2, y2)]:
            marker = QtWidgets.QGraphicsEllipseItem(px - 3, py - 3, 6, 6)
            marker.setPen(pen)
            brush_color = QtGui.QColor(pen.color())
            brush_color.setAlpha(int(opacity * 200))
            marker.setBrush(brush_color)
            marker.setZValue(16)
            view.addItem(marker)
            self._items.append(marker)

        if show_distance:
            dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if pixel_size is None:
                pixel_size = g.settings.get('pixel_size', None)
            if pixel_size and unit != 'px':
                # pixel_size is in nm
                if unit == '\u00b5m':
                    dist_phys = dist_px * pixel_size / 1000
                elif unit == 'nm':
                    dist_phys = dist_px * pixel_size
                elif unit == 'mm':
                    dist_phys = dist_px * pixel_size / 1e6
                else:
                    dist_phys = dist_px
                    unit = 'px'
                label_text = f'{dist_phys:.2f} {unit}'
            else:
                label_text = f'{dist_px:.1f} px'

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            text = pg.TextItem(label_text, color=pen.color(), anchor=(0.5, 1))
            text.setPos(mid_x, mid_y - 3)
            text.setZValue(16)
            view.addItem(text)
            self._items.append(text)

    def draw_angle(self, x1: float, y1: float, x2: float, y2: float,
                   x3: float, y3: float, color: str = 'Yellow',
                   width: float = 2.0, opacity: float = 0.8):
        """Draw an angle measurement between three points (vertex at x2,y2)."""
        view = self.window.imageview.view
        pen = _make_pen(color, width, opacity, 'solid')

        # Two lines meeting at vertex
        line1 = QtWidgets.QGraphicsLineItem(x1, y1, x2, y2)
        line1.setPen(pen)
        line1.setZValue(16)
        view.addItem(line1)
        self._items.append(line1)

        line2 = QtWidgets.QGraphicsLineItem(x2, y2, x3, y3)
        line2.setPen(pen)
        line2.setZValue(16)
        view.addItem(line2)
        self._items.append(line2)

        # Compute angle
        v1 = np.array([x1 - x2, y1 - y2], dtype=float)
        v2 = np.array([x3 - x2, y3 - y2], dtype=float)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

        text = pg.TextItem(f'{angle_deg:.1f}\u00b0', color=pen.color(), anchor=(0.5, 1))
        text.setPos(x2, y2 - 5)
        text.setZValue(16)
        view.addItem(text)
        self._items.append(text)

    def get_props(self) -> dict:
        """Return current grid properties for baking."""
        return dict(self._props)


# ---------------------------------------------------------------------------
# Text Annotation
# ---------------------------------------------------------------------------

class TextAnnotation:
    """A positioned text label on a window.

    Parameters
    ----------
    window : flika.window.Window
    text : str
    x, y : float
        Position in image coordinates.
    """

    def __init__(self, window, text: str, x: float, y: float,
                 font_size: int = 12, color: str = 'White',
                 bg_color: str = 'None', bold: bool = False,
                 frame: Optional[int] = None):
        self.window = window
        self.text = text
        self.x = x
        self.y = y
        self.font_size = font_size
        self.color = color
        self.bg_color = bg_color
        self.bold = bold
        self.frame = frame  # None = visible on all frames
        self._item: Optional[pg.TextItem] = None
        self._create()

    def _create(self):
        from flika.process.overlay import _color_to_css
        color_css = _color_to_css(self.color)
        bg_css = _color_to_css(self.bg_color)
        bold_css = 'font-weight:bold;' if self.bold else ''
        html = (f"<span style='font-size:{self.font_size}pt; color:{color_css}; "
                f"background-color:{bg_css}; {bold_css} padding:2px;'>"
                f"{self.text}</span>")
        self._item = pg.TextItem(html=html, anchor=(0, 0))
        self._item.setPos(self.x, self.y)
        self._item.setZValue(17)
        self.window.imageview.view.addItem(self._item)

    def remove(self):
        if self._item is not None:
            try:
                self.window.imageview.view.removeItem(self._item)
            except Exception:
                pass
            self._item = None

    def set_visible(self, vis: bool):
        if self._item:
            self._item.setVisible(vis)

    def update_for_frame(self, frame_idx: int):
        """Show/hide based on frame association."""
        if self._item is None:
            return
        if self.frame is None:
            self._item.setVisible(True)
        else:
            self._item.setVisible(frame_idx == self.frame)

    def to_dict(self) -> dict:
        return {
            'text': self.text, 'x': self.x, 'y': self.y,
            'font_size': self.font_size, 'color': self.color,
            'bg_color': self.bg_color, 'bold': self.bold,
            'frame': self.frame,
        }
