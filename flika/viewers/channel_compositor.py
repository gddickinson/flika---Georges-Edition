# -*- coding: utf-8 -*-
"""
Channel Compositor
==================

Overlay multiple grayscale window images with different colormaps (LUTs) for
multi-channel composite viewing.  Each source window is represented as a
:class:`ChannelLayer` that owns a :class:`pyqtgraph.ImageItem` rendered into
the host window's ViewBox.  The :class:`ChannelCompositor` manages the layer
stack and provides additive-blend export to RGB.

Usage::

    from flika.viewers.channel_compositor import ChannelCompositor
    compositor = ChannelCompositor(host_window)
    compositor.add_channel(green_window, 'Green')
    compositor.add_channel(red_window, 'Red')

"""
from flika.logger import logger
logger.debug("Started 'reading viewers/channel_compositor.py'")

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QObject, Signal

# ---------------------------------------------------------------------------
# Colormap construction helpers
# ---------------------------------------------------------------------------

def _build_lut(r_func, g_func, b_func):
    """Build a (256, 4) uint8 RGBA lookup table from three channel functions.

    Each function takes a float array on [0, 1] and returns a float array on
    [0, 1].
    """
    x = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[:, 0] = np.clip(r_func(x) * 255.0, 0, 255).astype(np.uint8)
    lut[:, 1] = np.clip(g_func(x) * 255.0, 0, 255).astype(np.uint8)
    lut[:, 2] = np.clip(b_func(x) * 255.0, 0, 255).astype(np.uint8)
    lut[:, 3] = 255  # fully opaque
    return lut


def _zero(x):
    """Return an array of zeros with the same shape as *x*."""
    return np.zeros_like(x)


def _linear(x):
    """Identity function (passthrough)."""
    return x


def _fire_r(x):
    """Red channel of the Fire colormap (black -> red -> orange -> yellow -> white)."""
    # Ramp up quickly so red is dominant early.
    return np.clip(2.0 * x, 0.0, 1.0)


def _fire_g(x):
    """Green channel of the Fire colormap."""
    # Delayed ramp: starts after ~25 %, reaches 1 at ~75 %.
    return np.clip(2.0 * x - 0.5, 0.0, 1.0)


def _fire_b(x):
    """Blue channel of the Fire colormap."""
    # Only appears in the bright end to push towards white.
    return np.clip(4.0 * x - 3.0, 0.0, 1.0)


def _ice_r(x):
    """Red channel of the Ice colormap (black -> blue -> cyan -> white)."""
    return np.clip(2.0 * x - 1.0, 0.0, 1.0)


def _ice_g(x):
    """Green channel of the Ice colormap."""
    return np.clip(2.0 * x - 0.5, 0.0, 1.0)


def _ice_b(x):
    """Blue channel of the Ice colormap."""
    return np.clip(2.0 * x, 0.0, 1.0)


COLORMAPS = {
    'Green':   _build_lut(_zero,   _linear, _zero),
    'Red':     _build_lut(_linear, _zero,   _zero),
    'Blue':    _build_lut(_zero,   _zero,   _linear),
    'Cyan':    _build_lut(_zero,   _linear, _linear),
    'Magenta': _build_lut(_linear, _zero,   _linear),
    'Yellow':  _build_lut(_linear, _linear, _zero),
    'Gray':    _build_lut(_linear, _linear, _linear),
    'Fire':    _build_lut(_fire_r, _fire_g, _fire_b),
    'Ice':     _build_lut(_ice_r,  _ice_g,  _ice_b),
}

AUTO_COLORMAP_ORDER = ['Green', 'Magenta', 'Red', 'Cyan', 'Blue', 'Yellow']

# ---------------------------------------------------------------------------
# ChannelLayer
# ---------------------------------------------------------------------------

class ChannelLayer(QObject):
    """A single channel in the composite overlay.

    Each layer wraps a source :class:`~flika.window.Window` and renders its
    current frame through a chosen LUT into a :class:`pyqtgraph.ImageItem`.

    Signals
    -------
    sigChanged
        Emitted whenever the layer appearance changes (colormap, levels,
        opacity, visibility).
    """

    sigChanged = Signal()

    def __init__(self, source_window, colormap_name='Green', parent=None):
        super().__init__(parent)
        self.source_window = source_window
        self.colormap_name = colormap_name
        self.lut = COLORMAPS[colormap_name].copy()
        self.min_level = 0.0
        self.max_level = 255.0
        self.opacity = 1.0
        self.visible = True

        # The ImageItem that will be added to the host view.
        self.image_item = pg.ImageItem()
        self.image_item.setLookupTable(self.lut)
        self.image_item.setOpts(axisOrder='row-major')
        self.image_item.setCompositionMode(
            pg.QtGui.QPainter.CompositionMode.CompositionMode_Plus
        )

        # Auto-detect initial levels from the source.
        self._auto_levels()

    # ----- public API -----

    def set_colormap(self, name):
        """Change the colormap to *name* (must be a key in :data:`COLORMAPS`).
        """
        if name not in COLORMAPS:
            raise ValueError(
                f"Unknown colormap '{name}'. "
                f"Available: {list(COLORMAPS.keys())}"
            )
        self.colormap_name = name
        self.lut = COLORMAPS[name].copy()
        self.image_item.setLookupTable(self.lut)
        self.sigChanged.emit()

    def get_current_frame(self):
        """Return the current 2-D (y, x) float64 frame from the source window.

        Handles 2-D, 3-D (t, y, x) and RGB images.  For RGB data the colour
        channels are averaged to produce a single grayscale plane.
        """
        img = self.source_window.image
        if img is None:
            return None

        ndim = img.ndim

        if ndim == 2:
            frame = img
        elif ndim == 3:
            # Could be (t, y, x) or (y, x, 3/4)
            if img.shape[2] in (3, 4):
                # RGB(A) single frame — average colour channels.
                frame = np.mean(img[:, :, :3].astype(np.float64), axis=2)
            else:
                frame = img[self.source_window.currentIndex]
        elif ndim == 4:
            # (t, y, x, 3/4) — time-series RGB
            frame = img[self.source_window.currentIndex]
            if frame.ndim == 3 and frame.shape[2] in (3, 4):
                frame = np.mean(frame[:, :, :3].astype(np.float64), axis=2)
        else:
            frame = img

        return np.asarray(frame, dtype=np.float64)

    def update_display(self):
        """Re-render the layer from the current source frame."""
        if not self.visible:
            return

        frame = self.get_current_frame()
        if frame is None:
            return

        self.image_item.setImage(frame, autoLevels=False)
        self.image_item.setLevels([self.min_level, self.max_level])
        self.image_item.setLookupTable(self.lut)
        self.image_item.setOpacity(self.opacity)

    def set_levels(self, min_val, max_val):
        """Set the brightness / contrast range for this layer."""
        self.min_level = float(min_val)
        self.max_level = float(max_val)
        self.image_item.setLevels([self.min_level, self.max_level])
        self.sigChanged.emit()

    def set_opacity(self, val):
        """Set the layer opacity (0.0 = transparent, 1.0 = opaque)."""
        self.opacity = float(np.clip(val, 0.0, 1.0))
        self.image_item.setOpacity(self.opacity)
        self.sigChanged.emit()

    def set_visible(self, vis):
        """Show or hide the layer."""
        self.visible = bool(vis)
        if self.visible:
            self.image_item.show()
            self.update_display()
        else:
            self.image_item.hide()
        self.sigChanged.emit()

    # ----- internal helpers -----

    def _auto_levels(self):
        """Set *min_level* / *max_level* from the first available frame."""
        frame = self.get_current_frame()
        if frame is not None and frame.size > 0:
            self.min_level = float(np.nanmin(frame))
            self.max_level = float(np.nanmax(frame))
            # Guard against flat images.
            if self.min_level == self.max_level:
                self.max_level = self.min_level + 1.0

    def __repr__(self):
        src_name = getattr(self.source_window, 'name', '?')
        return (
            f"<ChannelLayer src={src_name!r} cmap={self.colormap_name!r} "
            f"visible={self.visible} opacity={self.opacity:.2f}>"
        )


# ---------------------------------------------------------------------------
# ChannelCompositor
# ---------------------------------------------------------------------------

class ChannelCompositor(QObject):
    """Manages a stack of :class:`ChannelLayer` objects composited into a host
    window's :class:`pyqtgraph.ViewBox`.

    The host window's own :class:`~pyqtgraph.ImageItem` is hidden while the
    compositor is active; all visual content comes from the per-layer image
    items.

    Signals
    -------
    sigLayersChanged
        Emitted when a layer is added or removed.
    sigCompositeUpdated
        Emitted after all visible layers have been re-rendered.
    """

    sigLayersChanged = Signal()
    sigCompositeUpdated = Signal()

    def __init__(self, host_window):
        super().__init__()
        self.host_window = host_window
        self.layers = []
        self._solo_layer = None
        self._colormap_index = 0  # tracks auto-cycling position

        # Hide the host window's own image so only our layers show.
        self._host_image_item = host_window.imageview.imageItem
        self._host_image_item_was_visible = self._host_image_item.isVisible()
        self._host_image_item.hide()

        # Clean up when the host window closes.
        self._host_close_connected = False
        try:
            host_window.closeSignal.connect(self._on_host_closed)
            self._host_close_connected = True
        except (AttributeError, RuntimeError):
            pass

        logger.debug("ChannelCompositor created for host %r", getattr(host_window, 'name', '?'))

    # ----- layer management -----

    def add_channel(self, source_window, colormap_name=None):
        """Add a new channel layer for *source_window*.

        Parameters
        ----------
        source_window : Window
            The flika window whose image data will be used.
        colormap_name : str or None
            Key into :data:`COLORMAPS`.  If *None*, the next colour from
            :data:`AUTO_COLORMAP_ORDER` is used automatically.

        Returns
        -------
        ChannelLayer
            The newly created layer.
        """
        if colormap_name is None:
            colormap_name = AUTO_COLORMAP_ORDER[
                self._colormap_index % len(AUTO_COLORMAP_ORDER)
            ]
            self._colormap_index += 1

        layer = ChannelLayer(source_window, colormap_name, parent=self)

        # Add the image item to the host view.
        self.host_window.imageview.view.addItem(layer.image_item)

        # Connect source time changes so overlay follows source playback.
        self._connect_source(layer, source_window)

        self.layers.append(layer)

        # Initial render.
        layer.update_display()

        self.sigLayersChanged.emit()
        logger.debug(
            "Added channel: %r -> host %r",
            getattr(source_window, 'name', '?'),
            getattr(self.host_window, 'name', '?'),
        )
        return layer

    def remove_channel(self, layer):
        """Remove *layer* from the composite.

        The layer's :class:`~pyqtgraph.ImageItem` is removed from the host
        view and all signal connections are severed.
        """
        if layer not in self.layers:
            return

        # If this layer was soloed, clear solo state.
        if self._solo_layer is layer:
            self._solo_layer = None

        # Remove overlay image item from the host view.
        try:
            self.host_window.imageview.view.removeItem(layer.image_item)
        except (RuntimeError, AttributeError):
            pass

        # Disconnect signals.
        self._disconnect_source(layer)

        self.layers.remove(layer)
        self.sigLayersChanged.emit()
        logger.debug("Removed channel layer: %r", layer)

    def _connect_source(self, layer, source_window):
        """Connect source window signals to update the layer."""
        # Store references for later disconnection.
        layer._time_slot = lambda idx, _l=layer: _l.update_display()
        layer._close_slot = lambda _l=layer: self.remove_channel(_l)
        layer._time_connected = False
        layer._close_connected = False

        try:
            source_window.sigTimeChanged.connect(layer._time_slot)
            layer._time_connected = True
        except (AttributeError, RuntimeError):
            pass

        try:
            source_window.closeSignal.connect(layer._close_slot)
            layer._close_connected = True
        except (AttributeError, RuntimeError):
            pass

    def _disconnect_source(self, layer):
        """Disconnect all signal connections for *layer*."""
        sw = layer.source_window
        if sw is None:
            return

        if getattr(layer, '_time_connected', False):
            try:
                sw.sigTimeChanged.disconnect(layer._time_slot)
            except (RuntimeError, TypeError):
                pass
            layer._time_connected = False

        if getattr(layer, '_close_connected', False):
            try:
                sw.closeSignal.disconnect(layer._close_slot)
            except (RuntimeError, TypeError):
                pass
            layer._close_connected = False

    # ----- solo -----

    def set_solo(self, layer):
        """Solo *layer* — hide all other layers, showing only this one.

        Parameters
        ----------
        layer : ChannelLayer
            Must already be in :attr:`layers`.
        """
        if layer not in self.layers:
            return
        self._solo_layer = layer
        for l in self.layers:
            if l is layer:
                l.image_item.show()
                l.update_display()
            else:
                l.image_item.hide()
        self.sigCompositeUpdated.emit()

    def unsolo(self):
        """Remove the solo state — restore each layer's own visibility."""
        self._solo_layer = None
        for layer in self.layers:
            if layer.visible:
                layer.image_item.show()
                layer.update_display()
            else:
                layer.image_item.hide()
        self.sigCompositeUpdated.emit()

    # ----- composite update -----

    def update_composite(self):
        """Re-render all visible layers.

        This is typically called in response to a time change on the host or
        source windows.
        """
        for layer in self.layers:
            if self._solo_layer is not None:
                # In solo mode, only update the soloed layer.
                if layer is self._solo_layer:
                    layer.update_display()
            elif layer.visible:
                layer.update_display()
        self.sigCompositeUpdated.emit()

    # ----- RGB export -----

    def export_composite_rgb(self):
        """Render the current composite into a single (H, W, 3) uint8 array.

        Layers are blended additively: each layer's normalised frame is
        mapped through its LUT, scaled by opacity, and summed.  The result is
        clipped to [0, 255].

        Returns
        -------
        numpy.ndarray
            RGB image of shape (H, W, 3) and dtype ``uint8``.
        """
        # Determine spatial shape from the host window.
        host_img = self.host_window.image
        if host_img is None:
            raise RuntimeError("Host window has no image data.")

        if host_img.ndim == 2:
            h, w = host_img.shape
        elif host_img.ndim == 3:
            if host_img.shape[2] in (3, 4):
                h, w = host_img.shape[:2]
            else:
                h, w = host_img.shape[1], host_img.shape[2]
        elif host_img.ndim == 4:
            h, w = host_img.shape[1], host_img.shape[2]
        else:
            h, w = host_img.shape[-2], host_img.shape[-1]

        composite = np.zeros((h, w, 3), dtype=np.float64)

        for layer in self.layers:
            if not layer.visible:
                continue
            if self._solo_layer is not None and layer is not self._solo_layer:
                continue

            frame = layer.get_current_frame()
            if frame is None:
                continue

            # Ensure frame matches composite shape, padding or cropping if
            # the source has a different size.
            fh, fw = frame.shape[:2]
            if fh != h or fw != w:
                out = np.zeros((h, w), dtype=np.float64)
                mh, mw = min(fh, h), min(fw, w)
                out[:mh, :mw] = frame[:mh, :mw]
                frame = out

            # Normalise to [0, 255] using the layer's levels.
            span = layer.max_level - layer.min_level
            if span <= 0:
                span = 1.0
            normalised = (frame - layer.min_level) / span * 255.0
            normalised = np.clip(normalised, 0.0, 255.0)

            # Map through the LUT.  Use integer indices into the 256-entry
            # table.
            indices = normalised.astype(np.intp)
            np.clip(indices, 0, 255, out=indices)
            lut = layer.lut  # (256, 4) uint8
            mapped = lut[indices][:, :, :3].astype(np.float64)  # (H, W, 3)

            # Scale by layer opacity.
            mapped *= layer.opacity

            composite += mapped

        np.clip(composite, 0.0, 255.0, out=composite)
        return composite.astype(np.uint8)

    # ----- cleanup -----

    def cleanup(self):
        """Remove all layers and restore the host window to its original state.

        This disconnects all signals and removes all overlay image items from
        the host view.
        """
        # Remove layers in reverse to avoid index shifting issues.
        for layer in list(reversed(self.layers)):
            self.remove_channel(layer)

        self._solo_layer = None

        # Restore host image item visibility.
        try:
            if self._host_image_item_was_visible:
                self._host_image_item.show()
        except RuntimeError:
            pass  # underlying C++ object already deleted

        # Disconnect host close signal.
        if self._host_close_connected:
            try:
                self.host_window.closeSignal.disconnect(self._on_host_closed)
            except (RuntimeError, TypeError):
                pass
            self._host_close_connected = False

        logger.debug("ChannelCompositor cleaned up for host %r",
                      getattr(self.host_window, 'name', '?'))

    def _on_host_closed(self):
        """Slot called when the host window emits ``closeSignal``."""
        self.cleanup()

    # ----- convenience queries -----

    @property
    def visible_layers(self):
        """Return the list of currently visible layers."""
        if self._solo_layer is not None:
            return [self._solo_layer]
        return [l for l in self.layers if l.visible]

    @property
    def layer_count(self):
        """Number of layers currently managed."""
        return len(self.layers)

    def get_layer_by_source(self, source_window):
        """Return the first layer whose source is *source_window*, or *None*.
        """
        for layer in self.layers:
            if layer.source_window is source_window:
                return layer
        return None

    def move_layer(self, layer, new_index):
        """Move *layer* to *new_index* in the layer stack.

        A lower index means the layer is rendered first (further back).
        """
        if layer not in self.layers:
            return
        self.layers.remove(layer)
        self.layers.insert(new_index, layer)

        # Re-stack the image items in the view so draw order matches.
        for i, l in enumerate(self.layers):
            l.image_item.setZValue(i)

        self.sigLayersChanged.emit()

    def __repr__(self):
        host_name = getattr(self.host_window, 'name', '?')
        return (
            f"<ChannelCompositor host={host_name!r} "
            f"layers={self.layer_count}>"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def quick_composite(host_window, source_windows, colormaps=None):
    """Create a :class:`ChannelCompositor` and add multiple channels at once.

    Parameters
    ----------
    host_window : Window
        The window to composite onto.
    source_windows : list[Window]
        List of source windows providing image data.
    colormaps : list[str] or None
        Colormap names for each source.  If *None*, the auto-cycle order is
        used.

    Returns
    -------
    ChannelCompositor
        The fully initialised compositor.
    """
    compositor = ChannelCompositor(host_window)
    for i, src in enumerate(source_windows):
        cmap = colormaps[i] if colormaps and i < len(colormaps) else None
        compositor.add_channel(src, cmap)
    return compositor


logger.debug("Completed 'reading viewers/channel_compositor.py'")
