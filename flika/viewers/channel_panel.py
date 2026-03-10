# -*- coding: utf-8 -*-
"""
Channel Panel
=============

Dockable UI panel for the :class:`ChannelCompositor`.  Provides per-channel
controls (colormap, levels, opacity, visibility) and buttons for adding
channels, exporting the composite, and tearing down the compositor.

Each channel source window is represented by a :class:`ChannelRowWidget` with
controls bound bidirectionally to the underlying :class:`ChannelLayer`.

Usage::

    from flika.viewers.channel_panel import ChannelPanel
    panel = ChannelPanel.instance(parent=g.m)
    panel.set_compositor(compositor)
    panel.show()

"""
from flika.logger import logger
logger.debug("Started 'reading viewers/channel_panel.py'")

import numpy as np
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Signal

from flika.viewers.channel_compositor import ChannelCompositor, COLORMAPS, AUTO_COLORMAP_ORDER
from flika.utils.singleton import DockSingleton


# ---------------------------------------------------------------------------
# ChannelRowWidget
# ---------------------------------------------------------------------------

class ChannelRowWidget(QtWidgets.QWidget):
    """A single row of controls for one :class:`ChannelLayer`.

    The row contains:

    - Name label (source window name, truncated to 20 characters)
    - Visible checkbox
    - Colormap combo box
    - Min / max level spin boxes
    - Opacity slider with percentage label
    - Remove button

    All controls are wired bidirectionally to the layer so that user edits
    propagate to the layer and programmatic layer changes could be reflected
    back (though the primary flow is UI -> layer).

    Signals
    -------
    sigChanged
        Emitted whenever any control value changes.
    sigRemoveRequested
        Emitted with *self* when the user clicks the remove button.
    """

    sigChanged = Signal()
    sigRemoveRequested = Signal(object)

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        self._updating = False  # guard against signal loops

        self._init_ui()
        self._set_initial_values()
        self._connect_signals()

    # ----- UI construction -----

    def _init_ui(self):
        """Build all child widgets and lay them out horizontally."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Source window name
        self.name_label = QtWidgets.QLabel()
        self.name_label.setFixedWidth(130)
        self.name_label.setToolTip('')
        layout.addWidget(self.name_label)

        # Visible checkbox
        self.visible_cb = QtWidgets.QCheckBox('Vis')
        self.visible_cb.setToolTip('Toggle channel visibility')
        layout.addWidget(self.visible_cb)

        # Colormap combo
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.setToolTip('Channel colormap / LUT')
        for name in COLORMAPS:
            self.colormap_combo.addItem(name)
        self.colormap_combo.setFixedWidth(90)
        layout.addWidget(self.colormap_combo)

        # Min level spin
        min_label = QtWidgets.QLabel('Min:')
        layout.addWidget(min_label)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setRange(-1e6, 1e6)
        self.min_spin.setDecimals(2)
        self.min_spin.setSingleStep(1.0)
        self.min_spin.setToolTip('Minimum display level (black point)')
        self.min_spin.setFixedWidth(80)
        layout.addWidget(self.min_spin)

        # Max level spin
        max_label = QtWidgets.QLabel('Max:')
        layout.addWidget(max_label)
        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin.setRange(-1e6, 1e6)
        self.max_spin.setDecimals(2)
        self.max_spin.setSingleStep(1.0)
        self.max_spin.setToolTip('Maximum display level (white point)')
        self.max_spin.setFixedWidth(80)
        layout.addWidget(self.max_spin)

        # Opacity slider
        opacity_label_left = QtWidgets.QLabel('Op:')
        layout.addWidget(opacity_label_left)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip('Layer opacity (0 %% = transparent, 100 %% = opaque)')
        self.opacity_slider.setFixedWidth(80)
        layout.addWidget(self.opacity_slider)

        self.opacity_label = QtWidgets.QLabel('100%')
        self.opacity_label.setFixedWidth(35)
        self.opacity_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.opacity_label)

        # Remove button
        self.remove_btn = QtWidgets.QPushButton('X')
        self.remove_btn.setFixedWidth(24)
        self.remove_btn.setToolTip('Remove this channel')
        layout.addWidget(self.remove_btn)

        self.setLayout(layout)

    # ----- initial values from layer -----

    def _set_initial_values(self):
        """Populate controls from the current layer state."""
        layer = self.layer

        # Name label — truncate to 20 characters
        src_name = getattr(layer.source_window, 'name', '???')
        display_name = src_name if len(src_name) <= 20 else '...' + src_name[-17:]
        self.name_label.setText(display_name)
        self.name_label.setToolTip(src_name)

        # Visible
        self.visible_cb.setChecked(layer.visible)

        # Colormap
        idx = self.colormap_combo.findText(layer.colormap_name)
        if idx >= 0:
            self.colormap_combo.setCurrentIndex(idx)

        # Levels
        self.min_spin.setValue(layer.min_level)
        self.max_spin.setValue(layer.max_level)

        # Opacity
        opacity_pct = int(round(layer.opacity * 100.0))
        self.opacity_slider.setValue(opacity_pct)
        self.opacity_label.setText(f'{opacity_pct}%')

    # ----- signal wiring -----

    def _connect_signals(self):
        """Wire control changes to layer setters and local signals."""
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.min_spin.valueChanged.connect(self._on_levels_changed)
        self.max_spin.valueChanged.connect(self._on_levels_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.visible_cb.toggled.connect(self._on_visible_changed)
        self.remove_btn.clicked.connect(self._on_remove_clicked)

    # ----- slots -----

    def _on_colormap_changed(self, name):
        """Handle colormap combo box change."""
        if self._updating:
            return
        self._updating = True
        try:
            self.layer.set_colormap(name)
            self.layer.update_display()
            self.sigChanged.emit()
        finally:
            self._updating = False

    def _on_levels_changed(self):
        """Handle min or max spin box change."""
        if self._updating:
            return
        self._updating = True
        try:
            self.layer.set_levels(self.min_spin.value(), self.max_spin.value())
            self.layer.update_display()
            self.sigChanged.emit()
        finally:
            self._updating = False

    def _on_opacity_changed(self, value):
        """Handle opacity slider change."""
        if self._updating:
            return
        self._updating = True
        try:
            self.opacity_label.setText(f'{value}%')
            self.layer.set_opacity(value / 100.0)
            self.layer.update_display()
            self.sigChanged.emit()
        finally:
            self._updating = False

    def _on_visible_changed(self, checked):
        """Handle visible checkbox toggle."""
        if self._updating:
            return
        self._updating = True
        try:
            self.layer.set_visible(checked)
            self.sigChanged.emit()
        finally:
            self._updating = False

    def _on_remove_clicked(self):
        """Handle remove button click."""
        self.sigRemoveRequested.emit(self)


# ---------------------------------------------------------------------------
# ChannelPanel
# ---------------------------------------------------------------------------

class ChannelPanel(DockSingleton):
    """Dockable panel that provides the UI for the :class:`ChannelCompositor`.

    This is implemented as a singleton so that only one panel exists at a time.
    Use :meth:`instance` to obtain the panel.

    Layout (top to bottom):

    1. **Add Channel** — a button and :class:`WindowSelector` for picking a
       source window to add as a new channel layer.
    2. **Channel rows** — a scrollable list of :class:`ChannelRowWidget`
       instances, one per layer.
    3. **Export / Close** — buttons to export the current composite as an RGB
       window or to tear down the compositor.
    """

    # ----- construction -----

    def __init__(self, parent=None):
        super().__init__('Channel Compositor', parent)
        self.setObjectName('ChannelPanel')
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self._compositor = None
        self._rows = []  # list[ChannelRowWidget]

        self._build_ui()
        logger.debug('ChannelPanel created')

    def _build_ui(self):
        """Construct the panel layout."""
        container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # --- top: Add Channel ---
        top_layout = QtWidgets.QHBoxLayout()
        self._add_btn = QtWidgets.QPushButton('Add Channel')
        self._add_btn.setToolTip(
            'Add the selected window as a new channel in the composite'
        )
        top_layout.addWidget(self._add_btn)

        # WindowSelector is imported inside the method to avoid circular
        # imports at module level.
        from flika.utils.BaseProcess import WindowSelector
        self._window_selector = WindowSelector()
        top_layout.addWidget(self._window_selector)
        main_layout.addLayout(top_layout)

        # --- middle: scrollable channel rows ---
        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setMinimumHeight(100)

        self._scroll_widget = QtWidgets.QWidget()
        self._scroll_layout = QtWidgets.QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(2)
        self._scroll_layout.addStretch()  # keep rows packed to the top

        self._scroll_area.setWidget(self._scroll_widget)
        main_layout.addWidget(self._scroll_area, stretch=1)

        # --- bottom: Export / Close ---
        bottom_layout = QtWidgets.QHBoxLayout()

        self._export_btn = QtWidgets.QPushButton('Export Composite')
        self._export_btn.setToolTip(
            'Create a new window with the current composite as an RGB image'
        )
        bottom_layout.addWidget(self._export_btn)

        self._close_btn = QtWidgets.QPushButton('Close Compositor')
        self._close_btn.setToolTip('Remove all layers and close this panel')
        bottom_layout.addWidget(self._close_btn)

        main_layout.addLayout(bottom_layout)

        container.setLayout(main_layout)
        self.setWidget(container)

        # --- connect top-level buttons ---
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._export_btn.clicked.connect(self._export_composite)
        self._close_btn.clicked.connect(self._close_compositor)

    # ----- public API -----

    def set_compositor(self, compositor):
        """Bind this panel to a :class:`ChannelCompositor`.

        If there is already a compositor bound, it is cleaned up first.

        Parameters
        ----------
        compositor : ChannelCompositor
            The compositor whose layers this panel will control.
        """
        if self._compositor is not None:
            self.cleanup()

        self._compositor = compositor

        # Populate rows for any layers already present.
        for layer in compositor.layers:
            self._add_channel_row(layer)

        logger.debug(
            'ChannelPanel bound to compositor with %d existing layers',
            len(compositor.layers),
        )

    # ----- channel row management -----

    def _add_channel_row(self, layer):
        """Create a :class:`ChannelRowWidget` for *layer* and add it to the
        scroll area.

        Parameters
        ----------
        layer : ChannelLayer
            The layer to create a control row for.

        Returns
        -------
        ChannelRowWidget
            The newly created row widget.
        """
        row = ChannelRowWidget(layer, parent=self._scroll_widget)
        row.sigChanged.connect(self._on_row_changed)
        row.sigRemoveRequested.connect(self._remove_channel_row)

        # Insert before the stretch item at the end of the scroll layout.
        insert_index = self._scroll_layout.count() - 1  # before the stretch
        if insert_index < 0:
            insert_index = 0
        self._scroll_layout.insertWidget(insert_index, row)

        self._rows.append(row)
        logger.debug('Added channel row for layer %r', layer)
        return row

    def _remove_channel_row(self, row_widget):
        """Remove a channel row widget and its associated layer.

        Parameters
        ----------
        row_widget : ChannelRowWidget
            The row to remove.
        """
        if row_widget not in self._rows:
            return

        layer = row_widget.layer

        # Remove the row from the layout and our tracking list.
        self._scroll_layout.removeWidget(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()
        self._rows.remove(row_widget)

        # Remove the layer from the compositor.
        if self._compositor is not None:
            self._compositor.remove_channel(layer)

        logger.debug('Removed channel row for layer %r', layer)

    def _on_row_changed(self):
        """Slot called when any row's controls change.

        Triggers a full composite update so the viewport reflects the change.
        """
        if self._compositor is not None:
            self._compositor.update_composite()

    # ----- button handlers -----

    def _on_add_clicked(self):
        """Handle the 'Add Channel' button click.

        Reads the currently selected window from the :class:`WindowSelector`
        and adds it as a new channel in the compositor.
        """
        import flika.global_vars as g

        if self._compositor is None:
            logger.warning('No compositor set; cannot add channel')
            return

        window = self._window_selector.value()
        if window is None:
            # Fall back to the current window if nothing is selected.
            window = g.win
        if window is None:
            logger.warning('No window selected; cannot add channel')
            return

        # Avoid adding the same source twice.
        if self._compositor.get_layer_by_source(window) is not None:
            logger.info(
                'Window %r is already a channel in the compositor',
                getattr(window, 'name', '?'),
            )
            return

        layer = self._compositor.add_channel(window)
        self._add_channel_row(layer)

    def _export_composite(self):
        """Export the current composite as a new RGB Window.

        Creates a new :class:`~flika.window.Window` containing the composited
        (H, W, 3) uint8 array.
        """
        from flika.window import Window
        import flika.global_vars as g

        if self._compositor is None:
            logger.warning('No compositor set; cannot export')
            return

        try:
            rgb = self._compositor.export_composite_rgb()
        except Exception as exc:
            logger.error('Failed to export composite: %s', exc)
            if g.m is not None:
                g.alert('Export failed: {}'.format(exc))
            return

        host_name = getattr(self._compositor.host_window, 'name', 'Composite')
        w = Window(
            rgb,
            name=host_name + ' - Composite',
            metadata={'is_rgb': True},
        )
        logger.info(
            'Exported composite RGB (%dx%d) to window %r',
            rgb.shape[1], rgb.shape[0], w.name,
        )

    def _close_compositor(self):
        """Handle the 'Close Compositor' button click.

        Cleans up the compositor and hides this panel.
        """
        self.cleanup()
        self.close()

    # ----- cleanup -----

    def cleanup(self):
        """Remove all channel rows and release the compositor reference.

        This does **not** close the panel itself; call :meth:`close` or
        :meth:`_close_compositor` for that.
        """
        # Remove all rows (iterate over a copy since list is mutated).
        for row in list(self._rows):
            self._scroll_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        # Clean up the compositor.
        if self._compositor is not None:
            try:
                self._compositor.cleanup()
            except Exception as exc:
                logger.error('Error cleaning up compositor: %s', exc)
            self._compositor = None

        logger.debug('ChannelPanel cleaned up')

    # closeEvent is handled by DockSingleton (calls cleanup + clears _instance)


logger.debug("Completed 'reading viewers/channel_panel.py'")
