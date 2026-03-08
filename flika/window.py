# -*- coding: utf-8 -*-
from .logger import logger
logger.debug("Started 'reading window.py'")
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import os, time
import numpy as np
from . import global_vars as g
from .roi import *
from .utils.misc import save_file_gui
from .utils.BaseProcess import WindowSelector, SliderLabel

pg.setConfigOptions()

# ----- pyqtgraph bug-fix: GradientEditorItem.currentColorRejected -----
# In pyqtgraph 0.14 the `currentColorRejected` slot does not guard against
# `self.currentTick` being None (the sibling `currentColorChanged` does).
# This causes an AttributeError when the colour dialog is dismissed without
# a tick being selected.  Monkey-patch it to add the missing check.
_GEI = pg.graphicsItems.GradientEditorItem.GradientEditorItem
_orig_currentColorRejected = _GEI.currentColorRejected

def _safe_currentColorRejected(self):
    if self.currentTick is not None:
        _orig_currentColorRejected(self)

_GEI.currentColorRejected = _safe_currentColorRejected
# -----------------------------------------------------------------------

# ----- pyqtgraph bug-fix: ViewBoxMenu.setViewList deleted QComboBox -----
# When a ViewBox is destroyed its menu may already be garbage-collected,
# causing a RuntimeError on QComboBox access.  Guard with a try/except.
_VBM = pg.graphicsItems.ViewBox.ViewBoxMenu.ViewBoxMenu
_orig_setViewList = _VBM.setViewList

def _safe_setViewList(self, views):
    try:
        _orig_setViewList(self, views)
    except RuntimeError:
        pass  # widget already deleted during shutdown

_VBM.setViewList = _safe_setViewList
# -----------------------------------------------------------------------


class Bg_im_dialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.parent = parent
        self.setWindowTitle("Select background image")
        self.window_selector = WindowSelector()
        self.window_selector.valueChanged.connect(self.bg_win_changed)
        self.alpha_slider = SliderLabel(3)
        self.alpha_slider.setRange(0,1)
        self.alpha_slider.setValue(.5)
        self.alpha_slider.valueChanged.connect(self.alpha_changed)
        self.formlayout = QtWidgets.QFormLayout()
        self.formlayout.setLabelAlignment(QtCore.Qt.AlignRight)
        self.formlayout.addRow("Select window with background image", self.window_selector)
        self.formlayout.addRow("Set background opacity", self.alpha_slider)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.formlayout)
        self.setLayout(self.layout)

    def alpha_changed(self, value):
        if self.parent.bg_im is not None:
            self.parent.bg_im.setOpacity(value)

    def bg_win_changed(self):
        if self.parent.bg_im is not None:
            self.parent.imageview.view.removeItem(self.parent.bg_im)
            self.bg_im = None
        self.parent.bg_im = pg.ImageItem(self.window_selector.window.imageview.imageItem.image)
        self.parent.bg_im.setOpacity(self.alpha_slider.value())
        self.parent.imageview.view.addItem(self.parent.bg_im)


    def closeEvent(self,ev):
        if self.parent.bg_im is not None:
            self.parent.imageview.view.removeItem(self.parent.bg_im)
            self.bg_im = None


class ImageView(pg.ImageView):
    def __init__(self, *args, **kargs):
        pg.ImageView.__init__(self, *args, **kargs)
        self.view.unregister()
        self.view.removeItem(self.roi)
        self.view.removeItem(self.normRoi)
        self.roi.setParent(None)
        self.normRoi.setParent(None)
        self.ui.menuBtn.setParent(None)
        self.ui.roiBtn.setParent(None) # gets rid of 'roi' button that comes with ImageView
        self.ui.normLUTbtn = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.normLUTbtn.setObjectName("LUT norm")
        self.ui.normLUTbtn.setText("LUT norm")
        self.ui.gridLayout.addWidget(self.ui.normLUTbtn, 1, 1, 1, 1)

        self.ui.bg_imbtn = QtWidgets.QPushButton(self.ui.layoutWidget)
        self.ui.bg_imbtn.setObjectName("bg im")
        self.ui.bg_imbtn.setText("bg im")
        self.ui.gridLayout.addWidget(self.ui.bg_imbtn, 1, 2, 1, 1)

        self.ui.roiPlot.setMaximumHeight(40)
        self.ui.roiPlot.getPlotItem().getViewBox().setMouseEnabled(False)
        self.ui.roiPlot.getPlotItem().hideButtons()

    def hasTimeAxis(self):
        return 't' in self.axes and not (self.axes['t'] is None or self.image.shape[self.axes['t']] == 1)

    def roiClicked(self):
        showRoiPlot = False
        if self.hasTimeAxis():
            showRoiPlot = True
            mn = self.tVals.min()
            mx = self.tVals.max() + .01
            self.ui.roiPlot.setXRange(mn, mx, padding=0.01)
            self.timeLine.show()
            self.timeLine.setBounds([mn, mx])
            self.ui.roiPlot.show()
            if not self.ui.roiBtn.isChecked():
                self.ui.splitter.setSizes([self.height()-35, 35])
        else:
            self.timeLine.hide()
            #self.ui.roiPlot.hide()
            
        self.ui.roiPlot.setVisible(showRoiPlot)


class OrthogonalViewer(QtWidgets.QWidget):
    """Side panels showing XZ and YZ slices through a 3-D stack.

    Toggled via *View > Orthogonal Views* in the parent :class:`Window`.
    When the parent has a 4D ``volume``, slices are taken through the Z
    dimension giving true XZ and YZ cross-sections.

    Hold **C** in the main window to position the slice crosshair.
    """

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle(f'Ortho: {parent_window.name}')
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel("Hold C in the image window to position the slice crosshair")
        hint.setStyleSheet("color: #888; font-style: italic; padding: 2px;")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(hint)

        views_layout = QtWidgets.QHBoxLayout()
        self.xz_item = pg.ImageItem()
        self.yz_item = pg.ImageItem()
        xz_view = pg.GraphicsLayoutWidget()
        yz_view = pg.GraphicsLayoutWidget()
        xz_plot = xz_view.addPlot(title='XZ')
        yz_plot = yz_view.addPlot(title='YZ')
        xz_plot.addItem(self.xz_item)
        yz_plot.addItem(self.yz_item)
        xz_plot.setAspectLocked(True)
        yz_plot.setAspectLocked(True)

        views_layout.addWidget(xz_view)
        views_layout.addWidget(yz_view)
        layout.addLayout(views_layout)
        self.resize(600, 320)

        # Initial crosshair position: centre
        self._y_pos = parent_window.my // 2
        self._x_pos = parent_window.mx // 2

        # Connect signals for live updates
        parent_window.sigTimeChanged.connect(lambda _: self.update_slices())
        # Connect dimension sliders (Z slider, etc.)
        for dim_idx, (slider, _label) in parent_window._dim_sliders.items():
            slider.valueChanged.connect(lambda _: self.update_slices())

        self.update_slices()

    def set_crosshair(self, x, y):
        """Update the slice position from the main view crosshair."""
        self._x_pos = int(np.clip(x, 0, self.parent_window.mx - 1))
        self._y_pos = int(np.clip(y, 0, self.parent_window.my - 1))
        self.update_slices()

    def update_slices(self):
        pw = self.parent_window
        if pw.metadata.get('is_rgb', False):
            return

        vol = pw.volume
        if vol is not None and vol.ndim == 4:
            # 4D volume: (T, X, Y, Z) — show true cross-sections through Z
            t = pw.currentIndex
            t = min(t, vol.shape[0] - 1)
            y = min(self._y_pos, vol.shape[2] - 1)
            x = min(self._x_pos, vol.shape[1] - 1)
            xz_slice = vol[t, :, y, :]   # shape (X, Z)
            yz_slice = vol[t, x, :, :]   # shape (Y, Z)
            self.xz_item.setImage(xz_slice.T)
            self.yz_item.setImage(yz_slice.T)
        else:
            # 3D fallback: (T, X, Y) — slice through time
            img = pw.image
            if img is None or img.ndim < 3:
                return
            y = min(self._y_pos, img.shape[2] - 1)
            x = min(self._x_pos, img.shape[1] - 1)
            xz_slice = img[:, :, y]   # shape (T, X)
            yz_slice = img[:, x, :]   # shape (T, Y)
            self.xz_item.setImage(xz_slice.T)
            self.yz_item.setImage(yz_slice.T)

    def closeEvent(self, event):
        self.parent_window._ortho_viewer = None
        event.accept()


class Window(QtWidgets.QWidget):
    """
    Window objects are the central objects in flika. Almost all functions in the 
    :mod:`process <flika.process>` module are performed on Window objects and 
    output Window objects. 

    Args:
        tif (numpy.array): The image the window will store and display
        name (str): The name of the window.
        filename (str): The filename (including full path) of file this window's image orinated from.
        commands (list of str): a list of the commands used to create this window, starting with loading the file.
        metadata (dict): dict: a dictionary containing the original file's metadata.


    """
    closeSignal = QtCore.Signal()
    keyPressSignal = QtCore.Signal(QtCore.QEvent)
    sigTimeChanged = QtCore.Signal(int)
    sigSliceChanged = QtCore.Signal()  # emitted when a Z/C/D4 slider changes the displayed image
    sigROICreated = QtCore.Signal(object)
    sigROIRemoved = QtCore.Signal(object)
    gainedFocusSignal = QtCore.Signal()
    lostFocusSignal = QtCore.Signal()

    def __init__(self, tif, name='flika', filename='', commands=None, metadata=None):
        from .process.measure import measure
        QtWidgets.QWidget.__init__(self)
        if commands is None:
            commands = []
        if metadata is None:
            metadata = {}
        self.name = name  #: str: The name of the window.
        self.filename = filename  #: str: The filename (including full path) of file this window's image orinated from.
        self.commands = commands  #: list of str: a list of the commands used to create this window, starting with loading the file.
        self.metadata = metadata  #: dict: a dictionary containing the original file's metadata.
        self.volume = None  # When attaching a 4D array to this Window object, where self.image is a 3D slice of this volume, attach it here. This will remain None for all 3D Windows
        self.scatterPlot = None
        self.closed = False  #: bool: True if the window has been closed, False otherwise.
        self.mx = 0  #: int: The number of pixels wide the image is in the x (left to right) dimension.
        self.my = 0  #: int: The number of pixels heigh the image is in the y (up to down) dimension.
        self.mt = 0  #: int: The number of frames in the image stack.
        self.framerate = None  #: float: The number of frames per second (Hz).
        # Detect lazy arrays and materialize before use
        from .io.lazy import is_lazy
        if is_lazy(tif):
            self._lazy_source = tif
            g.status_msg(f'Materializing {name} ({tif.shape})...')
            tif = tif.materialize()
            g.status_msg(f'{name} loaded.')
        else:
            self._lazy_source = None
        self.image = tif
        self.dtype = tif.dtype  #: dtype: The datatype of the stored image, e.g. ``uint8``.
        self.top_left_label = None
        self.rois = []  #: list of ROIs: a list of all the :class:`ROIs <flika.roi.ROI_Base>` inside this window.
        self.currentROI = None  #: :class:`ROI <flika.roi.ROI_Base>`: When an ROI is clicked, it becomes the currentROI of that window and can be accessed via this variable.
        self.creatingROI = False
        self.imageview = None
        self.bg_im = None
        self.currentIndex = 0
        self.linkedWindows = set()
        self.measure = measure
        self.resizeEvent = self.onResize
        self.moveEvent = self.onMove
        self.pasteAct = QtWidgets.QAction("&Paste", self, triggered=self.paste)
        self.sigTimeChanged.connect(self.showFrame)
        self._check_for_infinities(tif)
        self._dim_sliders = {}
        self._ortho_viewer = None
        self._volume_viewer = None
        self._crosshair_active = False
        self._crosshair_x_line = None
        self._crosshair_y_line = None
        self._compositor = None  # ChannelCompositor, set by process/compositing.py
        # Detect RGB before dimension sliders: 4D with last dim <= 4 is RGB
        if 'is_rgb' not in self.metadata:
            self.metadata['is_rgb'] = (tif.ndim == 4 and tif.shape[-1] <= 4)
        self._init_dimension_sliders(tif)
        # For 4D+ non-RGB data, display only the initial 3D slice
        display_tif = tif
        if self.volume is not None:
            idx = [slice(None)] * 3 + [0] * (tif.ndim - 3)
            display_tif = tif[tuple(idx)]
        self._init_dimensions(display_tif)
        self._init_imageview(display_tif)
        self.setWindowTitle(name)
        self.normLUT(tif)
        self._init_scatterplot()
        self._init_menu()
        self._init_geometry()
        self.setAsCurrentWindow()

    def _init_geometry(self):
        assert g.win != self  # self.setAsCurrentWindow() must be called after this function
        if g.win is None:
            if 'window_settings' not in g.settings:
                g.settings['window_settings'] = dict()
            if 'coords' in g.settings['window_settings']:
                geometry = QtCore.QRect(*g.settings['window_settings']['coords'])
            else:
                width = 684
                height = 585
                nwindows = len(g.windows)
                x = 10 + 10 * nwindows
                y = 300 + 10 * nwindows
                geometry = QtCore.QRect(x, y, width, height)
                g.settings['window_settings']['coords'] = geometry.getRect()
        else:
            geometry = g.win.geometry()

        desktopGeom = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        maxX = (desktopGeom.width() - geometry.width()) or 1
        maxY = (desktopGeom.height() - geometry.height()) or 1
        newX = (geometry.x() + 10) % maxX
        newY = ((geometry.y() + 10) % maxY) or 30
        
        geometry = QtCore.QRect(newX, newY, geometry.width(), geometry.height())
        self.setGeometry(geometry)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.imageview)
        for dim_idx in sorted(self._dim_sliders.keys()):
            slider, label = self._dim_sliders[dim_idx]
            row = QtWidgets.QHBoxLayout()
            row.addWidget(label)
            row.addWidget(slider)
            self.layout.addLayout(row)
        self.layout.setContentsMargins(0, 0, 0, 0)
        if g.settings['show_windows']:
            self.show()
            self.raise_()
            QtWidgets.QApplication.processEvents()

    def _init_dimension_sliders(self, tif):
        """Add QSliders for extra dimensions beyond 3D (non-RGB).

        For a 4D array that is *not* RGB, we treat dim-0 as time and add sliders
        for each additional dimension (e.g. Z, C).  The displayed 3-D sub-stack
        is ``self.volume[t, :, :, slider_index]`` (or similar).
        """
        if tif.ndim <= 3 or self.metadata.get('is_rgb', False):
            return
        # Store the full volume
        self.volume = tif
        dim_names = ['Z', 'C', 'D4', 'D5']  # fallback labels
        # Extra dims are everything after (T, X, Y)
        for i in range(3, tif.ndim):
            dim_label = dim_names[i - 3] if (i - 3) < len(dim_names) else f'D{i}'
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(tif.shape[i] - 1)
            slider.setValue(0)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            label = QtWidgets.QLabel(f'{dim_label}: 0')
            slider.valueChanged.connect(lambda val, lbl=label, dl=dim_label: lbl.setText(f'{dl}: {val}'))
            slider.valueChanged.connect(self._update_displayed_slice)
            self._dim_sliders[i] = (slider, label)

    def _update_displayed_slice(self):
        """Re-slice ``self.volume`` using the current slider positions and update
        the displayed image."""
        if self.volume is None:
            return
        idx = [slice(None)] * 3  # T, X, Y
        for dim_idx in sorted(self._dim_sliders.keys()):
            slider, _ = self._dim_sliders[dim_idx]
            idx.append(slider.value())
        sub = self.volume[tuple(idx)]
        self.image = sub
        self._init_dimensions(sub)
        self.imageview.setImage(sub, autoLevels=False)
        self.sigSliceChanged.emit()

    def _check_for_infinities(self, tif):
        try:
            if np.any(np.isinf(tif)):
                tif[np.isinf(tif)] = 0
                g.alert('Some array values were inf. Setting those values to 0')
        except MemoryError:
            pass

    def _init_imageview(self, tif):
        self.imageview = ImageView(self)
        self.imageview.setMouseTracking(True)
        self.imageview.installEventFilter(self)
        self.imageview.ui.normLUTbtn.pressed.connect(lambda: self.normLUT(self.image))
        self.imageview.ui.bg_imbtn.pressed.connect(self.set_bg_im)
        rp = self.imageview.ui.roiPlot.getPlotItem()
        self.linkMenu = QtWidgets.QMenu("Link frame")
        rp.ctrlMenu = self.linkMenu
        self.linkMenu.aboutToShow.connect(self.make_link_menu)
        self.imageview.setImage(tif)
        def clicked(evt):
            self.measure.pointclicked(evt, window=self)
        self.imageview.scene.sigMouseClicked.connect(clicked)
        self.imageview.timeLine.sigPositionChanged.connect(self.updateindex)
        self.currentIndex = self.imageview.currentIndex
        self.imageview.scene.sigMouseMoved.connect(self.mouseMoved)
        self.imageview.view.mouseDragEvent = self.mouseDragEvent
        self.imageview.view.mouseClickEvent = self.mouseClickEvent
        assert self.top_left_label is not None
        self.imageview.ui.graphicsView.addItem(self.top_left_label)

    def _init_dimensions(self, tif):
        if 'is_rgb' not in self.metadata:
            self.metadata['is_rgb'] = (tif.ndim == 4 and tif.shape[-1] <= 4)
        self.nDims = len(np.shape(tif))  #: int: The number of dimensions of the stored image.
        dimensions_txt = ""
        if self.nDims == 3:
            if self.metadata['is_rgb']:
                self.mx, self.my, mc = tif.shape
                self.mt = 1
                dimensions_txt = f"{self.mx}x{self.my} pixels; {mc} colors; "
            else:
                self.mt, self.mx, self.my = tif.shape
                dimensions_txt = f"{self.mt} frames; {self.mx}x{self.my} pixels; "
        elif self.nDims == 4:
            self.mt, self.mx ,self.my, mc = tif.shape
            dimensions_txt = f"{self.mt} frames; {self.mx}x{self.my} pixels; {mc} colors; "
        elif self.nDims == 2:
            self.mt = 1
            self.mx, self.my = tif.shape
            dimensions_txt = f"{self.mx}x{self.my} pixels; "
        dimensions_txt += 'dtype=' + str(self.dtype)
        if self.framerate is None:
            if 'timestamps' in self.metadata:
                ts = self.metadata['timestamps']
                self.framerate = (ts[-1] - ts[0]) / len(ts)
        if self.framerate is not None:
            tu = self.metadata['timestamp_units']
            dimensions_txt += f'; {self.framerate:.4f} {tu}/frame'

        if self.top_left_label is not None and self.imageview is not None and self.top_left_label in self.imageview.ui.graphicsView.items():
            self.imageview.ui.graphicsView.removeItem(self.top_left_label)
        self.top_left_label = pg.LabelItem(dimensions_txt, justify='right')

    def _init_scatterplot(self):
        if self.scatterPlot in self.imageview.ui.graphicsView.items():
            self.imageview.ui.graphicsView.removeItem(self.scatterPlot)
        pointSize = g.settings['point_size']
        pointColor = QtGui.QColor(g.settings['point_color'])
        self.scatterPlot = pg.ScatterPlotItem(size=pointSize, pen=pg.mkPen([0, 0, 0, 255]), brush=pg.mkBrush(*pointColor.getRgb()))
        self.scatterPoints = [[] for _ in np.arange(self.mt)]
        self._brushCache = {}
        self.scatterPlot.sigClicked.connect(self.clickedScatter)
        self.imageview.addItem(self.scatterPlot)

    def _getCachedBrush(self, color):
        """Return a cached pg.mkBrush for the given QColor to avoid recreating brushes."""
        rgba = color.getRgb()
        brush = self._brushCache.get(rgba)
        if brush is None:
            brush = pg.mkBrush(*rgba)
            self._brushCache[rgba] = brush
        return brush

    def _init_menu(self):
        self.menu = QtWidgets.QMenu(self)

        pasteAct = QtWidgets.QAction("&Paste", self, triggered=self.paste)
        plotAllAct = QtWidgets.QAction('&Plot All ROIs', self.menu, triggered=self.plotAllROIs)
        copyAll = QtWidgets.QAction("Copy All ROIs", self.menu, triggered=lambda a: setattr(g, 'clipboard', self.rois))
        removeAll = QtWidgets.QAction("Remove All ROIs", self.menu, triggered=self.removeAllROIs)
        orthoAct = QtWidgets.QAction("Orthogonal Views", self.menu, triggered=self.toggleOrthogonalViews)
        volumeAct = QtWidgets.QAction("3D Volume Viewer", self.menu, triggered=self.toggleVolumeViewer)
        roiMgrAct = QtWidgets.QAction("ROI Manager", self.menu, triggered=self._open_roi_manager)

        self.menu.addAction(pasteAct)
        self.menu.addAction(plotAllAct)
        self.menu.addAction(copyAll)
        self.menu.addAction(removeAll)
        self.menu.addSeparator()
        self.menu.addAction(orthoAct)
        self.menu.addAction(volumeAct)
        snapshotAct = QtWidgets.QAction("Snapshot to Desktop", self.menu, triggered=self.snapshot)
        self.menu.addSeparator()
        self.menu.addAction(roiMgrAct)
        self.menu.addAction(snapshotAct)

        def updateMenu():
            from .roi import ROI_Base
            pasteAct.setEnabled(isinstance(g.clipboard, (list, ROI_Base)))
            ndims = self.nDims
            is_rgb = self.metadata.get('is_rgb', False)
            has_time = ndims >= 3 and not is_rgb
            has_volume = self.volume is not None
            # Plot All ROIs only makes sense with time or volume
            plotAllAct.setVisible(has_time or has_volume)
            # Orthogonal Views for 3D+ non-RGB
            orthoAct.setVisible(ndims >= 3 and not is_rgb)
            # Volume Viewer only for 4D
            volumeAct.setVisible(has_volume)

        self.menu.aboutToShow.connect(updateMenu)

    def snapshot(self):
        """Save a screenshot of the image view (including ROIs and overlays) to the Desktop."""
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        base = self.name.replace(' ', '_').replace('/', '_')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(desktop, f'{base}_{timestamp}.png')
        # Grab the graphics view which contains the image, ROIs, and all overlays
        pixmap = self.imageview.ui.graphicsView.grab()
        pixmap.save(filepath)
        g.status_msg(f'Snapshot saved to {filepath}')

    def _open_roi_manager(self):
        """Open the ROI Manager panel from the main application."""
        if g.m is not None:
            g.m._toggle_roi_manager()

    def toggleOrthogonalViews(self):
        """Toggle the XZ/YZ orthogonal-view panel."""
        if self._ortho_viewer is not None:
            self._ortho_viewer.close()
            self._ortho_viewer = None
        else:
            if self.image.ndim >= 3 and not self.metadata.get('is_rgb', False):
                self._ortho_viewer = OrthogonalViewer(self)
                self._ortho_viewer.show()

    def toggleVolumeViewer(self):
        """Toggle the 3D volume viewer (only for 4D windows)."""
        if self._volume_viewer is not None:
            self._volume_viewer.close()
            self._volume_viewer = None
        else:
            if self.volume is not None:
                from .viewers.volume_viewer import VolumeViewer
                try:
                    self._volume_viewer = VolumeViewer(self)
                    self._volume_viewer.show()
                except ImportError as e:
                    g.alert(str(e))

    def onResize(self, event):
        g.settings['window_settings']['coords'] = self.geometry().getRect()

    def onMove(self, event):
        g.settings['window_settings']['coords'] = self.geometry().getRect()

    def save(self, filename):
        """save(self, filename)
        Saves the current window to a specificed directory as a (.tif) file

        Args:
            filename (str): The filename, including the full path, where this (.tif) file will be saved.

        """
        from .process.file_ import save_file
        old_curr_win = g.win
        self.setAsCurrentWindow()
        save_file(filename)
        old_curr_win.setAsCurrentWindow()

    def normLUT(self, tif):
        def _padded_levels(lo, hi):
            """Return (lo, hi) with a small margin; cast to float to avoid
            NumPy 2.0 bool-subtraction errors."""
            lo, hi = float(lo), float(hi)
            margin = (hi - lo) / 100 if hi != lo else 0.01
            return lo - margin, hi + margin

        if self.nDims == 2:
            if np.min(tif) == 0 and (np.max(tif) == 0 or np.max(tif) == 1):
                self.imageview.setLevels(-.01, 1.01)
            else:
                self.imageview.setLevels(*_padded_levels(np.min(tif), np.max(tif)))
        if self.nDims == 3 and not self.metadata['is_rgb']:
            if np.all(tif[self.currentIndex] == 0):
                self.imageview.setLevels(*_padded_levels(np.min(tif), np.max(tif)))
            else:
                self.imageview.setLevels(*_padded_levels(
                    np.min(tif[self.currentIndex]), np.max(tif[self.currentIndex])))
        elif self.nDims == 4 and not self.metadata['is_rgb']:
            if np.min(tif) == 0 and (np.max(tif) == 0 or np.max(tif) == 1):
                self.imageview.setLevels(-.01, 1.01)

    def set_bg_im(self):
        self.bg_im_dialog = Bg_im_dialog(self)
        self.bg_im_dialog.show()

    def link(self, win):
        """link(self, win)
        Linking a window to another means when the current index of one changes, the index of the other will automatically change.

        Args:
            win (flika.window.Window): The window that will be linked with this one
        """
        if win not in self.linkedWindows:
            self.sigTimeChanged.connect(win.imageview.setCurrentIndex)
            self.linkedWindows.add(win)
            win.link(self)

    def unlink(self, win):
        """unlink(self, win)
        This unlinks a window from this one.

        Args:
            win (flika.window.Window): The window that will be unlinked from this one
        """
        if win in self.linkedWindows:
            self.linkedWindows.remove(win)
            self.sigTimeChanged.disconnect(win.imageview.setCurrentIndex)
            win.unlink(self)

    def link_toggled(self, win):
        return lambda b: self.link(win) if b else self.unlink(win)

    def make_link_menu(self):
        self.linkMenu.clear()
        for win in g.windows:
            if win == self or not win.isVisible():
                continue
            win_action = QtWidgets.QAction("%s" % win.name, self.linkMenu, checkable=True)
            win_action.setChecked(win in self.linkedWindows)
            win_action.toggled.connect(self.link_toggled(win))
            self.linkMenu.addAction(win_action)
        
    def updateindex(self):
        if self.mt == 1:
            t = 0
        else:
            (idx, t) = self.imageview.timeIndex(self.imageview.timeLine)
            t = int(np.floor(t))
        if 0 <= t < self.mt:
            self.currentIndex = t
            if not g.settings['show_all_points']:
                pointSizes = [pt[3] for pt in self.scatterPoints[t]]
                brushes = [self._getCachedBrush(pt[2]) for pt in self.scatterPoints[t]]
                self.scatterPlot.setData(pos=self.scatterPoints[t], size=pointSizes, brush=brushes)
            self.sigTimeChanged.emit(t)

    def setIndex(self, index):
        """setIndex(self, index)
        This sets the index (frame) of this window. 

        Args:
            index (int): The index of the image this window will display
        """
        if hasattr(self, 'image') and self.image.ndim > 2 and 0 <= index < len(self.image):
            self.imageview.setCurrentIndex(index)

    def showFrame(self, index):
        if index>=0 and index<self.mt:
            msg = 'frame {}'.format(index)
            if 'timestamps' in self.metadata and self.metadata.get('timestamp_units')=='ms':
                ttime = self.metadata['timestamps'][index]
                if ttime < 1*1000:
                    msg += '; {:.4f} ms'.format(ttime)
                elif ttime < 60*1000:
                    seconds = ttime / 1000
                    msg += '; {:.4f} s'.format(seconds)
                elif ttime < 3600*1000:
                    minutes = int(np.floor(ttime / (60*1000)))
                    seconds = (ttime/1000) % 60
                    msg += '; {} m {:.4f} s'.format(minutes, seconds)
                else:
                    seconds = ttime/1000
                    hours = int(np.floor(seconds / 3600))
                    mminutes = seconds - hours * 3600
                    minutes = int(np.floor(mminutes / 60))
                    seconds = mminutes - minutes * 60
                    msg += '; {} h {} m {:.4f} s'.format(hours, minutes, seconds)
            g.status_msg(msg)

    def setName(self,name):
        """setName(self,name)
        Set the name of this window.

        Args:
            name (str): the name for window to be set to
        """
        name = str(name)
        self.name = name
        self.setWindowTitle(name)
        
    def reset(self):
        if not self.closed:
            currentIndex = int(self.currentIndex)
            self.imageview.setImage(self.image, autoLevels=True) #I had autoLevels=False before.  I changed it to adjust after boolean previews.
            if self.imageview.axes['t'] is not None:
                self.imageview.setCurrentIndex(currentIndex)
            g.status_msg('')

    def closeEvent(self, event):
        if self.closed:
            logger.debug('Attempt to close window {} that was already closed'.format(self))
            event.accept()
        else:
            if self._compositor is not None:
                self._compositor.cleanup()
                self._compositor = None
            self.closeSignal.emit()
            for win in list(self.linkedWindows):
                self.unlink(win)
            if hasattr(self,'image'):
                del self.image
            self.imageview.setImage(np.zeros((2,2))) #clear the memory
            self.imageview.close()
            del self.imageview
            if g.win==self:
                g.win=None
            if self in g.windows:
                g.windows.remove(self)
            self.closed=True
            event.accept() # let the window close

    def imageArray(self):
        """imageArray(self)

        Returns:
             Image as a 3d array, correcting for color or 2d image
        """
        tif = self.image
        nDims = len(tif.shape)
        if nDims == 4:  # If this is an RGB image stack  #[t, x, y, colors]
            tif = np.mean(tif,3)
            mx, my = tif[0, :, :].shape
        elif nDims == 3:
            if self.metadata['is_rgb']:  # [x, y, colors]
                tif = np.mean(tif,2)
                mx, my = tif.shape
                tif = tif[np.newaxis]
            else: 
                mx, my = tif[0,:,:].shape
        elif nDims == 2:
            mx, my = tif.shape
            tif = tif[np.newaxis]
        return tif

    def imageDimensions(self):
        nDims = self.image.shape
        if len(nDims) == 4: #if this is an RGB image stack
            return nDims[1:3]
        elif len(nDims) == 3:
            if self.metadata['is_rgb']:  # [x, y, colors]
                return nDims[:2]
            else:                               # [t, x, y]
                return nDims[1:]
        if len(nDims) == 2:  # If this is a static image
            return nDims
        return nDims

    def resizeEvent(self, event):
        event.accept()
        self.imageview.resize(self.size())

    def paste(self):
        """ paste(self)
        This function pastes a ROI from one window into another.
        The ROIs will be automatically linked using the link() fucntion so that when you alter one of them, the other will be altered in the same way.
        """
        def pasteROI(roi):
            if roi in self.rois:
                return None
            if roi.kind == 'rect_line':
                self.currentROI = makeROI(roi.kind, roi.pts, self, width=roi.width)
            else:
                self.currentROI = makeROI(roi.kind, roi.pts, self)
            if roi in roi.window.rois:
                self.currentROI.link(roi)
            return self.currentROI

        if type(g.clipboard) == list:
            rois = []
            for roi in g.clipboard:
                rois.append(pasteROI(roi))
            return rois
        else:
            return pasteROI(g.clipboard)
        
    def mousePressEvent(self,ev):
        ev.accept()
        self.setAsCurrentWindow()

    def setAsCurrentWindow(self):
        """setAsCurrentWindow(self)
        This function sets this window as the current window. There is only one current window. All operations are performed on the
        current window. The current window can be accessed from the variable ``g.win``. 
        """

        if g.win is not None:
            g.win.setStyleSheet("border:1px solid rgb(0, 0, 0); ")
            g.win.lostFocusSignal.emit()
        g.win = self
        g.m.currentWindow = g.win
        g.currentWindow = g.win
        if self not in g.windows:
            g.windows.append(self)
        g.m.setWindowTitle("flika - {}".format(self.name))
        self.setStyleSheet("border:1px solid rgb(0, 255, 0); ")
        g.m.setCurrentWindowSignal.sig.emit()
        self.gainedFocusSignal.emit()
    
    def clickedScatter(self, plot, points):
        p = points[0]
        x, y = p.pos()
        if g.settings['show_all_points']:
            pts = []
            for t in np.arange(self.mt):
                self.scatterPoints[t] = [p for p in self.scatterPoints[t] if not (x == p[0] and y == p[1])]
                pts.extend(self.scatterPoints[t])
            pointSizes = [pt[3] for pt in pts]
            brushes = [self._getCachedBrush(pt[2]) for pt in pts]
            self.scatterPlot.setData(pos=pts, size=pointSizes, brush=brushes)
        else:
            t = self.currentIndex
            self.scatterPoints[t] = [p for p in self.scatterPoints[t] if not (x == p[0] and y == p[1])]
            pointSizes = [pt[3] for pt in self.scatterPoints[t]]
            brushes = [self._getCachedBrush(pt[2]) for pt in self.scatterPoints[t]]
            self.scatterPlot.setData(pos=self.scatterPoints[t], size=pointSizes, brush=brushes)

    def getScatterPts(self):
        """getScatterPts(self)

        Returns:
            numpy array: an Nx3 array of scatter points, where N is the number of points. Col0 is frame, Col1 is x, Col2 is y. 
        """
        p_in = self.scatterPoints
        n_total = sum(len(pts) for pts in p_in)
        if n_total == 0:
            return np.empty((0, 3))
        p_out = np.empty((n_total, 3))
        idx = 0
        for t, pts in enumerate(p_in):
            for p in pts:
                p_out[idx, 0] = t
                p_out[idx, 1] = p[0]
                p_out[idx, 2] = p[1]
                idx += 1
        return p_out

    def plotAllROIs(self):
        for roi in self.rois:
            if roi.traceWindow == None:
                roi.plot()

    def removeAllROIs(self):
        for roi in self.rois[:]:
            roi.delete()

    def addPoint(self, p=None):
        if p is None:
            p = [self.currentIndex, self.x, self.y]
        elif len(p) != 3:
            raise Exception("addPoint takes a 3-tuple (t, x, y) as argument")

        t, x, y = p

        pointSize=g.m.settings['point_size']
        pointColor = QtGui.QColor(g.settings['point_color'])
        position=[x, y, pointColor, pointSize]
        self.scatterPoints[t].append(position)
        self.scatterPlot.addPoints(pos=[[x, y]], size=pointSize, brush=self._getCachedBrush(pointColor))

    def mouseClickEvent(self, ev):
        ''''mouseClickevent(self, ev)
        Event handler for when the mouse is pressed in a flika window.
        '''
        if self.x is not None and self.y is not None and ev.button() == QtCore.Qt.RightButton and not self.creatingROI:
            mm = g.settings['mousemode']
            if mm == 'point':
                self.addPoint()
            elif mm == 'point_roi':
                pts = [pg.Point(round(self.x), round(self.y))]
                self.currentROI = makeROI("point_roi", pts)
            elif mm == 'rectangle' and g.settings['default_roi_on_click']:
                pts = [pg.Point(self.x - g.settings['rect_width']/2, self.y - g.settings['rect_height']/2), pg.Point(g.settings['rect_width'], g.settings['rect_height'])]
                self.currentROI = makeROI("rectangle", pts)
            else:
                self.menu.exec_(ev.screenPos().toQPoint())
        elif self.creatingROI:
            self.currentROI.cancel()
            self.creatingROI = None

    def save_rois(self, filename=None):
        """save_rois(self, filename=None)

        Args:
            filename (str): The filename, including the full path, where the ROI file will be saved.

        """
        if not isinstance(filename, str):
            if filename is not None and os.path.isfile(filename):
                filename = os.path.splitext(g.settings['filename'])[0]
                filename = save_file_gui('Save ROI', filename, '*.txt')
            else:
                filename = save_file_gui('Save ROI', '', '*.txt')

        if filename != '' and isinstance(filename, str):
            reprs = [roi._str() for roi in self.rois]
            reprs = '\n'.join(reprs)
            open(filename, 'w').write(reprs)
        else:
            g.status_msg('No File Selected')

    
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Delete:
            i = 0
            while i < len(self.rois):
                if self.rois[i].mouseHovering:
                    self.rois[i].delete()
                else:
                    i += 1
        elif ev.key() == QtCore.Qt.Key_C and not ev.isAutoRepeat():
            if self._ortho_viewer is not None or (hasattr(self, '_volume_viewer') and self._volume_viewer is not None):
                self._crosshair_active = True
                self._show_crosshair_lines()
                # Immediately update to current mouse position
                if self.x is not None and self.y is not None:
                    self._update_crosshair_lines(self.x, self.y)
                    if self._ortho_viewer is not None:
                        self._ortho_viewer.set_crosshair(self.x, self.y)
                    if hasattr(self, '_volume_viewer') and self._volume_viewer is not None:
                        self._volume_viewer.set_crosshair(self.x, self.y)
        self.keyPressSignal.emit(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_C and not ev.isAutoRepeat():
            self._crosshair_active = False
            self._hide_crosshair_lines()

    def _show_crosshair_lines(self):
        """Add coloured crosshair lines to the image view."""
        view = self.imageview.view
        if self._crosshair_x_line is None:
            self._crosshair_x_line = pg.InfiniteLine(
                pos=self.mx // 2, angle=90, pen=pg.mkPen('c', width=1, style=QtCore.Qt.DashLine))
            self._crosshair_y_line = pg.InfiniteLine(
                pos=self.my // 2, angle=0, pen=pg.mkPen('m', width=1, style=QtCore.Qt.DashLine))
        view.addItem(self._crosshair_x_line)
        view.addItem(self._crosshair_y_line)

    def _hide_crosshair_lines(self):
        """Remove crosshair lines from the image view."""
        view = self.imageview.view
        if self._crosshair_x_line is not None:
            try:
                view.removeItem(self._crosshair_x_line)
            except Exception:
                pass
        if self._crosshair_y_line is not None:
            try:
                view.removeItem(self._crosshair_y_line)
            except Exception:
                pass

    def _update_crosshair_lines(self, x, y):
        """Move crosshair overlay lines to (x, y)."""
        if self._crosshair_x_line is not None:
            self._crosshair_x_line.setValue(x)
        if self._crosshair_y_line is not None:
            self._crosshair_y_line.setValue(y)
        
    def mouseMoved(self,point):
        '''mouseMoved(self,point)
        Event handler function for mouse movement.
        '''
        point=self.imageview.getImageItem().mapFromScene(point)
        self.point = point
        self.x = point.x()
        self.y = point.y()
        image=self.imageview.getImageItem().image
        if self.x < 0 or self.y < 0 or self.x >= image.shape[0] or self.y >= image.shape[1]:
            pass# if we are outside the image
        else:
            z=self.imageview.currentIndex
            value=image[int(self.x),int(self.y)]
            g.status_msg('x={}, y={}, z={}, value={}'.format(int(self.x),int(self.y),z,value))
            # Only forward crosshair when crosshair mode is active (hold C)
            if self._crosshair_active:
                self._update_crosshair_lines(self.x, self.y)
                if self._ortho_viewer is not None:
                    self._ortho_viewer.set_crosshair(self.x, self.y)
                if hasattr(self, '_volume_viewer') and self._volume_viewer is not None:
                    self._volume_viewer.set_crosshair(self.x, self.y)

    def mouseDragEvent(self, ev):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            pass #This is how I detect that the shift key is held down.
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            if g.settings['mousemode'] == 'pencil':
                if ev.isStart():
                    self.last_x = None
                    self.last_y = None
                else:
                    if self.x < 0 or self.y < 0 or self.x >= self.mx or self.y >= self.my:
                        self.last_x = None  # if we are outside the image
                        self.last_y = None
                    else:
                        x = int(self.x)
                        y = int(self.y)
                        if self.last_x == x and self.last_y == y:
                            pass
                        z = self.imageview.currentIndex
                        image = self.imageview.getImageItem().image
                        v = g.settings['pencil_value']
                        if self.last_x is None:
                            image[x, y] = v
                        else:
                            xs, ys = get_line(x, y, self.last_x, self.last_y).T
                            image[xs, ys] = v
                        self.imageview.imageItem.updateImage(image)
                        self.last_x = x
                        self.last_y = y
                        self.ev = ev
            else:
                difference=self.imageview.getImageItem().mapFromScene(ev.lastScenePos())-self.imageview.getImageItem().mapFromScene(ev.scenePos())
                self.imageview.view.translateBy(difference)
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            mm = g.settings['mousemode']
            if mm in ('freehand', 'line', 'rectangle', 'rect_line', 'ellipse', 'center_surround'):
                if ev.isStart():
                    self.ev = ev
                    pt = self.imageview.getImageItem().mapFromScene(ev.buttonDownScenePos())
                    self.x = pt.x()  # This sets x and y to the button down position, not the current position.
                    self.y = pt.y()
                    self.creatingROI = True
                    self.currentROI = ROI_Drawing(self, self.x, self.y, mm)
                if ev.isFinish():
                    if self.creatingROI:   
                        if ev._buttons | QtCore.Qt.RightButton != ev._buttons:
                            self.currentROI = self.currentROI.drawFinished()
                            self.creatingROI = False
                        else:
                            self.currentROI.cancel()
                            self.creatingROI = False
                else:  # If we are in the middle of the drag between starting and finishing.
                    if self.creatingROI:
                        self.currentROI.extend(self.x, self.y)

    def updateTimeStampLabel(self,frame):
        label = self.timeStampLabel
        if self.framerate == 0:
            label.setHtml("<span style='font-size: 12pt;color:white;background-color:None;'>Frame rate is 0 Hz</span>" )
            return False
        ttime = frame/self.framerate  # Time elapsed since the first frame until the current frame, in seconds.
        if ttime < 1:
            ttime = ttime * 1000
            label.setHtml("<span style='font-size: 12pt;color:white;background-color:None;'>{:.0f} ms</span>".format(ttime))
        elif ttime < 60:
            label.setHtml("<span style='font-size: 12pt;color:white;background-color:None;'>{:.3f} s</span>".format(ttime))
        elif ttime < 3600:
            minutes = int(np.floor(ttime/60))
            seconds = ttime % 60
            label.setHtml("<span style='font-size: 12pt;color:white;background-color:None;'>{}m {:.3f} s</span>".format(minutes,seconds))
        else:
            hours = int(np.floor(ttime/3600))
            mminutes = ttime-hours*3600
            minutes = int(np.floor(mminutes/60))
            seconds = mminutes-minutes*60
            label.setHtml("<span style='font-size: 12pt;color:white;background-color:None;'>{}h {}m {:.3f} s</span>".format(hours,minutes,seconds))


def get_line(x1, y1, x2, y2):
    """Bresenham's Line Algorithm
    Produces a list of tuples """
    # Setup initial conditions
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else [x, y]
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return np.array(points)

# Add array interoperability protocols to Window class
from .interop.array_protocol import add_array_protocol
add_array_protocol(Window)

logger.debug("Completed 'reading window.py'")
