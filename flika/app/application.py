from ..logger import logger, handle_exception
logger.debug("Started 'reading app/application.py'")

import sys, os, time
import ctypes
import platform
import traceback
logger.debug("Started 'reading app/application.py, importing qtpy'")
from qtpy import QtCore, QtWidgets, QtGui
logger.debug("Completed 'reading app/application.py, importing qtpy'")

from ..utils.misc import nonpartial
from ..utils.app import get_qapp
from ..app.settings_editor import SettingsEditor, rectSettings, pointSettings, pencilSettings, csSettings
from .. import global_vars as g
from .plugin_manager import PluginManager, Load_Local_Plugins_Thread
from .script_editor import ScriptEditor
from ..utils.misc import load_ui, send_error_report, Send_User_Stats_Thread
from ..images import image_path
from ..version import __version__
from ..update_flika import checkUpdates


def status_pixmap(attention=False):
    """status_pixmap(attention=False)
    A small icon to grab attention

    Args:
        attention (bool): pixmap is red if True, gray if otherwise

    Returns:
        QtGui.QPixmap: attention icon to display
    """
    color = QtCore.Qt.red if attention else QtCore.Qt.lightGray

    pm = QtGui.QPixmap(15, 15)
    p = QtGui.QPainter(pm)
    b = QtGui.QBrush(color)
    p.fillRect(-1, -1, 20, 20, b)
    return pm


class ClickableLabel(QtWidgets.QLabel):
    """A QtGui.QLabel you can click on to generate events
    """

    clicked = QtCore.Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()


class Logger(QtWidgets.QWidget):
    """A window to display error messages
    """

    def __init__(self, parent=None):
        super(Logger, self).__init__(parent)
        self._text = QtWidgets.QTextEdit()
        self._text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(nonpartial(self._clear))
        report = QtWidgets.QPushButton("Send Bug Report")
        report.clicked.connect(nonpartial(self._send_report))
        self._status = ClickableLabel()
        self._status.setToolTip("View Errors and Warnings")
        self._status.clicked.connect(self._show)
        self._status.setPixmap(status_pixmap())
        self._status.setContentsMargins(0, 0, 0, 0)

        l = QtWidgets.QVBoxLayout()
        h = QtWidgets.QHBoxLayout()
        l.setContentsMargins(2, 2, 2, 2)
        l.setSpacing(2)
        h.setContentsMargins(0, 0, 0, 0)

        l.addWidget(self._text)
        h.insertStretch(0)
        h.addWidget(report)
        h.addWidget(clear)
        l.addLayout(h)

        self.setLayout(l)

    @property
    def status_light(self):
        """status_light(self)

        Returns:
            The icon representing the status of the log
        """
        return self._status

    def write(self, message):
        """write(self, message)
        Interface for sys.excepthook
        """
        self._text.insertPlainText(message)
        self._status.setPixmap(status_pixmap(attention=True))

    def flush(self):
        """flush(self)
        Interface for sys.excepthook
        """
        pass

    def _send_report(self):
        """_send_report(self)
        Send the contents of the log as a bug report
        """
        text = self._text.document().toPlainText()
        email = QtWidgets.QInputDialog.getText(self, "Response email", "Enter your email if you would like us to contact you about this bug.")
        if isinstance(email, tuple) and len(email) == 2:
            email = email[0]
        response = send_error_report(email=email, report=text)
        if response is None or response.status_code != 200:
            g.alert("Failed to send error report. Response {}:\n{}".format((response.status_code, response._content)))
        else:
            if email != '':
                g.alert("Bug report sent. We will contact you as soon as we can.")
            else:
                g.alert("Bug report sent. Thank you!")

    def _clear(self):
        """_clear(self)
        Erase the log
        """
        self._text.setText('')
        self._status.setPixmap(status_pixmap(attention=False))
        self.close()

    def _show(self):
        """_show(self)
        Show the log
        """
        self.show()
        self.raise_()

    def keyPressEvent(self, event):
        """keyPressEvent(self, event)
        Hide window on escape key
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.hide()



class FlikaApplication(QtWidgets.QMainWindow):
    """The main window of flika, stored as g.m
    """
    def __init__(self, headless=False):
        logger.debug("Started 'creating app.application.FlikaApplication'")
        from ..process.file_ import open_file, open_file_from_gui, open_image_sequence_from_gui, open_points, save_file, save_movie_gui, save_points, save_rois
        from ..process import setup_menus
        self._headless = headless
        logger.debug("Started 'creating app.application.FlikaApplication.app'")
        self.app = get_qapp(image_path('favicon.png'), headless=headless)
        logger.debug("Completed 'creating app.application.FlikaApplication.app'")
        super(FlikaApplication, self).__init__()
        if not headless:
            self.app.setQuitOnLastWindowClosed(True)
        setup_menus()

        if not headless:
            logger.debug("Started 'loading main.ui'")
            load_ui('main.ui', self, directory=os.path.dirname(__file__))
            logger.debug("Completed 'loading main.ui'")

        g.m = self
        # These are all added for backwards compatibility for plugins
        self.windows = g.windows
        self.traceWindows = g.traceWindows
        self.dialogs = g.dialogs
        self.currentWindow = g.win
        self.currentTrace = g.currentTrace
        self.clipboard = g.clipboard

        if not headless:
            self.setWindowSize()
            if platform.system() == 'Windows':
                myappid = 'flika-org.FLIKA.' + str(__version__)
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            self.menuBar().setNativeMenuBar(False)
            self._make_menu()
            self._make_tools()

            self._log = Logger()
            def handle_exception_wrapper(exc_type, exc_value, exc_traceback):
                handle_exception(exc_type, exc_value, exc_traceback)
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                tb_str = ''.join(tb_str)+'\n'
                self._log.write(tb_str)
            sys.excepthook = handle_exception_wrapper

            g.dialogs.append(self._log)
            self._log.window().setWindowTitle("Console Log")
            self._log.resize(550, 550)

            self.statusBar().addPermanentWidget(self._log.status_light)
            self.statusBar().setContentsMargins(2, 0, 20, 2)
            self.statusBar().setSizeGripEnabled(False)
            self.setCurrentWindowSignal = SetCurrentWindowSignal(self)
            self.setAcceptDrops(True)
            self.load_local_plugins_thread = Load_Local_Plugins_Thread()
            self.load_local_plugins_thread.start()
            self.load_local_plugins_thread.plugins_done_sig.connect(self.plugins_done)
            self.load_local_plugins_thread.error_loading.connect(g.alert)
        else:
            # Minimal headless init
            self.setCurrentWindowSignal = SetCurrentWindowSignal(self)

        logger.debug("Completed 'creating app.application.FlikaApplication'")

    def plugins_done(self, plugins):
        from .plugin_manager import is_plugin_disabled
        for p in plugins.values():
            if p.loaded and not is_plugin_disabled(p.directory or ''):
                p.bind_menu_and_methods()
        PluginManager.plugins = plugins

    def start(self):
        logger.debug("Started 'app.application.FlikaApplication.start()'")
        self.show()
        self.raise_()
        QtWidgets.QApplication.processEvents()
        logger.debug("Started 'app.application.FlikaApplication.send_user_stats()'")
        self.send_user_stats_thread = Send_User_Stats_Thread()
        self.send_user_stats_thread.start()
        logger.debug("Completed 'app.application.FlikaApplication.send_user_stats()'")
        logger.debug("Completed 'app.application.FlikaApplication.start()'")
        #if 'PYCHARM_HOSTED' not in os.environ and 'SPYDER_SHELL_ID' not in os.environ:
        #    return self.app.exec_()

    def setWindowSize(self):
        #desktop = QtWidgets.QApplication.desktop()
        #width_px=int(desktop.logicalDpiX()*3.4)
        #height_px=int(desktop.logicalDpiY()*.9)
        #self.setGeometry(QtCore.QRect(15, 33, width_px, height_px))
        #self.setFixedSize(326, 80)
        #self.setMaximumSize(width_px*3, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum);
        self.setMinimumWidth(540)
        self.move(0, 0)

    def _make_menu(self):
        logger.debug("Started 'app.application.FlikaApplication._make_menu()'")
        from ..roi import open_rois
        from ..process.file_ import open_file, open_file_from_gui, open_image_sequence_from_gui, open_points, open_spt_results, save_file, save_movie_gui, save_points, save_rois, close
        from ..process.stacks import duplicate
        from ..process.filters import gaussian_blur
        from ..process.binary import threshold
        fileMenu = self.menuBar().addMenu('File')
        openMenu = fileMenu.addMenu("Open")
        openMenu.addAction("Open Image/Movie", open_file_from_gui)
        openMenu.addAction("Open Image Sequence", open_image_sequence_from_gui)
        openMenu.addAction("Open ROIs", open_rois)
        openMenu.addAction("Open Points", open_points)
        openMenu.addAction("Open SPT Results", open_spt_results)
        self.recentFileMenu = fileMenu.addMenu('Recent Files')
        self.recentFileMenu.aboutToShow.connect(self._make_recents)
        self.recentFileMenu.triggered.connect(lambda a: open_file(a.text()))
        saveMenu = fileMenu.addMenu("Save")
        saveMenu.addAction("Save Image", save_file)
        saveMenu.addAction("Save Movie (.mp4)", save_movie_gui)
        saveMenu.addAction("Save Points", save_points)
        saveMenu.addAction("Save All ROIs", save_rois)
        saveMenu.addSeparator()
        saveMenu.addAction("Export Provenance", self._export_provenance)

        fileMenu.addSeparator()
        interopMenu = fileMenu.addMenu("Interop")
        interopMenu.addAction("Send to napari", self._send_to_napari)
        interopMenu.addAction("Import from napari", self._import_from_napari)
        interopMenu.addSeparator()
        ijMenu = interopMenu.addMenu("ImageJ")
        ijMenu.addAction("Send to ImageJ", self._send_to_imagej)
        ijMenu.addAction("Import from ImageJ", self._import_from_imagej)
        interopMenu.addSeparator()
        interopMenu.addAction("Export OME-TIFF", self._export_ome_tiff)
        interopMenu.addAction("Export OME-Zarr", self._export_ome_zarr)

        fileMenu.addSeparator()
        fileMenu.addAction("Batch Process...", self._batch_process)
        fileMenu.addSeparator()
        fileMenu.addAction("Settings", SettingsEditor.show)
        fileMenu.addAction("&Quit", self.close)

        editMenu = self.menuBar().addMenu('Edit')
        undoAction = editMenu.addAction("&Undo", self._undo)
        undoAction.setShortcut('Ctrl+Z')
        redoAction = editMenu.addAction("&Redo", self._redo)
        redoAction.setShortcut('Ctrl+Shift+Z')

        # File shortcuts
        openAction = QtWidgets.QAction("&Open", self, triggered=lambda: open_file_from_gui())
        openAction.setShortcut('Ctrl+O')
        self.addAction(openAction)
        saveAction = QtWidgets.QAction("&Save", self, triggered=lambda: save_file())
        saveAction.setShortcut('Ctrl+S')
        self.addAction(saveAction)
        closeAction = QtWidgets.QAction("Close Window", self, triggered=lambda: close() if g.win else None)
        closeAction.setShortcut('Ctrl+W')
        self.addAction(closeAction)

        # Navigation shortcuts
        nextFrameAction = QtWidgets.QAction("Next Frame", self, triggered=self._next_frame)
        nextFrameAction.setShortcut('Right')
        self.addAction(nextFrameAction)
        prevFrameAction = QtWidgets.QAction("Previous Frame", self, triggered=self._prev_frame)
        prevFrameAction.setShortcut('Left')
        self.addAction(prevFrameAction)
        firstFrameAction = QtWidgets.QAction("First Frame", self, triggered=self._first_frame)
        firstFrameAction.setShortcut('Home')
        self.addAction(firstFrameAction)
        lastFrameAction = QtWidgets.QAction("Last Frame", self, triggered=self._last_frame)
        lastFrameAction.setShortcut('End')
        self.addAction(lastFrameAction)

        # View shortcuts
        dupAction = QtWidgets.QAction("Duplicate", self, triggered=lambda: duplicate() if g.win else None)
        dupAction.setShortcut('Ctrl+D')
        self.addAction(dupAction)
        gaussAction = QtWidgets.QAction("Gaussian Blur", self, triggered=lambda: gaussian_blur.gui() if g.win else None)
        gaussAction.setShortcut('Ctrl+G')
        self.addAction(gaussAction)
        threshAction = QtWidgets.QAction("Threshold", self, triggered=lambda: threshold.gui() if g.win else None)
        threshAction.setShortcut('Ctrl+T')
        self.addAction(threshAction)

        viewMenu = self.menuBar().addMenu('View')
        viewMenu.addAction('Orthogonal Views', lambda: g.win and g.win.toggleOrthogonalViews())
        viewMenu.addAction('3D Volume Viewer', lambda: g.win and g.win.toggleVolumeViewer())
        viewMenu.addAction('ROI Manager', self._toggle_roi_manager)
        viewMenu.addAction('Metadata Editor', self._show_metadata_editor)
        viewMenu.addAction('Figure Composer', self._show_figure_composer)
        viewMenu.addAction('Overlay Manager', self._show_overlay_manager)
        viewMenu.addAction('Counting Tool', self._show_counting_tool)

        for menu in g.menus:
            self.menuBar().addMenu(menu)

        self.pluginMenu = self.menuBar().addMenu('Plugins')
        self.pluginMenu.aboutToShow.connect(self._make_plugin_menu)

        self.scriptMenu = self.menuBar().addMenu('Scripts')
        self.scriptMenu.aboutToShow.connect(self._make_script_menu)

        cameraMenu = self.menuBar().addMenu('Camera')
        cameraMenu.addAction("Live Camera", self._show_camera)

        aiMenu = self.menuBar().addMenu('AI')
        aiMenu.addAction("Generate Script", self._ai_generate_script)
        aiMenu.addAction("Generate Plugin", self._ai_generate_plugin)
        aiMenu.addAction("AI Denoiser", self._ai_denoise)
        aiMenu.addAction("Pixel Classifier", self._ai_classify)
        aiMenu.addAction("Particle Localizer", self._ai_localize)
        aiSegMenu = aiMenu.addMenu("Segmentation")
        aiSegMenu.addAction("Cellpose", self._ai_cellpose)
        aiSegMenu.addAction("StarDist", self._ai_stardist)
        aiSegMenu.addAction("SAM Interactive", self._ai_sam)
        aiMenu.addAction("Object Detection", self._ai_detect)
        aiMenu.addAction("BioImage.IO Model Zoo", self._ai_model_zoo)

        helpMenu = self.menuBar().addMenu("Help")
        helpMenu.addAction("Documentation", self._show_documentation)
        url = 'http://flika-org.github.io'
        helpMenu.addAction("Online Documentation", lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)))
        helpMenu.addAction("Check For Updates", checkUpdates)
        helpMenu.addSeparator()
        helpMenu.addAction("Check Core Dependencies", self._check_core_deps)
        helpMenu.addAction("Check Plugin Dependencies", self._check_plugin_deps)
        helpMenu.addAction("GPU/Acceleration Status", self._show_gpu_status)
        logger.debug("Completed 'app.application.FlikaApplication._make_menu()'")

    def _undo(self):
        from ..core.undo import undo_stack
        undo_stack.undo()

    def _redo(self):
        from ..core.undo import undo_stack
        undo_stack.redo()

    def _next_frame(self):
        if g.win and hasattr(g.win, 'image') and g.win.image.ndim >= 3:
            idx = g.win.currentIndex + 1
            if idx < g.win.mt:
                g.win.setIndex(idx)

    def _prev_frame(self):
        if g.win and hasattr(g.win, 'image') and g.win.image.ndim >= 3:
            idx = g.win.currentIndex - 1
            if idx >= 0:
                g.win.setIndex(idx)

    def _first_frame(self):
        if g.win and hasattr(g.win, 'image') and g.win.image.ndim >= 3:
            g.win.setIndex(0)

    def _last_frame(self):
        if g.win and hasattr(g.win, 'image') and g.win.image.ndim >= 3:
            g.win.setIndex(g.win.mt - 1)

    def _toggle_roi_manager(self):
        from .roi_manager import ROIManager
        mgr = ROIManager.instance(self)
        if mgr.isVisible():
            mgr.hide()
        else:
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, mgr)
            mgr.show()

    def _ai_generate_script(self):
        from ..ai.assistant import _show_generate_script_dialog
        _show_generate_script_dialog()

    def _ai_generate_plugin(self):
        from ..ai.plugin_generator import _show_generate_plugin_dialog
        _show_generate_plugin_dialog()

    def _ai_cellpose(self):
        from ..ai.segmentation import cellpose_segment
        cellpose_segment.gui()

    def _ai_stardist(self):
        from ..ai.segmentation import stardist_segment
        stardist_segment.gui()

    def _ai_denoise(self):
        from ..ai.segmentation import ai_denoise
        ai_denoise.gui()

    def _ai_classify(self):
        from ..ai.segmentation import ai_classify
        ai_classify.gui()

    def _ai_localize(self):
        from ..ai.segmentation import ai_localize
        ai_localize.gui()

    def _ai_model_zoo(self):
        from ..ai.segmentation import ai_model_zoo
        ai_model_zoo.gui()

    def _ai_sam(self):
        from ..ai.segmentation import ai_sam
        ai_sam.gui()

    def _ai_detect(self):
        from ..ai.segmentation import ai_detect
        ai_detect.gui()

    def _show_overlay_manager(self):
        from ..viewers.overlay_manager import OverlayManagerPanel
        panel = OverlayManagerPanel.instance(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
        panel.show()
        panel.raise_()

    def _show_counting_tool(self):
        from ..viewers.counting_overlay import CountingPanel
        panel = CountingPanel.instance(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, panel)
        panel.show()
        panel.raise_()

    def _show_camera(self):
        from ..camera.camera_dialog import show_camera_dialog
        show_camera_dialog(parent=self)

    def _export_provenance(self):
        if g.win is None:
            g.alert('No window selected.')
            return
        from ..utils.misc import save_file_gui
        path = save_file_gui('Export Provenance', filetypes='JSON files (*.json)')
        if path:
            from ..utils.provenance import export_provenance
            export_provenance(g.win, path)
            g.m.statusBar().showMessage(f'Provenance exported to {os.path.basename(path)}')

    def _show_metadata_editor(self):
        from .metadata_editor import MetadataEditorDialog
        dlg = MetadataEditorDialog(parent=self)
        dlg.show()
        g.dialogs.append(dlg)

    def _show_figure_composer(self):
        from ..viewers.figure_composer import show_figure_composer
        show_figure_composer(parent=self)

    def _show_documentation(self):
        from .doc_browser import show_documentation
        show_documentation(parent=self)

    def _check_core_deps(self):
        from .dependency_checker import CoreDependencyDialog
        dlg = CoreDependencyDialog(self)
        dlg.exec()

    def _check_plugin_deps(self):
        from .dependency_checker import PluginDependencyDialog
        dlg = PluginDependencyDialog(self)
        dlg.exec()

    def _show_gpu_status(self):
        from .gpu_status import GPUStatusDialog
        dlg = GPUStatusDialog(self)
        dlg.exec()

    def _send_to_napari(self):
        try:
            from ..interop.napari_bridge import to_napari
            to_napari()
        except ImportError:
            g.alert('napari is not installed. Install with: pip install napari')

    def _import_from_napari(self):
        try:
            import napari
            from ..interop.napari_bridge import from_napari
            viewer = napari.current_viewer()
            if viewer is None or len(viewer.layers) == 0:
                g.alert('No napari viewer or layers found')
                return
            from_napari(viewer.layers[-1])
        except ImportError:
            g.alert('napari is not installed. Install with: pip install napari')

    def _send_to_imagej(self):
        try:
            from ..interop.imagej_bridge import to_imagej, _ij_instance
            if _ij_instance is None:
                self.statusBar().showMessage(
                    'Initializing ImageJ (first launch may download JRE)...')
                QtWidgets.QApplication.processEvents()
            self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                to_imagej()
            finally:
                self.unsetCursor()
                self.statusBar().clearMessage()
        except ImportError:
            g.alert('pyimagej is not installed. Install with: pip install pyimagej')
        except (EnvironmentError, OSError) as exc:
            g.alert(f'Could not initialize ImageJ: {exc}')

    def _import_from_imagej(self):
        try:
            from ..interop.imagej_bridge import from_imagej, _ij_instance
            if _ij_instance is None:
                self.statusBar().showMessage(
                    'Initializing ImageJ (first launch may download JRE)...')
                QtWidgets.QApplication.processEvents()
            self.setCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                from_imagej()
            finally:
                self.unsetCursor()
                self.statusBar().clearMessage()
        except ImportError:
            g.alert('pyimagej is not installed. Install with: pip install pyimagej')
        except (EnvironmentError, OSError) as exc:
            g.alert(f'Could not initialize ImageJ: {exc}')

    def _export_ome_tiff(self):
        from ..interop.ome import to_ome_tiff
        to_ome_tiff()

    def _export_ome_zarr(self):
        from ..interop.ome import to_ome_zarr
        to_ome_zarr()

    def _batch_process(self):
        from ..batch import BatchProcessor
        from ..utils.misc import open_file_gui
        script = open_file_gui('Select macro script', filetypes='*.py')
        if not script:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select input directory')
        if not directory:
            return
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select output directory')
        if not output_dir:
            return
        files = BatchProcessor.collect_files(directory)
        if not files:
            g.alert('No supported files found in directory')
            return
        bp = BatchProcessor(None, files, output_dir)
        bp.run_from_macro(script)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in g.__dict__:
            return g.__dict__[item]
        raise AttributeError(item)


    def _make_tools(self):
        self.freehand.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'freehand'))
        self.line.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'line'))
        self.rect_line.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'rect_line'))
        self.pencil.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'pencil'))
        self.rectangle.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'rectangle'))
        self.point.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'point'))
        self.mouse.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'mouse'))

        self.point.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.point.customContextMenuRequested.connect(pointSettings)
        self.pencil.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.pencil.customContextMenuRequested.connect(pencilSettings)
        self.rectangle.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.rectangle.customContextMenuRequested.connect(rectSettings)

        # Add ellipse button programmatically (after pencil at x=250)
        central = self.centralWidget()
        icon_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')

        self.ellipse_btn = QtWidgets.QPushButton(central)
        self.ellipse_btn.setGeometry(QtCore.QRect(290, 0, 41, 31))
        self.ellipse_btn.setCheckable(True)
        self.ellipse_btn.setAutoExclusive(True)
        self.ellipse_btn.setToolTip("Ellipse ROI")
        self.ellipse_btn.setIcon(QtGui.QIcon(os.path.join(icon_dir, 'ellipse.png')))
        self.ellipse_btn.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'ellipse'))

        # Add center-surround button
        self.cs_btn = QtWidgets.QPushButton(central)
        self.cs_btn.setGeometry(QtCore.QRect(330, 0, 41, 31))
        self.cs_btn.setCheckable(True)
        self.cs_btn.setAutoExclusive(True)
        self.cs_btn.setToolTip("Center-Surround ROI")
        self.cs_btn.setIcon(QtGui.QIcon(os.path.join(icon_dir, 'center_surround.png')))
        self.cs_btn.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'center_surround'))
        self.cs_btn.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.cs_btn.customContextMenuRequested.connect(csSettings)

        # Add point ROI button
        self.point_roi_btn = QtWidgets.QPushButton(central)
        self.point_roi_btn.setGeometry(QtCore.QRect(370, 0, 41, 31))
        self.point_roi_btn.setCheckable(True)
        self.point_roi_btn.setAutoExclusive(True)
        self.point_roi_btn.setToolTip("Point ROI (single pixel)")
        self.point_roi_btn.setIcon(QtGui.QIcon(os.path.join(icon_dir, 'point_roi.png')))
        self.point_roi_btn.clicked.connect(lambda: g.settings.__setitem__('mousemode', 'point_roi'))

    def _make_script_menu(self):
        logger.debug('Making script editor')
        from .macro_recorder import _toggle_recording, _save_macro, _run_macro, macro_recorder
        self.scriptMenu.clear()
        self.scriptEditorAction = self.scriptMenu.addAction('Script Editor', ScriptEditor.show)
        self.scriptMenu.addSeparator()
        rec_text = 'Stop Recording' if macro_recorder.is_recording else 'Record Macro'
        self.scriptMenu.addAction(rec_text, _toggle_recording)
        self.scriptMenu.addAction('Save Macro', _save_macro)
        self.scriptMenu.addAction('Run Macro', _run_macro)
        self.scriptMenu.addSeparator()
        def openScript(script):
            return lambda : ScriptEditor.importScript(script)
        for recent_script in g.settings['recent_scripts']:
            self.scriptMenu.addAction(recent_script, openScript(recent_script))
        self.scriptMenu.addSeparator()
        from .templates import template_manager
        template_manager.populate_menu(self.scriptMenu)
        logger.debug('Script editor complete')

    def _make_plugin_menu(self):
        logger.debug('Making Plugin Manager')
        from .plugin_manager import is_plugin_disabled
        self.pluginMenu.clear()
        self.pluginMenu.addAction('Plugin Manager', PluginManager.show)
        self.pluginMenu.addSeparator()
        logger.debug('Plugin Manager complete')

        installedPlugins = [plugin for plugin in PluginManager.plugins.values()
                           if plugin.installed and not is_plugin_disabled(plugin.directory or '')]
        for plugin in sorted(installedPlugins, key=lambda a: -a.lastModified()):
            if isinstance(plugin.menu, QtWidgets.QMenu):
                self.pluginMenu.addMenu(plugin.menu)

    def _make_recents(self):
        logger.debug('Making recent files')
        self.recentFileMenu.clear()
        g.settings['recent_files'] = [f for f in g.settings['recent_files'] if os.path.exists(f)]
        if len(g.settings['recent_files']) == 0:
            noAction = QtWidgets.QAction('No Recent Files', self.recentFileMenu)
            noAction.setEnabled(False)
            self.recentFileMenu.addAction(noAction)
        else:
            for name in g.settings['recent_files'][::-1]:
                if isinstance(name, str) and os.path.exists(name):
                    self.recentFileMenu.addAction(name)
        logger.debug('Recent files complete')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()   # must accept the dragEnterEvent or else the dropEvent can't occur !!!
        else:
            event.ignore()

    def dropEvent(self, event):
        from ..process.file_ import open_file
        if event.mimeData().hasUrls():   # if file or link is dropped
            for url in event.mimeData().urls():
                filename = url.toLocalFile()
                filename = str(filename)
                #if platform.system() == 'Windows':
                #    filename = filename.split('file:///')[1]
                #else:
                #    filename = filename.split('file://')[1]
                logger.debug("filename = '{}'".format(filename))
                open_file(filename)  # This fails on windows symbolic links.  http://stackoverflow.com/questions/15258506/os-path-islink-on-windows-with-python
                event.accept()
        else:
            event.ignore()

    def clear(self):
        """clear(self)
        Close all dialogs, trace windows, and windows
        """
        while g.dialogs:
            g.dialogs.pop(0).close()
        while g.traceWindows:
            g.traceWindows.pop(0).close()
        while g.windows:
            g.windows.pop(0).close()

    def closeEvent(self, event):
        """closeEvent(self, event)
        Close all widgets and exit flika
        """
        logger.info('Closing flika')
        event.accept()
        ScriptEditor.close()
        PluginManager.close()
        self.clear()
        g.settings.save()
        if g.m == self:
            g.m = None

class SetCurrentWindowSignal(QtWidgets.QWidget):
    sig=QtCore.Signal()

    def __init__(self,parent):
        super(SetCurrentWindowSignal, self).__init__(parent)
        self.hide()
logger.debug("Completed 'reading app/application.py'")