# -*- coding: utf-8 -*-
"""Plugin Manager for flika.

Provides:
- Enable/disable individual plugins (persisted in settings)
- Control plugin logging verbosity (startup messages, debug, etc.)
- Download plugins from a configurable central repository (default: GitHub)
- Remove plugins from the plugins folder
- Plugin search, install, update, and dependency management
"""
from ..logger import logger
logger.debug("Started 'reading app/plugin_manager.py'")

from glob import glob
import os, sys, difflib, zipfile, time, shutil, traceback, subprocess, json
from os.path import expanduser
from qtpy import QtGui, QtWidgets, QtCore
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
import threading
import tempfile
from xml.etree import ElementTree
import platform
import importlib.metadata
from packaging.version import Version

from .. import global_vars as g
from ..utils.misc import load_ui
from ..images import image_path

# Legacy plugin list (kept for backwards compatibility)
plugin_list = {
    'Beam Splitter':    'https://raw.githubusercontent.com/BrettJSettle/BeamSplitter/master/',
    'Detect Puffs':     'https://raw.githubusercontent.com/kyleellefsen/detect_puffs/master/',
    'Global Analysis':  'https://raw.githubusercontent.com/BrettJSettle/GlobalAnalysisPlugin/master/',
    'Pynsight':         'http://raw.githubusercontent.com/kyleellefsen/pynsight/master/',
    'QuantiMus':        'http://raw.githubusercontent.com/Quantimus/quantimus/master/',
    'Rodent Tracker':   'https://raw.githubusercontent.com/kyleellefsen/rodentTracker/master/'
}

DEFAULT_REPO_URL = 'https://github.com/gddickinson/flika_plugins'
DEFAULT_REPO_RAW = 'https://raw.githubusercontent.com/gddickinson/flika_plugins/main'

# Plugin settings keys in g.settings
_SETTINGS_KEY_DISABLED = 'disabled_plugins'          # list of directory names
_SETTINGS_KEY_LOG_LEVEL = 'plugin_log_level'         # 'normal', 'verbose', 'quiet'
_SETTINGS_KEY_REPO_URL = 'plugin_repo_url'           # custom repository URL
_SETTINGS_KEY_SUPPRESS_STARTUP = 'plugin_suppress_startup_msgs'  # bool


def _get_plugin_settings():
    """Ensure plugin settings exist and return them."""
    if g.settings[_SETTINGS_KEY_DISABLED] is None:
        g.settings.d[_SETTINGS_KEY_DISABLED] = []
    if g.settings[_SETTINGS_KEY_LOG_LEVEL] is None:
        g.settings.d[_SETTINGS_KEY_LOG_LEVEL] = 'normal'
    if g.settings[_SETTINGS_KEY_REPO_URL] is None:
        g.settings.d[_SETTINGS_KEY_REPO_URL] = DEFAULT_REPO_URL
    if g.settings[_SETTINGS_KEY_SUPPRESS_STARTUP] is None:
        g.settings.d[_SETTINGS_KEY_SUPPRESS_STARTUP] = False
    return {
        'disabled': g.settings[_SETTINGS_KEY_DISABLED],
        'log_level': g.settings[_SETTINGS_KEY_LOG_LEVEL],
        'repo_url': g.settings[_SETTINGS_KEY_REPO_URL],
        'suppress_startup': g.settings[_SETTINGS_KEY_SUPPRESS_STARTUP],
    }


def is_plugin_disabled(directory_name):
    """Check if a plugin is disabled by its directory name."""
    disabled = g.settings[_SETTINGS_KEY_DISABLED]
    if disabled is None:
        return False
    return directory_name in disabled


def set_plugin_enabled(directory_name, enabled=True):
    """Enable or disable a plugin by directory name. Persisted in settings."""
    disabled = g.settings[_SETTINGS_KEY_DISABLED]
    if disabled is None:
        disabled = []
    if enabled and directory_name in disabled:
        disabled.remove(directory_name)
    elif not enabled and directory_name not in disabled:
        disabled.append(directory_name)
    g.settings[_SETTINGS_KEY_DISABLED] = disabled


def plugin_log(msg, level='info'):
    """Log a plugin message, respecting the configured log level."""
    log_level = g.settings[_SETTINGS_KEY_LOG_LEVEL] or 'normal'
    if log_level == 'quiet' and level != 'error':
        return
    if log_level == 'normal' and level == 'debug':
        return
    getattr(logger, level, logger.info)(msg)


# ---------------------------------------------------------------------------
# Plugin output filter — intercepts print() and logging during plugin imports
# ---------------------------------------------------------------------------

import io
import logging as _logging

class _NullStream(io.TextIOBase):
    """A write-only stream that discards everything (like /dev/null)."""
    def write(self, s):
        return len(s)
    def writelines(self, lines):
        pass

class _LoggedStream(io.TextIOBase):
    """A stream that routes lines to the flika plugin logger."""
    def __init__(self, plugin_name, real_stdout):
        super().__init__()
        self._plugin_name = plugin_name
        self._real_stdout = real_stdout
        self._buf = ''

    def write(self, s):
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line.strip():
                plugin_log(f"[{self._plugin_name}] {line}", 'info')
        return len(s)

    def flush(self):
        if self._buf.strip():
            plugin_log(f"[{self._plugin_name}] {self._buf}", 'info')
            self._buf = ''

class _PluginOutputFilter:
    """Context manager that filters plugin print() and logging output.

    Redirects sys.stdout/stderr instead of replacing builtins.print,
    which avoids conflicts with libraries like numba that introspect
    the print builtin.

    In old flika (which doesn't use this class), plugins behave exactly
    as before — their code is never touched.
    """

    def __init__(self, plugin_name=''):
        self.plugin_name = plugin_name
        self._old_stdout = None
        self._old_stderr = None
        self._log_filters = []

    def __enter__(self):
        settings = _get_plugin_settings()
        log_level = settings['log_level']
        suppress_startup = settings['suppress_startup']

        # --- Redirect stdout/stderr ---
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        if log_level == 'quiet' or suppress_startup:
            sys.stdout = _NullStream()
            sys.stderr = _NullStream()
        elif log_level == 'normal':
            sys.stdout = _LoggedStream(self.plugin_name, self._old_stdout)
            # Leave stderr visible (warnings/errors should still show)

        # else 'verbose' — leave everything alone

        # --- Adjust logging threshold for plugin loggers ---
        if log_level == 'quiet':
            self._raise_logging_threshold(_logging.ERROR)
        elif log_level == 'normal':
            self._raise_logging_threshold(_logging.WARNING)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout/stderr
        if self._old_stdout is not None:
            # Flush the logged stream if applicable
            if hasattr(sys.stdout, 'flush'):
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
            sys.stdout = self._old_stdout
            self._old_stdout = None
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
            self._old_stderr = None

        # Remove temporary log filters
        for handler, filt in self._log_filters:
            handler.removeFilter(filt)
        self._log_filters.clear()

        return False  # don't suppress exceptions

    def _raise_logging_threshold(self, min_level):
        """Add a temporary filter to all root-logger handlers."""
        class _ThresholdFilter(_logging.Filter):
            def filter(self, record):
                return record.levelno >= min_level

        filt = _ThresholdFilter()
        root = _logging.getLogger()
        for handler in root.handlers:
            handler.addFilter(filt)
            self._log_filters.append((handler, filt))


helpHTML = '''
<h1 style="width:100%; text-align:center">Flika Plugin Manager</h1>
<p>Use the tabs to manage your plugins:</p>
<ul>
    <li><b>Installed</b> — Enable/disable, configure, or remove installed plugins</li>
    <li><b>Repository</b> — Browse and download plugins from a central repository</li>
    <li><b>Settings</b> — Configure repository URL, logging, and startup behavior</li>
</ul>
<h3>Develop a Plugin</h3>
<ul>
    <li>1. Download the <a href="https://github.com/flika-org/flika_plugin_template">Plugin Template</a></li>
    <li>2. Place it in your .FLIKA/plugins directory</li>
    <li>3. Update the info.xml file</li>
    <li>4. See the <a href="http://flika-org.github.io/writing_plugins.html">Documentation</a></li>
</ul>
'''


def get_plugin_directory():
    logger.debug('Calling app.plugin_manager.get_plugin_directory')
    local_flika_directory = os.path.join(expanduser("~"), '.FLIKA')
    plugin_directory = os.path.join(expanduser("~"), '.FLIKA', 'plugins')
    if not os.path.exists(plugin_directory):
        os.makedirs(plugin_directory)
    if not os.path.isfile(os.path.join(plugin_directory, '__init__.py')):
        open(os.path.join(plugin_directory, '__init__.py'), 'a').close()
    if plugin_directory not in sys.path:
        sys.path.append(plugin_directory)
    if local_flika_directory not in sys.path:
        sys.path.append(local_flika_directory)
    return plugin_directory

plugin_dir = get_plugin_directory()


def parse(x):
    tree = ElementTree.fromstring(x)
    def step(item):
        d = {}
        if item.text and item.text.strip():
            d['#text'] = item.text.strip()
        for k, v in item.items():
            d['@%s' % k] = v
        for k in list(item):
            if k.tag not in d:
                d[k.tag] = step(k)
            elif type(d[k.tag]) == list:
                d[k.tag].append(step(k))
            else:
                d[k.tag] = [d[k.tag], step(k)]
        if len(d) == 1 and '#text' in d:
            return d['#text']
        return d
    return step(tree)


def str2func(plugin_name, file_location, function):
    '''
    takes plugin_name, path to object, function as arguments
    imports plugin_name.path and gets the function from that imported object
    to be run when an action is clicked.

    The import is wrapped in _PluginOutputFilter so startup messages are
    controlled.  The returned callable is also wrapped so that runtime
    print/logging from plugin operations respects the configured log level.
    '''
    with _PluginOutputFilter(plugin_name):
        __import__(plugin_name)
        plugin_dir_str = "plugins.{}.{}".format(plugin_name, file_location)
        levels = function.split('.')
        module = __import__(plugin_dir_str, fromlist=[levels[0]]).__dict__[levels[0]]
        for i in range(1, len(levels)):
            module = getattr(module, levels[i])

    # Wrap the callable so runtime output is also filtered.
    # Qt's triggered signal passes a `checked` bool — drop it so
    # plugin gui() methods (which take no arguments) aren't broken.
    if callable(module):
        raw_func = module
        import functools
        @functools.wraps(raw_func)
        def _filtered_call(*args, **kwargs):
            with _PluginOutputFilter(plugin_name):
                return raw_func()
        return _filtered_call
    return module


def fake_str2func(plugin_name, file_location, function):
    def fake_fun():
        logger.debug(str(function))
    return fake_fun


def build_submenu(module_name, parent_menu, layout_dict):
    if len(layout_dict) == 0:
        g.alert("Error building submenu for the plugin '{}'. No items found in 'menu_layout' in the info.xml file.".format(module_name))
    for key, value in layout_dict.items():
        if type(value) != list:
            value = [value]
        if key == 'menu':
            for v in value:
                menu = parent_menu.addMenu(v["@name"])
                build_submenu(module_name, menu, v)
        elif key == 'action':
            for od in value:
                method = str2func(module_name, od['@location'], od['@function'])
                if method is not None:
                    action = QtWidgets.QAction(od['#text'], parent_menu, triggered=method)
                    parent_menu.addAction(action)


class Plugin():
    def __init__(self, name=None, info_url=None):
        self.name = name
        self.directory = None
        self.url = None
        self.author = None
        self.documentation = None
        self.version = ''
        self.latest_version = ''
        self.menu = None
        self.listWidget = QtWidgets.QListWidgetItem(self.name)
        self.installed = False
        self.description = ''
        self.dependencies = []
        self.loaded = False
        self.info_url = info_url
        if info_url:
            self.update_info()

    def lastModified(self):
        return os.path.getmtime(os.path.join(plugin_dir, self.directory))

    def fromLocal(self, path):
        with open(os.path.join(path, 'info.xml'), 'r') as f:
            text = f.read()
        info = parse(text)
        self.name = info['@name']
        self.directory = info['directory']
        self.version = info['version']
        self.latest_version = self.version
        self.author = info['author']
        with open(os.path.join(path, 'about.html'), 'r') as f:
            try:
                self.description = str(f.read())
            except FileNotFoundError:
                self.description = "No local description file found"
        self.url = info['url'] if 'url' in info else None
        self.documentation = info['documentation'] if 'documentation' in info else None
        if 'dependencies' in info and 'dependency' in info['dependencies']:
            deps = info['dependencies']['dependency']
            self.dependencies = [d['@name'] for d in deps] if isinstance(deps, list) else [deps['@name']]
        self.menu_layout = info.pop('menu_layout')
        self.listWidget = QtWidgets.QListWidgetItem(self.name)
        self.listWidget.setIcon(QtGui.QIcon(image_path('check.png')))
        self.loaded = True

    def bind_menu_and_methods(self):
        if len(self.menu_layout) > 0:
            self.menu = QtWidgets.QMenu(self.name)
            with _PluginOutputFilter(self.name):
                build_submenu(self.directory, self.menu, self.menu_layout)
        else:
            self.menu = None

    def update_info(self):
        logger.debug('Calling app.plugin_manager.update_info')
        if self.info_url is None:
            return False
        info_url = urljoin(self.info_url, 'info.xml')
        try:
            txt = urlopen(info_url).read()
        except HTTPError as e:
            g.alert("Failed to update information for {}.\n\t{}".format(self.name, e))
            return

        new_info = parse(txt)
        description_url = urljoin(self.info_url, 'about.html')
        try:
            new_info['description'] = urlopen(description_url).read().decode('utf-8')
        except HTTPError:
            new_info['description'] = "Unable to get description for {0} from <a href={1}>{1}</a>".format(self.name, description_url)
        self.menu_layout = new_info.pop('menu_layout')
        if 'date' in new_info:
            new_info['version'] = '.'.join(new_info['date'].split('/')[2:] + new_info['date'].split('/')[:2])
            new_info.pop('date')
        new_info['latest_version'] = new_info.pop('version')
        if 'dependencies' in new_info and 'dependency' in new_info['dependencies']:
            deps = new_info.pop('dependencies')['dependency']
            self.dependencies = [d['@name'] for d in deps] if isinstance(deps, list) else [deps['@name']]
        self.__dict__.update(new_info)
        self.loaded = True


# ---------------------------------------------------------------------------
# Plugin Manager GUI — full rewrite with tabs
# ---------------------------------------------------------------------------

class PluginManager(QtWidgets.QMainWindow):
    plugins = {}
    loadThread = None
    sigPluginLoaded = QtCore.Signal(str)

    @staticmethod
    def show():
        logger.debug('Calling app.plugin_manager.PluginManager.show')
        if not hasattr(PluginManager, 'gui') or PluginManager.gui is None:
            PluginManager.gui = PluginManager()
        PluginManager.gui._refresh_installed_list()
        QtWidgets.QMainWindow.show(PluginManager.gui)
        PluginManager.gui.raise_()
        if not os.access(plugin_dir, os.W_OK):
            g.alert("Plugin folder write permission denied. Restart flika as administrator to enable plugin installation.")

    @staticmethod
    def close():
        if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
            QtWidgets.QMainWindow.close(PluginManager.gui)

    @staticmethod
    def refresh_online_plugins():
        logger.debug('Calling app.plugin_manager.PluginManager.refresh_online_plugins()')
        for p in plugin_list.keys():
            PluginManager.load_online_plugin(p)

    @staticmethod
    def load_online_plugin(p):
        logger.debug('Calling app.plugin_manager.PluginManager.load_online_plugin()')
        if p not in plugin_list or (PluginManager.loadThread is not None and PluginManager.loadThread.is_alive()):
            return

        def loadThread():
            plug = PluginManager.plugins[p]
            plug.info_url = plugin_list[p]
            plug.update_info()
            if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
                PluginManager.gui.sigPluginLoaded.emit(p)

        PluginManager.loadThread = threading.Thread(None, loadThread)
        if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
            PluginManager.gui.statusBar().showMessage('Loading plugin information for {}...'.format(p))
        PluginManager.loadThread.start()

    def closeEvent(self, ev):
        if self.loadThread is not None and self.loadThread.is_alive():
            self.loadThread.join(0)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Plugin Manager')
        self.resize(850, 600)

        # Central widget with tab layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        self._tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self._tabs)

        # Tab 1: Installed Plugins
        self._build_installed_tab()

        # Tab 2: Repository Browser
        self._build_repository_tab()

        # Tab 3: Settings
        self._build_settings_tab()

        self.statusBar().showMessage('Ready')
        self.sigPluginLoaded.connect(self._on_plugin_loaded)

    # -----------------------------------------------------------------------
    # Tab 1: Installed Plugins
    # -----------------------------------------------------------------------
    def _build_installed_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        # Left panel: plugin list with checkboxes
        left = QtWidgets.QVBoxLayout()
        self._installed_search = QtWidgets.QLineEdit()
        self._installed_search.setPlaceholderText("Search installed plugins...")
        self._installed_search.textChanged.connect(self._filter_installed_list)
        left.addWidget(self._installed_search)

        self._installed_list = QtWidgets.QListWidget()
        self._installed_list.currentItemChanged.connect(self._on_installed_selected)
        left.addWidget(self._installed_list)

        # Buttons under the list
        btn_row = QtWidgets.QHBoxLayout()
        self._btn_enable_all = QtWidgets.QPushButton("Enable All")
        self._btn_enable_all.clicked.connect(self._enable_all_plugins)
        btn_row.addWidget(self._btn_enable_all)

        self._btn_disable_all = QtWidgets.QPushButton("Disable All")
        self._btn_disable_all.clicked.connect(self._disable_all_plugins)
        btn_row.addWidget(self._btn_disable_all)
        left.addLayout(btn_row)

        btn_row2 = QtWidgets.QHBoxLayout()
        self._btn_open_dir = QtWidgets.QPushButton("Open Plugins Folder")
        self._btn_open_dir.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl("file:///" + plugin_dir)))
        btn_row2.addWidget(self._btn_open_dir)

        self._btn_reload = QtWidgets.QPushButton("Reload Plugins")
        self._btn_reload.clicked.connect(self._reload_plugins)
        btn_row2.addWidget(self._btn_reload)
        left.addLayout(btn_row2)

        layout.addLayout(left, 1)

        # Right panel: plugin details
        right = QtWidgets.QVBoxLayout()

        self._detail_name = QtWidgets.QLabel("")
        self._detail_name.setStyleSheet("font-size: 14px; font-weight: bold;")
        right.addWidget(self._detail_name)

        self._detail_info = QtWidgets.QLabel("")
        right.addWidget(self._detail_info)

        self._detail_desc = QtWidgets.QTextBrowser()
        self._detail_desc.setOpenExternalLinks(True)
        right.addWidget(self._detail_desc)

        # Action buttons
        detail_btns = QtWidgets.QHBoxLayout()
        detail_btns.addStretch()

        self._btn_toggle = QtWidgets.QPushButton("Enable")
        self._btn_toggle.clicked.connect(self._toggle_selected_plugin)
        detail_btns.addWidget(self._btn_toggle)

        self._btn_docs = QtWidgets.QPushButton("Documentation")
        self._btn_docs.clicked.connect(self._open_docs)
        detail_btns.addWidget(self._btn_docs)

        self._btn_remove = QtWidgets.QPushButton("Remove")
        self._btn_remove.clicked.connect(self._remove_selected_plugin)
        detail_btns.addWidget(self._btn_remove)

        right.addLayout(detail_btns)

        layout.addLayout(right, 2)

        self._tabs.addTab(tab, "Installed")

    def _refresh_installed_list(self):
        self._installed_list.clear()
        disabled = g.settings[_SETTINGS_KEY_DISABLED] or []

        installed = [p for p in PluginManager.plugins.values() if p.installed]
        installed.sort(key=lambda p: p.name.lower())

        for plugin in installed:
            item = QtWidgets.QListWidgetItem(plugin.name)
            item.setData(QtCore.Qt.UserRole, plugin.directory)
            if plugin.directory in disabled:
                item.setIcon(QtGui.QIcon(image_path('exclamation.png')))
                item.setForeground(QtGui.QColor(150, 150, 150))
            else:
                item.setIcon(QtGui.QIcon(image_path('check.png')))
            self._installed_list.addItem(item)

        if self._installed_list.count() == 0:
            self._detail_desc.setHtml(helpHTML)

    def _filter_installed_list(self, text):
        text = text.lower()
        for i in range(self._installed_list.count()):
            item = self._installed_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _on_installed_selected(self, current, previous=None):
        if current is None:
            return
        name = current.text()
        plugin = PluginManager.plugins.get(name)
        if plugin is None:
            return

        self._detail_name.setText(plugin.name)
        info_parts = []
        if plugin.author:
            info_parts.append(f"Author: {plugin.author}")
        if plugin.version:
            info_parts.append(f"Version: {plugin.version}")
        if plugin.dependencies:
            info_parts.append(f"Dependencies: {', '.join(plugin.dependencies)}")

        directory = plugin.directory or ''
        disabled = is_plugin_disabled(directory)
        info_parts.append(f"Status: {'Disabled' if disabled else 'Enabled'}")
        self._detail_info.setText(" | ".join(info_parts))

        desc = plugin.description or "No description available."
        self._detail_desc.setHtml(desc)

        self._btn_toggle.setText("Enable" if disabled else "Disable")
        self._btn_docs.setVisible(plugin.documentation is not None)
        self._btn_remove.setVisible(True)
        self._btn_toggle.setVisible(True)

    def _toggle_selected_plugin(self):
        item = self._installed_list.currentItem()
        if item is None:
            return
        directory = item.data(QtCore.Qt.UserRole)
        if directory is None:
            return

        disabled = is_plugin_disabled(directory)
        set_plugin_enabled(directory, enabled=disabled)  # toggle

        action = "Enabled" if disabled else "Disabled"
        self.statusBar().showMessage(f"{item.text()} {action.lower()}. Restart flika for changes to take effect.")

        # Update visual
        self._refresh_installed_list()
        # Re-select same item
        for i in range(self._installed_list.count()):
            it = self._installed_list.item(i)
            if it.data(QtCore.Qt.UserRole) == directory:
                self._installed_list.setCurrentItem(it)
                break

    def _enable_all_plugins(self):
        g.settings[_SETTINGS_KEY_DISABLED] = []
        self.statusBar().showMessage("All plugins enabled. Restart flika for changes to take effect.")
        self._refresh_installed_list()

    def _disable_all_plugins(self):
        disabled = []
        for p in PluginManager.plugins.values():
            if p.installed and p.directory:
                disabled.append(p.directory)
        g.settings[_SETTINGS_KEY_DISABLED] = disabled
        self.statusBar().showMessage("All plugins disabled. Restart flika for changes to take effect.")
        self._refresh_installed_list()

    def _remove_selected_plugin(self):
        item = self._installed_list.currentItem()
        if item is None:
            return
        name = item.text()
        plugin = PluginManager.plugins.get(name)
        if plugin is None:
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Remove Plugin",
            f"Are you sure you want to remove '{name}'?\n\n"
            "This will delete the plugin folder and cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)

        if reply != QtWidgets.QMessageBox.Yes:
            return

        PluginManager.removePlugin(plugin)
        self._refresh_installed_list()

    def _open_docs(self):
        item = self._installed_list.currentItem()
        if item is None:
            return
        plugin = PluginManager.plugins.get(item.text())
        if plugin and plugin.documentation:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(plugin.documentation))

    def _reload_plugins(self):
        """Reload all plugins from disk (re-scan plugin directory)."""
        self.statusBar().showMessage("Reloading plugins...")
        # Re-run the load thread
        thread = Load_Local_Plugins_Thread()
        thread.plugins_done_sig.connect(self._on_reload_done)
        thread.error_loading.connect(lambda msg: self.statusBar().showMessage(msg))
        thread.start()
        self._reload_thread = thread  # prevent GC

    def _on_reload_done(self, plugins):
        for p in plugins.values():
            if p.loaded:
                p.bind_menu_and_methods()
        PluginManager.plugins = plugins
        self._refresh_installed_list()
        self.statusBar().showMessage("Plugins reloaded.")

    # -----------------------------------------------------------------------
    # Tab 2: Repository Browser
    # -----------------------------------------------------------------------
    def _build_repository_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Header
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Repository:"))
        self._repo_url_label = QtWidgets.QLabel("")
        self._repo_url_label.setStyleSheet("color: #0066cc;")
        header.addWidget(self._repo_url_label, 1)

        self._btn_refresh_repo = QtWidgets.QPushButton("Refresh")
        self._btn_refresh_repo.clicked.connect(self._fetch_repo_plugins)
        header.addWidget(self._btn_refresh_repo)
        layout.addLayout(header)

        # Search
        self._repo_search = QtWidgets.QLineEdit()
        self._repo_search.setPlaceholderText("Search repository...")
        self._repo_search.textChanged.connect(self._filter_repo_list)
        layout.addWidget(self._repo_search)

        # Split: list + details
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self._repo_list = QtWidgets.QListWidget()
        self._repo_list.currentItemChanged.connect(self._on_repo_selected)
        splitter.addWidget(self._repo_list)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        self._repo_detail_name = QtWidgets.QLabel("")
        self._repo_detail_name.setStyleSheet("font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self._repo_detail_name)

        self._repo_detail_info = QtWidgets.QLabel("")
        right_layout.addWidget(self._repo_detail_info)

        self._repo_detail_desc = QtWidgets.QTextBrowser()
        self._repo_detail_desc.setOpenExternalLinks(True)
        right_layout.addWidget(self._repo_detail_desc)

        repo_btns = QtWidgets.QHBoxLayout()
        repo_btns.addStretch()

        self._btn_install_repo = QtWidgets.QPushButton("Install")
        self._btn_install_repo.clicked.connect(self._install_from_repo)
        self._btn_install_repo.setVisible(False)
        repo_btns.addWidget(self._btn_install_repo)

        right_layout.addLayout(repo_btns)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        self._repo_plugins = {}  # name -> {url, description, author, ...}
        self._tabs.addTab(tab, "Repository")

    def _fetch_repo_plugins(self):
        """Fetch list of available plugins from the configured repository."""
        repo_url = g.settings[_SETTINGS_KEY_REPO_URL] or DEFAULT_REPO_URL
        self._repo_url_label.setText(repo_url)
        self.statusBar().showMessage("Fetching plugin list from repository...")
        self._btn_refresh_repo.setEnabled(False)

        thread = RepoFetchThread(repo_url)
        thread.sig_done.connect(self._on_repo_fetched)
        thread.sig_error.connect(self._on_repo_error)
        thread.start()
        self._repo_fetch_thread = thread

    def _on_repo_fetched(self, plugins_data):
        self._btn_refresh_repo.setEnabled(True)
        self._repo_plugins = plugins_data
        self._repo_list.clear()

        installed_dirs = {p.directory for p in PluginManager.plugins.values() if p.installed}

        for name, info in sorted(plugins_data.items()):
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, name)
            # Mark if already installed
            repo_dir = info.get('directory', name)
            if repo_dir in installed_dirs:
                item.setIcon(QtGui.QIcon(image_path('check.png')))
                item.setToolTip("Already installed")
            self._repo_list.addItem(item)

        self.statusBar().showMessage(f"Found {len(plugins_data)} plugins in repository.")

    def _on_repo_error(self, msg):
        self._btn_refresh_repo.setEnabled(True)
        self.statusBar().showMessage(f"Repository error: {msg}")
        g.alert(f"Failed to fetch plugins from repository:\n{msg}")

    def _filter_repo_list(self, text):
        text = text.lower()
        for i in range(self._repo_list.count()):
            item = self._repo_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _on_repo_selected(self, current, previous=None):
        if current is None:
            self._btn_install_repo.setVisible(False)
            return
        name = current.data(QtCore.Qt.UserRole)
        info = self._repo_plugins.get(name, {})

        self._repo_detail_name.setText(name)

        info_parts = []
        if info.get('author'):
            info_parts.append(f"Author: {info['author']}")
        if info.get('version'):
            info_parts.append(f"Version: {info['version']}")
        self._repo_detail_info.setText(" | ".join(info_parts))

        desc = info.get('description', 'No description available.')
        self._repo_detail_desc.setHtml(desc)

        # Check if already installed
        installed_dirs = {p.directory for p in PluginManager.plugins.values() if p.installed}
        repo_dir = info.get('directory', name)
        already_installed = repo_dir in installed_dirs
        self._btn_install_repo.setVisible(True)
        self._btn_install_repo.setText("Reinstall" if already_installed else "Install")

    def _install_from_repo(self):
        item = self._repo_list.currentItem()
        if item is None:
            return
        name = item.data(QtCore.Qt.UserRole)
        info = self._repo_plugins.get(name, {})
        download_url = info.get('download_url')

        if not download_url:
            g.alert(f"No download URL available for '{name}'.")
            return

        self.statusBar().showMessage(f"Installing {name}...")
        self._btn_install_repo.setEnabled(False)

        thread = RepoInstallThread(name, download_url, info)
        thread.sig_done.connect(self._on_repo_install_done)
        thread.sig_error.connect(self._on_repo_install_error)
        thread.start()
        self._repo_install_thread = thread

    def _on_repo_install_done(self, name):
        self._btn_install_repo.setEnabled(True)
        self.statusBar().showMessage(f"'{name}' installed successfully. Restart flika to load.")
        self._refresh_installed_list()
        self._fetch_repo_plugins()  # refresh icons

    def _on_repo_install_error(self, msg):
        self._btn_install_repo.setEnabled(True)
        self.statusBar().showMessage(f"Install failed: {msg}")
        g.alert(f"Plugin installation failed:\n{msg}")

    # -----------------------------------------------------------------------
    # Tab 3: Settings
    # -----------------------------------------------------------------------
    def _build_settings_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        settings = _get_plugin_settings()

        # Repository URL
        self._repo_url_edit = QtWidgets.QLineEdit(settings['repo_url'])
        self._repo_url_edit.setPlaceholderText(DEFAULT_REPO_URL)
        layout.addRow("Repository URL:", self._repo_url_edit)

        self._btn_reset_repo = QtWidgets.QPushButton("Reset to Default")
        self._btn_reset_repo.clicked.connect(
            lambda: self._repo_url_edit.setText(DEFAULT_REPO_URL))
        layout.addRow("", self._btn_reset_repo)

        layout.addRow(QtWidgets.QLabel(""))  # spacer

        # Log level
        self._log_level_combo = QtWidgets.QComboBox()
        self._log_level_combo.addItems(['quiet', 'normal', 'verbose'])
        idx = ['quiet', 'normal', 'verbose'].index(settings['log_level'])
        self._log_level_combo.setCurrentIndex(idx)
        layout.addRow("Plugin Log Level:", self._log_level_combo)

        log_help = QtWidgets.QLabel(
            "<small>quiet = errors only | normal = info + warnings | "
            "verbose = all debug messages</small>")
        log_help.setWordWrap(True)
        layout.addRow("", log_help)

        layout.addRow(QtWidgets.QLabel(""))  # spacer

        # Suppress startup messages
        self._suppress_startup = QtWidgets.QCheckBox(
            "Suppress plugin startup messages")
        self._suppress_startup.setChecked(settings['suppress_startup'])
        layout.addRow("", self._suppress_startup)

        layout.addRow(QtWidgets.QLabel(""))  # spacer

        # Plugin directory info
        dir_label = QtWidgets.QLabel(f"Plugin directory: <code>{plugin_dir}</code>")
        dir_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addRow(dir_label)

        # Save button
        self._btn_save_settings = QtWidgets.QPushButton("Save Settings")
        self._btn_save_settings.clicked.connect(self._save_settings)
        layout.addRow("", self._btn_save_settings)

        self._tabs.addTab(tab, "Settings")

    def _save_settings(self):
        with g.settings.batch():
            g.settings[_SETTINGS_KEY_REPO_URL] = self._repo_url_edit.text().strip() or DEFAULT_REPO_URL
            g.settings[_SETTINGS_KEY_LOG_LEVEL] = self._log_level_combo.currentText()
            g.settings[_SETTINGS_KEY_SUPPRESS_STARTUP] = self._suppress_startup.isChecked()
        self.statusBar().showMessage("Settings saved.")

    # -----------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------
    def _on_plugin_loaded(self, name):
        self.statusBar().showMessage(f"Finished loading {name}")

    @staticmethod
    def local_plugin_paths():
        paths = []
        for path in glob(os.path.join(plugin_dir, "*")):
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'info.xml')):
                paths.append(path)
        return paths

    @staticmethod
    def removePlugin(plugin):
        if os.path.isdir(os.path.join(plugin_dir, plugin.directory, '.git')):
            g.alert("This plugin's directory is managed by git. To remove, manually delete the directory.")
            return False
        try:
            shutil.rmtree(os.path.join(plugin_dir, plugin.directory), ignore_errors=True)
            plugin.version = ''
            plugin.menu = None
            plugin.installed = False
            if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
                PluginManager.gui.statusBar().showMessage(
                    f'{plugin.name} successfully uninstalled. Restart flika to complete removal.')
        except Exception as e:
            g.alert(
                title="Plugin Uninstall Failed",
                msg=f"Unable to remove the folder at {plugin.name}\n{e}\n"
                    "Delete the folder manually to uninstall the plugin",
                icon=QtWidgets.QMessageBox.Warning)

    @staticmethod
    def downloadPlugin(plugin):
        if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
            PluginManager.gui.statusBar().showMessage("Installing plugin")
        if isinstance(plugin, str):
            if plugin in PluginManager.plugins:
                plugin = PluginManager.plugins[plugin]
            else:
                return
        if plugin.url is None:
            return
        failed = []
        dists = [d.metadata['Name'] for d in importlib.metadata.distributions()]
        if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
            PluginManager.gui.statusBar().showMessage(
                "Installing dependencies for %s" % plugin.name)
        for pl in plugin.dependencies:
            try:
                if pl in dists:
                    continue
                __import__(pl)
            except ImportError:
                res = subprocess.call([sys.executable, '-m', 'pip', 'install',
                                       '{}'.format(pl), '--no-cache-dir'])
                if res != 0:
                    failed.append(pl)
        if failed:
            g.alert("Failed to install dependencies for {}:\n{}\n"
                    "You must install them on your own before installing this plugin.".format(
                        plugin.name, ', '.join(failed)))
            return

        if os.path.exists(os.path.join(plugin_dir, plugin.directory)):
            g.alert("A folder with name {} already exists in the plugins directory. "
                    "Please remove it to install this plugin!".format(plugin.directory))
            return

        # SECURITY NOTE: Plugin downloads lack checksum/signature verification.
        # A server-side component (e.g., SHA-256 hashes in the plugin index)
        # would be needed to verify download integrity. Until then, plugins
        # are trusted based on the repository URL alone.
        try:
            data = urlopen(plugin.url).read()
        except Exception:
            g.alert(title="Download Error",
                    msg=f"Failed to connect to download {plugin.name}. "
                        "Check your internet connection and try again.",
                    icon=QtWidgets.QMessageBox.Warning)
            return

        try:
            with tempfile.TemporaryFile() as tf:
                tf.write(data)
                tf.seek(0)
                with zipfile.ZipFile(tf) as z:
                    folder_name = os.path.dirname(z.namelist()[0])
                    z.extractall(plugin_dir)

            directory = os.path.join(plugin_dir, plugin.directory)
            os.rename(os.path.join(plugin_dir, folder_name), directory)
        except (PermissionError, Exception) as e:
            if isinstance(e, PermissionError):
                g.alert("Unable to download plugin to {}. Rerun flika as administrator.".format(
                    plugin.name), title='Permission Denied')
            else:
                g.alert("Error occurred while installing {}.\n\t{}".format(
                    plugin.name, e), title='Plugin Install Failed')
            return

        plugin.version = plugin.latest_version
        plugin.installed = True
        if hasattr(PluginManager, 'gui') and PluginManager.gui is not None:
            PluginManager.gui.statusBar().showMessage(
                f'Successfully installed {plugin.name}')


# ---------------------------------------------------------------------------
# Repository fetch/install threads
# ---------------------------------------------------------------------------

class RepoFetchThread(QtCore.QThread):
    """Fetch plugin metadata from a GitHub repository."""
    sig_done = QtCore.Signal(dict)    # {name: {info dict}}
    sig_error = QtCore.Signal(str)

    def __init__(self, repo_url):
        super().__init__()
        self.repo_url = repo_url.rstrip('/')

    def run(self):
        try:
            plugins = self._fetch_github_repo()
            self.sig_done.emit(plugins)
        except Exception as e:
            self.sig_error.emit(str(e))

    def _fetch_github_repo(self):
        """Fetch plugin list from a GitHub repository.

        Tries GitHub API first, falls back to raw content.
        """
        plugins = {}

        # Parse owner/repo from URL
        parts = self.repo_url.rstrip('/').split('/')
        if 'github.com' in self.repo_url:
            # https://github.com/owner/repo
            idx = parts.index('github.com')
            owner = parts[idx + 1]
            repo = parts[idx + 2] if len(parts) > idx + 2 else ''
        else:
            # Fallback: try treating last two segments as owner/repo
            owner = parts[-2] if len(parts) >= 2 else ''
            repo = parts[-1] if len(parts) >= 1 else ''

        if not owner or not repo:
            raise ValueError(f"Cannot parse GitHub owner/repo from: {self.repo_url}")

        # Try GitHub API to list directories (each = a plugin)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        try:
            req = Request(api_url, headers={'User-Agent': 'flika-plugin-manager'})
            resp = urlopen(req, timeout=15)
            contents = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed to access GitHub API: {e}")

        for item in contents:
            if item.get('type') != 'dir':
                continue
            dir_name = item['name']
            if dir_name.startswith('.') or dir_name.startswith('_'):
                continue

            # Try to fetch info.xml from this directory
            info = self._fetch_plugin_info(owner, repo, dir_name)
            if info:
                plugins[info.get('name', dir_name)] = info

        return plugins

    def _fetch_plugin_info(self, owner, repo, dir_name):
        """Fetch info.xml for a single plugin directory."""
        raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/main"
        info_url = f"{raw_base}/{dir_name}/info.xml"

        try:
            req = Request(info_url, headers={'User-Agent': 'flika-plugin-manager'})
            txt = urlopen(req, timeout=10).read()
        except Exception:
            # Try 'master' branch instead of 'main'
            info_url_master = info_url.replace('/main/', '/master/')
            try:
                req = Request(info_url_master, headers={'User-Agent': 'flika-plugin-manager'})
                txt = urlopen(req, timeout=10).read()
                raw_base = raw_base.replace('/main', '/master')
            except Exception:
                return None

        try:
            info = parse(txt)
        except Exception:
            return None

        result = {
            'name': info.get('@name', dir_name),
            'directory': info.get('directory', dir_name),
            'version': info.get('version', ''),
            'author': info.get('author', ''),
            'url': info.get('url', ''),
            'documentation': info.get('documentation', ''),
        }

        # Try to fetch description
        about_url = f"{raw_base}/{dir_name}/about.html"
        try:
            req = Request(about_url, headers={'User-Agent': 'flika-plugin-manager'})
            result['description'] = urlopen(req, timeout=10).read().decode('utf-8')
        except Exception:
            result['description'] = f"Plugin: {result['name']}"

        # Dependencies
        if 'dependencies' in info and 'dependency' in info['dependencies']:
            deps = info['dependencies']['dependency']
            result['dependencies'] = [d['@name'] for d in deps] if isinstance(deps, list) else [deps['@name']]
        else:
            result['dependencies'] = []

        # Download URL: GitHub archive zipball for the specific folder
        # Use the full repo zipball — we'll extract only the plugin folder
        result['download_url'] = f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
        result['_owner'] = owner
        result['_repo'] = repo
        result['_dir_name'] = dir_name

        return result


class RepoInstallThread(QtCore.QThread):
    """Download and install a plugin from a repository."""
    sig_done = QtCore.Signal(str)   # plugin name
    sig_error = QtCore.Signal(str)

    def __init__(self, name, download_url, info):
        super().__init__()
        self.name = name
        self.download_url = download_url
        self.info = info

    def run(self):
        try:
            self._install()
            self.sig_done.emit(self.name)
        except Exception as e:
            self.sig_error.emit(str(e))

    def _install(self):
        dir_name = self.info.get('_dir_name', self.info.get('directory', self.name))
        target = os.path.join(plugin_dir, dir_name)

        if os.path.exists(target):
            # Remove existing to allow reinstall
            shutil.rmtree(target, ignore_errors=True)

        # Install dependencies first
        deps = self.info.get('dependencies', [])
        if deps:
            dists = {d.metadata['Name'].lower() for d in importlib.metadata.distributions()}
            for dep in deps:
                if dep.lower() in dists:
                    continue
                try:
                    __import__(dep)
                except ImportError:
                    subprocess.call([sys.executable, '-m', 'pip', 'install',
                                     dep, '--no-cache-dir'])

        # Download the repository zip
        owner = self.info.get('_owner', '')
        repo = self.info.get('_repo', '')

        # Try main branch, then master
        for branch in ['main', 'master']:
            url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
            try:
                req = Request(url, headers={'User-Agent': 'flika-plugin-manager'})
                data = urlopen(req, timeout=60).read()
                break
            except Exception:
                data = None

        if data is None:
            raise RuntimeError(f"Failed to download from {self.download_url}")

        # Extract only the plugin directory from the zip
        with tempfile.TemporaryFile() as tf:
            tf.write(data)
            tf.seek(0)
            with zipfile.ZipFile(tf) as z:
                # GitHub zips have a top-level folder like "repo-main/"
                top = z.namelist()[0].split('/')[0]
                prefix = f"{top}/{dir_name}/"

                # Find all files under the plugin directory
                members = [m for m in z.namelist() if m.startswith(prefix)]
                if not members:
                    raise RuntimeError(
                        f"Plugin directory '{dir_name}' not found in repository.")

                # Extract to a temp dir first, then move
                with tempfile.TemporaryDirectory() as tmpdir:
                    for member in members:
                        z.extract(member, tmpdir)
                    extracted = os.path.join(tmpdir, top, dir_name)
                    if os.path.isdir(extracted):
                        shutil.copytree(extracted, target)
                    else:
                        raise RuntimeError(
                            f"Extracted path is not a directory: {extracted}")


# ---------------------------------------------------------------------------
# Plugin loading classes (preserved from original)
# ---------------------------------------------------------------------------

class PluginDescriptor:
    """Lightweight metadata-only representation of a plugin.

    Parsed from ``info.xml`` without importing the plugin's Python code, so
    startup stays fast.
    """
    __slots__ = ('name', 'directory', 'version', 'author', 'path',
                 'menu_layout', 'dependencies')

    def __init__(self, path):
        with open(os.path.join(path, 'info.xml'), 'r') as f:
            info = parse(f.read())
        self.name = info['@name']
        self.directory = info['directory']
        self.version = info.get('version', '0.0.0')
        self.author = info.get('author', 'Unknown')
        self.path = path
        self.menu_layout = info.pop('menu_layout', {})
        if 'dependencies' in info and 'dependency' in info['dependencies']:
            deps = info['dependencies']['dependency']
            self.dependencies = ([d['@name'] for d in deps]
                                 if isinstance(deps, list)
                                 else [deps['@name']])
        else:
            self.dependencies = []


class LazyPlugin:
    """Wraps a :class:`PluginDescriptor` so that the real import only happens
    when the user clicks on the plugin's menu item."""

    def __init__(self, descriptor: PluginDescriptor):
        self.descriptor = descriptor
        self._loaded = False
        self._real_plugin = None

    @property
    def name(self):
        return self.descriptor.name

    def _ensure_loaded(self):
        if self._loaded:
            return
        p = Plugin(self.descriptor.name)
        p.fromLocal(self.descriptor.path)
        p.bind_menu_and_methods()
        self._real_plugin = p
        self._loaded = True

    def get_menu(self):
        self._ensure_loaded()
        return self._real_plugin.menu if self._real_plugin else None


class Load_Local_Plugins_Thread(QtCore.QThread):
    plugins_done_sig = QtCore.Signal(dict)
    error_loading = QtCore.Signal(str)

    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        plugins = {n: Plugin(n) for n in plugin_list}
        installed_plugins = {}
        disabled = g.settings[_SETTINGS_KEY_DISABLED] or []
        suppress = g.settings[_SETTINGS_KEY_SUPPRESS_STARTUP] or False

        for pluginPath in PluginManager.local_plugin_paths():
            # Check if this plugin is disabled
            dir_name = os.path.basename(pluginPath)

            if dir_name in disabled:
                # Still register it but don't load/bind
                try:
                    p = Plugin()
                    p.fromLocal(pluginPath)
                    p.installed = True
                    p.menu = None  # Don't create menu for disabled plugin
                    plugins[p.name] = p
                    if not suppress:
                        plugin_log(f"Plugin '{p.name}' is disabled — skipping.", 'debug')
                except Exception:
                    pass
                continue

            p = Plugin()
            p.fromLocal(pluginPath)
            try:
                # Filter wraps bind_menu_and_methods which triggers
                # plugin imports via str2func — this is where plugins
                # print startup messages, configure logging, etc.
                with _PluginOutputFilter(p.name):
                    p.bind_menu_and_methods()
                if p.name not in plugins.keys() or p.name not in installed_plugins.keys():
                    p.installed = True
                    plugins[p.name] = p
                    installed_plugins[p.name] = p
                else:
                    g.alert('Could not load the plugin {}. There is already a plugin with this same name. '
                            'Change the plugin name in the info.xml file'.format(p.name))
            except Exception as e:
                msg = "Could not load plugin {}".format(pluginPath)
                self.error_loading.emit(msg)
                logger.error(msg)
                ex_type, ex, tb = sys.exc_info()
                sys.excepthook(ex_type, ex, tb)
        self.plugins_done_sig.emit(plugins)


logger.debug("Completed 'reading app/plugin_manager.py'")
