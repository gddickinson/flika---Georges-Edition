"""Rich AI Plugin Generator dialog for flika."""
from __future__ import annotations

import ast
import os
import sys
import importlib
import traceback
from qtpy import QtWidgets, QtCore, QtGui
from ..logger import logger


_PLUGIN_INSTRUCTIONS = """\
<h3>AI Plugin Generator — How to Use</h3>
<p>Generate complete flika plugins from natural-language descriptions.
Plugins are reusable tools that appear in flika's Plugin menu.</p>

<h4>What makes a good plugin description:</h4>
<ul>
<li><b>Be specific:</b> "A plugin that applies a Gaussian blur with adjustable sigma and kernel size"</li>
<li><b>Mention GUI elements:</b> "Include a slider for threshold value and a checkbox for preview"</li>
<li><b>Describe the processing:</b> "Take the current image, apply bilateral filtering, then edge detection"</li>
<li><b>Specify parameters:</b> "Default sigma=2.0, range 0.1 to 20.0"</li>
</ul>

<h4>Plugin Architecture:</h4>
<ul>
<li>Plugins subclass <code>BaseProcess</code> (or <code>BaseProcess_noPriorWindow</code>)</li>
<li><code>gui()</code> sets up sliders, checkboxes, and combo boxes</li>
<li><code>__call__()</code> processes <code>self.tif</code> and sets <code>self.newtif</code></li>
<li>Convenience methods: <code>add_slider()</code>, <code>add_checkbox()</code>, <code>add_combo()</code></li>
<li>Saved to <code>~/.FLIKA/plugins/&lt;name&gt;/</code></li>
</ul>

<h4>After generation:</h4>
<ul>
<li><b>Preview</b> the code before saving</li>
<li><b>Save &amp; Load</b> to install immediately (no restart needed!)</li>
<li><b>Edit</b> in the Script Editor for modifications</li>
<li>Plugins appear in the <b>Plugins</b> menu</li>
</ul>
"""


class _GenerateWorker(QtCore.QThread):
    """Background thread for plugin generation."""
    sig_result = QtCore.Signal(str)
    sig_error = QtCore.Signal(str)

    def __init__(self, generator, description, name):
        super().__init__()
        self.generator = generator
        self.description = description
        self.name = name

    def run(self):
        try:
            code = self.generator.generate_plugin(self.description, self.name)
            self.sig_result.emit(code)
        except Exception as e:
            self.sig_error.emit(f"{type(e).__name__}: {e}")


class PluginGeneratorDialog(QtWidgets.QDialog):
    """AI Plugin Generator dialog with preview and auto-load."""

    _instance = None

    @classmethod
    def show_dialog(cls):
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent=None)
        cls._instance.show()
        cls._instance.raise_()
        cls._instance.activateWindow()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Plugin Generator")
        self.resize(900, 750)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self._generator = None
        self._worker = None
        self._generated_code = ""

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Instructions (collapsible)
        self._instructions_btn = QtWidgets.QPushButton("Show Instructions")
        self._instructions_btn.setCheckable(True)
        self._instructions_btn.toggled.connect(self._toggle_instructions)
        layout.addWidget(self._instructions_btn)

        self._instructions = QtWidgets.QTextBrowser()
        self._instructions.setHtml(_PLUGIN_INSTRUCTIONS)
        self._instructions.setMaximumHeight(300)
        self._instructions.setVisible(False)
        layout.addWidget(self._instructions)

        # Plugin name
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Plugin Name:"))
        self._name_input = QtWidgets.QLineEdit()
        self._name_input.setPlaceholderText("my_plugin")
        name_layout.addWidget(self._name_input, 1)
        layout.addLayout(name_layout)

        # Description
        layout.addWidget(QtWidgets.QLabel("Describe the plugin:"))
        self._description = QtWidgets.QPlainTextEdit()
        self._description.setMaximumHeight(120)
        self._description.setPlaceholderText(
            "e.g., A plugin that detects bright spots using Laplacian of Gaussian, "
            "with adjustable sigma and threshold parameters...")
        layout.addWidget(self._description)

        # Generate button
        gen_layout = QtWidgets.QHBoxLayout()
        self._generate_btn = QtWidgets.QPushButton("Generate Plugin")
        self._generate_btn.setMinimumHeight(36)
        self._generate_btn.clicked.connect(self._on_generate)
        gen_layout.addWidget(self._generate_btn)

        self._auto_load = QtWidgets.QCheckBox("Load in current session")
        self._auto_load.setChecked(True)
        self._auto_load.setToolTip(
            "Automatically load the plugin into flika after saving (no restart needed)")
        gen_layout.addWidget(self._auto_load)
        layout.addLayout(gen_layout)

        # Preview pane
        layout.addWidget(QtWidgets.QLabel("Generated Code Preview:"))
        self._preview = QtWidgets.QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Courier New', Courier, monospace;
                font-size: 13px;
                padding: 8px;
            }
        """)
        # Add syntax highlighting
        try:
            from ..app.syntax import PythonHighlighter
            self._highlighter = PythonHighlighter(self._preview.document())
        except Exception:
            pass
        layout.addWidget(self._preview, 1)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self._btn_save = QtWidgets.QPushButton("Save to Plugins")
        self._btn_save.setToolTip("Save the plugin to ~/.FLIKA/plugins/ and optionally load it")
        self._btn_save.clicked.connect(self._save_plugin)
        self._btn_save.setEnabled(False)
        btn_layout.addWidget(self._btn_save)

        self._btn_editor = QtWidgets.QPushButton("Open in Script Editor")
        self._btn_editor.setToolTip("Open the code in the Script Editor for editing")
        self._btn_editor.clicked.connect(self._open_in_editor)
        self._btn_editor.setEnabled(False)
        btn_layout.addWidget(self._btn_editor)

        self._btn_regenerate = QtWidgets.QPushButton("Regenerate")
        self._btn_regenerate.setToolTip("Generate the plugin again with the same description")
        self._btn_regenerate.clicked.connect(self._on_generate)
        self._btn_regenerate.setEnabled(False)
        btn_layout.addWidget(self._btn_regenerate)

        self._btn_copy = QtWidgets.QPushButton("Copy to Clipboard")
        self._btn_copy.clicked.connect(self._copy_to_clipboard)
        self._btn_copy.setEnabled(False)
        btn_layout.addWidget(self._btn_copy)

        btn_layout.addStretch()

        self._btn_push = QtWidgets.QPushButton("Push to GitHub")
        self._btn_push.setToolTip(
            "Push the saved plugin to the flika plugins GitHub repository")
        self._btn_push.clicked.connect(self._push_to_github)
        self._btn_push.setEnabled(False)
        btn_layout.addWidget(self._btn_push)

        layout.addLayout(btn_layout)

        # Status bar
        self._status = QtWidgets.QLabel()
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._status)
        self._update_status()

        # Ctrl+Enter shortcut
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self._description)
        shortcut.activated.connect(self._on_generate)

    def _toggle_instructions(self, checked):
        self._instructions.setVisible(checked)
        self._instructions_btn.setText(
            "Hide Instructions" if checked else "Show Instructions")

    def _update_status(self, msg=None):
        if msg:
            self._status.setText(msg)
            return
        try:
            from .. import global_vars as g
            model = g.settings['ai_model'] or 'claude-sonnet-4-20250514'
        except Exception:
            model = 'claude-sonnet-4-20250514'
        self._status.setText(f"Model: {model}")

    def _on_generate(self):
        description = self._description.toPlainText().strip()
        if not description:
            self._update_status("Please enter a description.")
            return

        name = self._name_input.text().strip()
        if not name:
            name = "generated_plugin"
            self._name_input.setText(name)

        # Sanitize name
        name = name.replace(" ", "_").replace("-", "_")
        self._name_input.setText(name)

        self._generate_btn.setEnabled(False)
        self._generate_btn.setText("Generating...")
        self._btn_regenerate.setEnabled(False)
        self._preview.setPlainText("Generating plugin code...")
        self._update_status("Calling AI model...")

        try:
            if self._generator is None:
                from .plugin_generator import PluginGenerator
                self._generator = PluginGenerator()

            self._worker = _GenerateWorker(
                self._generator, description, name)
            self._worker.sig_result.connect(self._on_result)
            self._worker.sig_error.connect(self._on_error)
            self._worker.finished.connect(self._on_finished)
            self._worker.start()
        except Exception as e:
            self._on_error(str(e))

    def _on_result(self, code):
        self._generated_code = code
        self._preview.setPlainText(code)
        self._btn_save.setEnabled(True)
        self._btn_editor.setEnabled(True)
        self._btn_copy.setEnabled(True)
        self._update_status("Plugin generated successfully! Review the code below.")

    def _on_error(self, msg):
        self._preview.setPlainText(f"Error: {msg}")
        self._update_status(f"Generation failed: {msg}")
        self._on_finished()

    def _on_finished(self):
        self._generate_btn.setEnabled(True)
        self._generate_btn.setText("Generate Plugin")
        self._btn_regenerate.setEnabled(True)

    def _save_plugin(self):
        if not self._generated_code:
            return

        name = self._name_input.text().strip() or "generated_plugin"
        description = self._description.toPlainText().strip()

        try:
            from .plugin_generator import PluginGenerator
            gen = self._generator or PluginGenerator()
            path = gen.save_plugin(self._generated_code, name, description)
            self._saved_path = path
            self._update_status(f"Plugin saved to {path}")
            self._btn_push.setEnabled(True)

            # Auto-load if checked
            if self._auto_load.isChecked():
                self._load_plugin_live(name, path)
        except Exception as e:
            self._update_status(f"Save failed: {e}")
            logger.error("Plugin save error: %s", e)

    def _load_plugin_live(self, name, filepath):
        """Dynamically load a saved plugin into the current session."""
        try:
            plugin_dir = os.path.dirname(filepath)
            parent_dir = os.path.dirname(plugin_dir)

            # Ensure plugin directory is on sys.path
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the module
            module_name = name
            if module_name in sys.modules:
                # Reload if already imported
                mod = importlib.reload(sys.modules[module_name])
            else:
                mod = importlib.import_module(module_name)

            # Look for BaseProcess subclass instances and add to Plugins menu
            from ..utils.BaseProcess import BaseProcess, BaseProcess_noPriorWindow
            from .. import global_vars as g

            loaded_any = False
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, (BaseProcess, BaseProcess_noPriorWindow)):
                    # Add to Plugins menu
                    if hasattr(g, 'm') and g.m is not None:
                        plugin_menu = g.m.pluginMenu
                        display_name = name.replace("_", " ").title()
                        action = QtWidgets.QAction(
                            display_name, g.m,
                            triggered=obj.gui)
                        plugin_menu.addAction(action)
                        loaded_any = True
                        logger.info("Loaded AI plugin '%s' into Plugins menu", display_name)

            if loaded_any:
                self._update_status(
                    f"Plugin '{name}' saved and loaded! Check the Plugins menu.")
            else:
                self._update_status(
                    f"Plugin saved to {os.path.dirname(filepath)}. "
                    "No BaseProcess instance found to auto-load.")

        except Exception as e:
            logger.error("Failed to auto-load plugin: %s", e)
            self._update_status(
                f"Plugin saved but auto-load failed: {e}. "
                "Restart flika to load the plugin.")

    def _open_in_editor(self):
        if not self._generated_code:
            return
        from ..app.script_editor import ScriptEditor
        ScriptEditor.show()
        editor = ScriptEditor.gui.addEditor()
        editor.setPlainText(self._generated_code)
        self._update_status("Code opened in Script Editor.")

    def _copy_to_clipboard(self):
        if not self._generated_code:
            return
        QtWidgets.QApplication.clipboard().setText(self._generated_code)
        self._update_status("Copied to clipboard!")
        QtCore.QTimer.singleShot(2000, self._update_status)

    def _push_to_github(self):
        """Push the saved plugin to the flika plugins GitHub repository."""
        if not hasattr(self, '_saved_path') or not self._saved_path:
            self._update_status("Save the plugin first before pushing to GitHub.")
            return

        name = self._name_input.text().strip() or "generated_plugin"
        plugin_dir = os.path.dirname(self._saved_path)

        try:
            from .. import global_vars as g
            repo_url = g.settings.get(
                'plugin_repo_url',
                'https://github.com/gddickinson/flika_plugins')
        except Exception:
            repo_url = 'https://github.com/gddickinson/flika_plugins'

        # Check if git is available
        import subprocess
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._update_status("Git is not installed. Install git to push plugins.")
            return

        # Confirm with user
        reply = QtWidgets.QMessageBox.question(
            self, "Push to GitHub",
            f"This will:\n\n"
            f"1. Initialize a git repo in:\n   {plugin_dir}\n"
            f"2. Commit all plugin files\n"
            f"3. Push to: {repo_url}\n\n"
            f"You may need to configure git credentials.\n\n"
            f"Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            # Initialize git repo if needed
            if not os.path.exists(os.path.join(plugin_dir, '.git')):
                subprocess.run(['git', 'init'], cwd=plugin_dir,
                               capture_output=True, check=True)

            # Add all files
            subprocess.run(['git', 'add', '.'], cwd=plugin_dir,
                           capture_output=True, check=True)

            # Commit
            subprocess.run(
                ['git', 'commit', '-m',
                 f'Add AI-generated plugin: {name}'],
                cwd=plugin_dir, capture_output=True, check=True)

            # Add remote if needed
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=plugin_dir, capture_output=True)
            if result.returncode != 0:
                subprocess.run(
                    ['git', 'remote', 'add', 'origin', repo_url],
                    cwd=plugin_dir, capture_output=True, check=True)

            # Push
            result = subprocess.run(
                ['git', 'push', '-u', 'origin', 'main'],
                cwd=plugin_dir, capture_output=True)
            if result.returncode != 0:
                # Try master branch
                result = subprocess.run(
                    ['git', 'push', '-u', 'origin', 'master'],
                    cwd=plugin_dir, capture_output=True)

            if result.returncode == 0:
                self._update_status(f"Plugin '{name}' pushed to {repo_url}")
            else:
                stderr = result.stderr.decode('utf-8', errors='replace')
                self._update_status(f"Push failed: {stderr[:200]}")
                QtWidgets.QMessageBox.warning(
                    self, "Push Failed",
                    f"Git push failed:\n\n{stderr}\n\n"
                    "You may need to:\n"
                    "- Configure git credentials (git config)\n"
                    "- Create the repository on GitHub first\n"
                    "- Set up SSH keys or a personal access token")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            self._update_status(f"Git error: {stderr[:200]}")
            logger.error("Git push error: %s", stderr)
        except Exception as e:
            self._update_status(f"Push failed: {e}")
            logger.error("Push to GitHub error: %s", e)
