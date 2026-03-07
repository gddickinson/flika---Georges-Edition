"""AI-powered plugin generator for flika.

Generates a BaseProcess subclass from a natural-language description,
validates the result with ``ast.parse``, and optionally saves it to the
user's plugin directory.
"""
from __future__ import annotations

import ast
import os
from typing import Optional

from ..logger import logger

_SYSTEM_PROMPT = """\
You are an expert Python developer who writes *flika* plugins.

A flika plugin is a Python module containing a subclass of
``flika.utils.BaseProcess.BaseProcess``.  The subclass must:

1. Implement ``__init__`` calling ``super().__init__()``.
2. Implement ``gui()`` that sets up ``self.items`` and calls ``super().gui()``.
3. Implement ``__call__(self, ..., keepSourceWindow=False)`` that calls
   ``self.start(keepSourceWindow)``, processes ``self.tif``, sets
   ``self.newtif`` and ``self.newname``, and returns ``self.end()``.

Example skeleton:

    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox
    from flika import global_vars as g

    class MyProcess(BaseProcess):
        def __init__(self):
            super().__init__()
        def gui(self):
            self.gui_reset()
            # add items …
            super().gui()
        def __call__(self, keepSourceWindow=False):
            self.start(keepSourceWindow)
            self.newtif = self.tif  # processing here
            self.newname = self.oldname + ' - MyProcess'
            return self.end()
    my_process = MyProcess()

Return ONLY valid Python code (no markdown fences).
"""


class PluginGenerator:
    """Generate a flika plugin from a description."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for AI features. "
                "Install with:  pip install anthropic  (or  pip install flika[ai])"
            )
        if api_key is None:
            from ..app.settings_editor import get_api_key
            api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "No API key found.  Set ANTHROPIC_API_KEY environment variable "
                "or enter your key in Edit > Settings."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate_plugin(self, description: str, name: str = "generated_plugin") -> str:
        """Generate plugin code and validate it with ast.parse."""
        logger.info("AI plugin generator: creating %r from %r", name, description)
        prompt = f"Create a flika plugin named '{name}' that: {description}"
        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        code = message.content[0].text

        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax errors: {e}") from e

        return code

    def save_plugin(self, code: str, name: str) -> str:
        """Save generated plugin to ``~/.FLIKA/plugins/<name>/``."""
        from os.path import expanduser
        plugin_dir = os.path.join(expanduser("~"), '.FLIKA', 'plugins', name)
        os.makedirs(plugin_dir, exist_ok=True)
        filepath = os.path.join(plugin_dir, f'{name}.py')
        with open(filepath, 'w') as f:
            f.write(code)
        logger.info("Plugin saved to %s", filepath)
        return filepath


def _show_generate_plugin_dialog():
    """Menu callback for AI > Generate Plugin."""
    from qtpy import QtWidgets
    from .. import global_vars as g

    description, ok = QtWidgets.QInputDialog.getMultiLineText(
        g.m, "AI Plugin Generator",
        "Describe the plugin you want to create:"
    )
    if not ok or not description.strip():
        return

    name, ok2 = QtWidgets.QInputDialog.getText(
        g.m, "Plugin Name", "Enter a name for the plugin:"
    )
    if not ok2 or not name.strip():
        name = "generated_plugin"

    try:
        gen = PluginGenerator()
        code = gen.generate_plugin(description, name)
    except Exception as e:
        g.alert(f"Plugin generation error: {e}")
        return

    from qtpy.QtWidgets import QMessageBox
    reply = g.messageBox(
        "Save Plugin?",
        f"Plugin '{name}' generated successfully.\nSave to ~/.FLIKA/plugins/{name}/?",
        QMessageBox.Yes | QMessageBox.No,
    )
    if reply == QMessageBox.Yes:
        try:
            path = gen.save_plugin(code, name)
            g.alert(f"Plugin saved to {path}")
        except Exception as e:
            g.alert(f"Failed to save plugin: {e}")
    else:
        from ..app.script_editor import ScriptEditor
        ScriptEditor.show()
        ScriptEditor.gui.editor.setPlainText(code)
