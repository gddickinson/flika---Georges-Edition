"""AI-powered plugin generator for flika.

Generates a BaseProcess subclass from a natural-language description,
validates the result with ``ast.parse``, and optionally saves it to the
user's plugin directory.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Optional

from flika.logger import logger


def _load_guidelines() -> str:
    """Load plugin guidelines from ~/.FLIKA/plugin_guidelines.md if it exists."""
    path = os.path.join(os.path.expanduser("~"), '.FLIKA', 'plugin_guidelines.md')
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _validate_plugin_structure(code: str) -> list[str]:
    """Validate generated plugin code against critical rules.

    Returns a list of warning strings (empty = all checks passed).
    """
    warnings = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Code has syntax errors"]

    # Find all class definitions (top-level)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        warnings.append("No class definition found")
        return warnings

    cls = classes[0]
    class_name = cls.name

    # Check for module-level instance: ClassName()
    has_instance = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if (isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Name) and
                    node.value.func.id == class_name):
                has_instance = True
                break
    if not has_instance:
        warnings.append(
            f"No module-level instance of {class_name}() found — "
            "plugin will not load")

    # Check class methods
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    # Check gui() method
    if 'gui' in methods:
        gui_src = ast.get_source_segment(code, methods['gui'])
        if gui_src:
            if 'gui_reset' not in gui_src:
                warnings.append("gui() missing self.gui_reset() call")
            if 'super().gui()' not in gui_src and 'super().gui(' not in gui_src:
                warnings.append("gui() missing super().gui() call")

    # Check __call__() method
    if '__call__' in methods:
        call_node = methods['__call__']
        call_src = ast.get_source_segment(code, call_node)
        if call_src:
            # self.start() is ALWAYS required — even for noPriorWindow
            # (it sets self.command which self.end() needs)
            if 'self.start(' not in call_src and 'self.start()' not in call_src:
                warnings.append(
                    "__call__() missing self.start() — causes "
                    "AttributeError: 'command' when end() is called")
            if 'self.end()' not in call_src:
                warnings.append("__call__() missing self.end() call")
            if 'return' not in call_src or 'self.end()' not in call_src:
                warnings.append("__call__() should end with return self.end()")
    else:
        warnings.append("No __call__() method found")

    return warnings

_SYSTEM_PROMPT = """\
You are an expert Python developer who writes *flika* plugins.

Source code: {source_url}
Plugin repository: https://github.com/gddickinson/flika_plugins

# Plugin Architecture
A flika plugin is a Python module containing a subclass of
``flika.utils.BaseProcess.BaseProcess``.  The subclass must:

1. Implement ``__init__`` calling ``super().__init__()``.
2. Implement ``gui()`` that sets up ``self.items`` and calls ``super().gui()``.
3. Implement ``__call__(self, ..., keepSourceWindow=False)`` that calls
   ``self.start(keepSourceWindow)``, processes ``self.tif``, sets
   ``self.newtif`` and ``self.newname``, and returns ``self.end()``.

# Example Plugin Skeleton

    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
    from flika import global_vars as g
    import numpy as np

    class MyFilter(BaseProcess):
        def __init__(self):
            super().__init__()

        def gui(self):
            self.gui_reset()
            # Convenience methods (preferred):
            self.add_slider('sigma', 'Sigma', 2.0, 0.1, 20.0, decimals=2)
            self.add_checkbox('keep', 'Keep Source', checked=False)
            self.add_combo('method', 'Method', ['Option A', 'Option B'])
            # Or manual approach:
            # slider = SliderLabel(2); slider.setRange(0.1, 20); slider.setValue(2.0)
            # self.items.append({{'name': 'sigma', 'string': 'Sigma', 'object': slider}})
            super().gui()

        def __call__(self, sigma=2.0, keep=False, method='Option A',
                     keepSourceWindow=False):
            self.start(keepSourceWindow)
            # self.tif is the numpy array from the active window
            # Process: (images are 2D=YX, 3D=TYX, 4D=TYXZ)
            from scipy.ndimage import gaussian_filter
            if self.tif.ndim == 2:
                self.newtif = gaussian_filter(self.tif, sigma)
            else:
                self.newtif = np.zeros_like(self.tif)
                for i in range(len(self.tif)):
                    self.newtif[i] = gaussian_filter(self.tif[i], sigma)
            self.newname = self.oldname + ' - MyFilter'
            return self.end()

    my_filter = MyFilter()

# For plugins that don't need an existing window:
    from flika.utils.BaseProcess import BaseProcess_noPriorWindow
    # IMPORTANT: noPriorWindow plugins MUST still call self.start() (no args)
    # in __call__() before self.end().  start() sets self.command which
    # end() requires.  Without it you get: AttributeError: 'command'

# Key flika APIs available to plugins:
## Global state
  from flika import global_vars as g
  g.win           # current Window
  g.win.image     # numpy array
  g.m             # main FlikaApplication
  g.m.windows     # list of all Windows

## Process modules (all callable objects)
  from flika.process.filters import gaussian_blur, median_filter, ...
  from flika.process.binary import threshold, binary_erosion, ...
  from flika.process.math_ import subtract, multiply, normalize, ...
  from flika.process.stacks import zproject, trim, resize, ...
  from flika.process.segmentation import connected_components, ...
  from flika.process.detection import blob_detection_log, peak_local_max, ...
  from flika.process.deconvolution import richardson_lucy, generate_psf
  from flika.process.color import split_channels, grayscale, ...

## ROIs
  from flika.roi import makeROI
  g.win.rois  # list of ROIs in the current window
  roi.getTrace()  # intensity trace as 1D array

## Window creation
  from flika.window import Window
  w = Window(numpy_array, name='My Result')

## I/O
  from flika.process.file_ import open_file, save_file

# Dimensions: 2D=(Y,X), 3D=(T,Y,X), 4D=(T,Y,X,Z)

Return ONLY valid Python code (no markdown fences).
"""


def _get_model():
    """Return the AI model from settings, with fallback."""
    try:
        import flika.global_vars as g
        return g.settings['ai_model'] or 'claude-sonnet-4-20250514'
    except Exception:
        return 'claude-sonnet-4-20250514'


def _get_source_url():
    """Return the flika source URL from settings."""
    try:
        import flika.global_vars as g
        return (g.settings['flika_source_url']
                or 'https://github.com/gddickinson/flika---Georges-Edition')
    except Exception:
        return 'https://github.com/gddickinson/flika---Georges-Edition'


class PluginGenerator:
    """Generate a flika plugin from a description."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for AI features. "
                "Install with:  pip install anthropic  (or  pip install flika[ai])"
            )
        if api_key is None:
            from flika.app.settings_editor import get_api_key
            api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "No API key found.  Set ANTHROPIC_API_KEY environment variable "
                "or enter your key in Edit > Settings."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or _get_model()

    def generate_plugin(self, description: str, name: str = "generated_plugin") -> str:
        """Generate plugin code, validate syntax and structure."""
        logger.info("AI plugin generator: creating %r from %r", name, description)
        prompt = f"Create a flika plugin named '{name}' that: {description}"

        # Build system prompt with guidelines and safety policy
        system = _SYSTEM_PROMPT.format(source_url=_get_source_url())
        guidelines = _load_guidelines()
        if guidelines:
            system += (
                "\n\n# Plugin Development Guidelines\n"
                "Follow these rules strictly:\n\n" + guidelines
            )
        try:
            from flika.ai.safety import get_policy_summary
            system += "\n\n" + get_policy_summary()
        except Exception:
            pass

        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        code = message.content[0].text

        # Strip markdown fences if the model wrapped the code
        code = re.sub(r'^```python\s*\n', '', code)
        code = re.sub(r'\n```\s*$', '', code)

        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax errors: {e}") from e

        # Validate plugin structure and attempt auto-fixes
        structure_warnings = _validate_plugin_structure(code)
        if structure_warnings:
            logger.warning("Plugin structure warnings: %s", structure_warnings)
            if any("No module-level instance" in w for w in structure_warnings):
                code = self._fix_missing_instance(code)
            if any("missing self.start()" in w for w in structure_warnings):
                code = self._fix_missing_start(code)
            # Re-validate after fixes
            structure_warnings = _validate_plugin_structure(code)

        self._structure_warnings = structure_warnings
        return code

    def _fix_missing_instance(self, code: str) -> str:
        """Append a module-level instance if one is missing."""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    # Convert CamelCase to snake_case
                    instance_name = re.sub(
                        r'(?<!^)(?=[A-Z])', '_', class_name
                    ).lower()
                    code = code.rstrip() + f"\n\n{instance_name} = {class_name}()\n"
                    logger.info("Auto-added module-level instance: %s = %s()",
                                instance_name, class_name)
                    return code
        except Exception:
            pass
        return code

    def _fix_missing_start(self, code: str) -> str:
        """Insert self.start() at the beginning of __call__."""
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__call__':
                        # Determine if this is noPriorWindow
                        is_no_prior = 'BaseProcess_noPriorWindow' in code
                        # Find the line of the first statement in __call__
                        if item.body:
                            first_line = item.body[0].lineno
                            lines = code.split('\n')
                            # Detect indentation from first statement
                            indent = ''
                            for ch in lines[first_line - 1]:
                                if ch in ' \t':
                                    indent += ch
                                else:
                                    break
                            if is_no_prior:
                                start_line = f"{indent}self.start()"
                            else:
                                start_line = f"{indent}self.start(keepSourceWindow)"
                            lines.insert(first_line - 1, start_line)
                            code = '\n'.join(lines)
                            logger.info("Auto-inserted self.start() into __call__()")
                            return code
        except Exception:
            pass
        return code

    def save_plugin(self, code: str, name: str, description: str = "") -> str:
        """Save generated plugin to ``~/.FLIKA/plugins/<name>/``.

        Creates all files needed for proper Plugin Manager integration:
        ``<name>.py``, ``__init__.py``, ``info.xml``, and ``about.html``.
        """
        from os.path import expanduser
        import datetime

        plugin_dir = os.path.join(expanduser("~"), '.FLIKA', 'plugins', name)
        os.makedirs(plugin_dir, exist_ok=True)

        # Main plugin code
        filepath = os.path.join(plugin_dir, f'{name}.py')
        with open(filepath, 'w') as f:
            f.write(code)

        # Find the class name from the code
        class_name = name
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    break
        except Exception:
            pass

        # Find the module-level instance name (e.g., "test_plugin = TestPlugin()")
        instance_name = None
        try:
            tree = ast.parse(code)
            # Only look at top-level statements, not nested assignments
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    # Check if the value is a call to the class we found
                    if (isinstance(node.value, ast.Call) and
                            isinstance(node.value.func, ast.Name) and
                            node.value.func.id == class_name):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                instance_name = target.id
                                break
                    if instance_name:
                        break
        except Exception:
            pass
        if not instance_name:
            instance_name = name

        # __init__.py
        init_path = os.path.join(plugin_dir, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write(f'from .{name} import *\n')

        # info.xml
        display_name = name.replace("_", " ").title()
        today = datetime.date.today().strftime("%m/%d/%Y")
        info_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<plugin name="{display_name}">
    <directory>{name}</directory>
    <version>1.0.0</version>
    <author>AI Generated</author>
    <url></url>
    <documentation></documentation>
    <date>{today}</date>
    <menu_layout>
        <action name="{display_name}" location="{name}" function="{instance_name}.gui">{display_name}</action>
    </menu_layout>
</plugin>
'''
        info_path = os.path.join(plugin_dir, 'info.xml')
        with open(info_path, 'w') as f:
            f.write(info_xml)

        # about.html
        about_html = f'''<html>
<body>
<h2>{display_name}</h2>
<p>AI-generated flika plugin.</p>
<p>{description or "No description provided."}</p>
<p><i>Generated by flika AI Plugin Generator</i></p>
</body>
</html>
'''
        about_path = os.path.join(plugin_dir, 'about.html')
        with open(about_path, 'w') as f:
            f.write(about_html)

        logger.info("Plugin saved to %s (with info.xml, __init__.py, about.html)",
                     filepath)
        return filepath


def _show_generate_plugin_dialog():
    """Menu callback for AI > Generate Plugin."""
    from flika.ai.plugin_dialog import PluginGeneratorDialog
    PluginGeneratorDialog.show_dialog()
