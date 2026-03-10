"""Workflow template manager for flika.

Scans bundled and user template directories, parses header metadata,
and populates the Scripts > Templates submenu.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from qtpy import QtWidgets

from flika.logger import logger
import flika.global_vars as g


# Template directories
_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
_USER_DIR = os.path.join(os.path.expanduser('~'), '.FLIKA', 'templates')


class TemplateInfo:
    """Parsed template metadata."""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.description = ''
        self.category = 'General'
        self._parse_header()

    def _parse_header(self):
        try:
            with open(self.path, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        break
                    m = re.match(r'#\s*Template:\s*(.+)', line)
                    if m:
                        self.name = m.group(1).strip()
                    m = re.match(r'#\s*Description:\s*(.+)', line)
                    if m:
                        self.description = m.group(1).strip()
                    m = re.match(r'#\s*Category:\s*(.+)', line)
                    if m:
                        self.category = m.group(1).strip()
        except (OSError, IOError):
            pass


class TemplateManager:
    """Scans template directories and builds menu entries."""

    def __init__(self):
        self._templates: List[TemplateInfo] = []

    def scan(self) -> List[TemplateInfo]:
        """Scan bundled and user template directories."""
        self._templates.clear()
        for d in (_BUNDLED_DIR, _USER_DIR):
            if os.path.isdir(d):
                for fname in sorted(os.listdir(d)):
                    if fname.endswith('.py') and not fname.startswith('_'):
                        path = os.path.join(d, fname)
                        self._templates.append(TemplateInfo(path))
        return self._templates

    def get_by_category(self) -> Dict[str, List[TemplateInfo]]:
        """Return templates grouped by category."""
        cats: Dict[str, List[TemplateInfo]] = {}
        for t in self._templates:
            cats.setdefault(t.category, []).append(t)
        return cats

    def populate_menu(self, parent_menu: QtWidgets.QMenu):
        """Add Templates submenu to *parent_menu*."""
        self.scan()
        templates_menu = parent_menu.addMenu('Templates')

        if not self._templates:
            no_action = QtWidgets.QAction('No Templates Found', templates_menu)
            no_action.setEnabled(False)
            templates_menu.addAction(no_action)
            return templates_menu

        by_cat = self.get_by_category()
        for cat_name in sorted(by_cat.keys()):
            if len(by_cat) > 1:
                cat_menu = templates_menu.addMenu(cat_name)
            else:
                cat_menu = templates_menu

            for tmpl in by_cat[cat_name]:
                action = QtWidgets.QAction(tmpl.name, cat_menu)
                if tmpl.description:
                    action.setToolTip(tmpl.description)
                action.triggered.connect(lambda checked, p=tmpl.path: self._open_template(p))
                cat_menu.addAction(action)

        return templates_menu

    def _open_template(self, path: str):
        """Open template in the Script Editor."""
        from flika.app.script_editor import ScriptEditor
        try:
            ScriptEditor.importScript(path)
            g.m.statusBar().showMessage(f'Template loaded: {os.path.basename(path)}')
        except Exception as e:
            g.alert(f'Error loading template: {e}')


# Singleton
template_manager = TemplateManager()
