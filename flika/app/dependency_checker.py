# -*- coding: utf-8 -*-
"""Dependency checker dialogs for flika's Help menu.

Provides two dialogs:
- CoreDependencyDialog: checks packages listed in pyproject.toml
- PluginDependencyDialog: scans plugin info.xml files for <dependency> tags
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
from xml.etree import ElementTree

from packaging.version import Version, InvalidVersion
from qtpy import QtCore, QtWidgets


# Core dependencies from pyproject.toml (package_name, min_version_or_None)
_CORE_DEPS = [
    ("numpy", "1.20"),
    ("scipy", "1.6"),
    ("pandas", "0.14"),
    ("matplotlib", "1.4"),
    ("pyqtgraph", "0.12"),
    ("PyQt6", None),
    ("qtpy", "1.1"),
    ("setuptools", "1.0"),
    ("scikit-image", "0.18"),
    ("scikit-learn", None),
    ("ipython", "7.0"),
    ("ipykernel", None),
    ("qtconsole", None),
    ("pyopengl", None),
    ("requests", None),
    ("nd2reader", None),
    ("markdown", None),
    ("packaging", None),
    ("tifffile", "2021.7.2"),
]

# Import name overrides (package name -> importable module name)
_IMPORT_NAMES = {
    "PyQt6": "PyQt6",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "ipython": "IPython",
    "pyopengl": "OpenGL",
    "nd2reader": "nd2reader",
}

# Metadata name overrides (some packages use different distribution names)
_DIST_NAMES = {
    "pyopengl": "PyOpenGL",
    "ipython": "ipython",
}


def _check_package(pkg_name, min_version=None):
    """Check a single package. Returns (status, installed_ver, message)."""
    dist_name = _DIST_NAMES.get(pkg_name, pkg_name)
    try:
        installed = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        # Try the raw name
        try:
            installed = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Last resort: check if importable
            import_name = _IMPORT_NAMES.get(pkg_name, pkg_name.replace("-", "_").lower())
            if importlib.util.find_spec(import_name) is not None:
                return ("OK (unknown ver)", "?", "")
            return ("MISSING", "-", f"pip install {pkg_name}")

    if min_version is None:
        return ("OK", installed, "")

    try:
        if Version(installed) < Version(min_version):
            return ("OUTDATED", installed,
                    f"pip install --upgrade {pkg_name}>={min_version}")
    except InvalidVersion:
        return ("OK (unparsable)", installed, "")

    return ("OK", installed, "")


class CoreDependencyDialog(QtWidgets.QDialog):
    """Dialog showing status of core flika dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Core Dependency Check")
        self.resize(650, 500)
        layout = QtWidgets.QVBoxLayout(self)

        self._text = QtWidgets.QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFontFamily("monospace")
        layout.addWidget(self._text)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

        self._run_check()

    def _run_check(self):
        lines = []
        lines.append(f"{'Package':<20} {'Min Ver':<12} {'Installed':<16} {'Status':<18} {'Action'}")
        lines.append("-" * 90)
        issues = 0
        for pkg, minv in _CORE_DEPS:
            status, installed, action = _check_package(pkg, minv)
            minv_str = minv if minv else "-"
            lines.append(f"{pkg:<20} {minv_str:<12} {installed:<16} {status:<18} {action}")
            if "MISSING" in status or "OUTDATED" in status:
                issues += 1

        lines.append("")
        if issues == 0:
            lines.append("All core dependencies are satisfied.")
        else:
            lines.append(f"{issues} issue(s) found. Copy the pip commands above to fix.")

        self._text.setPlainText("\n".join(lines))


def _scan_plugin_dependencies():
    """Scan ~/.FLIKA/plugins/*/info.xml for <dependency> tags.

    Returns dict: {plugin_name: [dep_name, ...]}
    """
    plugins_dir = os.path.join(os.path.expanduser("~"), ".FLIKA", "plugins")
    if not os.path.isdir(plugins_dir):
        return {}

    skip_dirs = {"NOT_WORKING", "NOT_WORKING_WITH_PY311", "OLD", "EXPERIMENTAL", "backups", "__pycache__"}
    results = {}

    for name in sorted(os.listdir(plugins_dir)):
        if name in skip_dirs or name.startswith("."):
            continue
        info_path = os.path.join(plugins_dir, name, "info.xml")
        if not os.path.isfile(info_path):
            continue
        try:
            tree = ElementTree.parse(info_path)
            root = tree.getroot()
            deps = []
            for dep in root.iter("dependency"):
                # Support both <dependency name='pkg'/> and <dependency>pkg</dependency>
                dep_name = dep.get("name", "").strip()
                if not dep_name:
                    dep_name = (dep.text or "").strip()
                if dep_name:
                    deps.append(dep_name)
            if deps:
                results[name] = deps
        except Exception:
            continue

    return results


def _check_importable(dep_name):
    """Check if a dependency is importable. Returns (status, action)."""
    # Normalize: strip version specifiers for import check
    base = re.split(r"[><=!]", dep_name)[0].strip()
    import_name = base.replace("-", "_").lower()

    # Try metadata first
    try:
        ver = importlib.metadata.version(base)
        return ("OK", ver, "")
    except importlib.metadata.PackageNotFoundError:
        pass

    # Try import
    if importlib.util.find_spec(import_name) is not None:
        return ("OK", "?", "")

    return ("MISSING", "-", f"pip install {dep_name}")


class PluginDependencyDialog(QtWidgets.QDialog):
    """Dialog showing status of plugin dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plugin Dependency Check")
        self.resize(650, 500)
        layout = QtWidgets.QVBoxLayout(self)

        self._text = QtWidgets.QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFontFamily("monospace")
        layout.addWidget(self._text)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

        self._run_check()

    def _run_check(self):
        plugin_deps = _scan_plugin_dependencies()
        lines = []

        if not plugin_deps:
            lines.append("No plugin dependencies found.")
            lines.append("")
            lines.append("(Plugins must have info.xml with <dependency> tags)")
            self._text.setPlainText("\n".join(lines))
            return

        total_issues = 0
        for plugin_name, deps in plugin_deps.items():
            lines.append(f"=== {plugin_name} ===")
            for dep in deps:
                status, ver, action = _check_importable(dep)
                lines.append(f"  {dep:<30} {ver:<12} {status:<12} {action}")
                if "MISSING" in status:
                    total_issues += 1
            lines.append("")

        if total_issues == 0:
            lines.append("All plugin dependencies are satisfied.")
        else:
            lines.append(f"{total_issues} missing plugin dependency(ies). "
                         "Copy the pip commands above to fix.")

        self._text.setPlainText("\n".join(lines))
