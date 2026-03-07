# -*- coding: utf-8 -*-
"""Dependency checker dialogs for flika's Help menu.

Provides two dialogs:
- CoreDependencyDialog: checks packages listed in pyproject.toml
- PluginDependencyDialog: scans plugin info.xml files for <dependency> tags

Core and optional dependencies are read directly from pyproject.toml at
runtime so they never fall out of sync with the actual package metadata.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
from pathlib import Path
from xml.etree import ElementTree

from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version, InvalidVersion
from qtpy import QtCore, QtWidgets


# ---------------------------------------------------------------------------
# Locate and parse pyproject.toml
# ---------------------------------------------------------------------------

def _find_pyproject() -> str | None:
    """Find pyproject.toml by walking up from the flika package directory."""
    pkg_dir = Path(__file__).resolve().parent.parent  # flika/flika/
    for candidate in [pkg_dir / 'pyproject.toml',
                      pkg_dir.parent / 'pyproject.toml']:
        if candidate.is_file():
            return str(candidate)
    return None


def _parse_pyproject(path: str) -> dict:
    """Minimal TOML parser for pyproject.toml dependency fields.

    Tries tomllib (3.11+), then tomli, then a basic regex fallback.
    """
    text = Path(path).read_text(encoding='utf-8')

    # Try stdlib tomllib (Python 3.11+)
    try:
        import tomllib
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except ImportError:
        pass

    # Try tomli
    try:
        import tomli
        with open(path, 'rb') as f:
            return tomli.load(f)
    except ImportError:
        pass

    # Regex fallback — extract dependencies and optional-dependencies
    result = {'project': {'dependencies': [], 'optional-dependencies': {}}}

    # Main dependencies
    m = re.search(r'\[project\].*?dependencies\s*=\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        deps = re.findall(r'"([^"]+)"', m.group(1))
        result['project']['dependencies'] = deps

    # Optional dependencies
    for m in re.finditer(
            r'\[project\.optional-dependencies\]\s*\n(.*?)(?=\n\[|\Z)', text, re.DOTALL):
        block = m.group(1)
        for line_match in re.finditer(
                r'^(\w[\w-]*)\s*=\s*\[(.*?)\]', block, re.MULTILINE | re.DOTALL):
            group_name = line_match.group(1)
            deps = re.findall(r'"([^"]+)"', line_match.group(2))
            result['project']['optional-dependencies'][group_name] = deps

    return result


def _get_dependencies() -> tuple[list[str], dict[str, list[str]]]:
    """Return (core_deps, optional_deps) from pyproject.toml.

    Falls back to hardcoded list if pyproject.toml cannot be found/parsed.
    """
    pyproject_path = _find_pyproject()
    if pyproject_path:
        try:
            data = _parse_pyproject(pyproject_path)
            project = data.get('project', {})
            core = project.get('dependencies', [])
            optional = project.get('optional-dependencies', {})
            if core:
                return core, optional
        except Exception:
            pass

    # Fallback — should not normally be needed
    return _FALLBACK_CORE_DEPS, _FALLBACK_OPTIONAL_DEPS


# Fallback in case pyproject.toml can't be found (e.g. installed as wheel)
_FALLBACK_CORE_DEPS = [
    "numpy>=1.24", "scipy>=1.10", "pandas>=1.5", "matplotlib>=3.6",
    "pyqtgraph>=0.13", "PyQt6", "qtpy>=2.3", "setuptools>=1.0",
    "scikit-image>=0.20", "scikit-learn>=1.2", "ipython>=8.0",
    "ipykernel", "qtconsole", "pyopengl", "requests", "nd2reader",
    "markdown", "packaging", "tifffile>=2022.5.4",
]

_FALLBACK_OPTIONAL_DEPS = {
    'ai': ["anthropic>=0.18"],
    'gpu': ["cupy"],
    'accel': ["numba>=0.57", "torch>=2.0"],
    'classifier': ["torch>=2.0"],
    'all-formats': ["h5py", "zarr", "ome-zarr>=0.9", "aicsimageio"],
    'lazy': ["dask[array]>=2022.1"],
    'segmentation': ["cellpose", "stardist", "csbdeep", "micro_sam"],
    'model-zoo': ["bioimageio.core>=0.6", "bioimageio.spec>=0.5"],
    'denoising': ["careamics>=0.1"],
    'detection': ["ultralytics>=8.0"],
    'camera': ["opencv-python>=4.5"],
    'camera-micro-manager': ["pymmcore-plus>=0.7"],
    'spt': ["trackpy>=0.6"],
}


# Import name overrides (package name -> importable module name)
_IMPORT_NAMES = {
    "PyQt6": "PyQt6",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "ipython": "IPython",
    "pyopengl": "OpenGL",
    "nd2reader": "nd2reader",
    "micro_sam": "micro_sam",
    "bioimageio.core": "bioimageio.core",
    "bioimageio.spec": "bioimageio.spec",
    "ome-zarr": "ome_zarr",
    "dask[array]": "dask",
    "aicsimageio": "aicsimageio",
    "careamics": "careamics",
}

# Metadata name overrides (some packages use different distribution names)
_DIST_NAMES = {
    "pyopengl": "PyOpenGL",
    "ipython": "ipython",
    "dask[array]": "dask",
}


# ---------------------------------------------------------------------------
# Checking logic
# ---------------------------------------------------------------------------

def _parse_requirement(dep_str: str) -> tuple[str, str | None]:
    """Parse a PEP 508 dependency string into (name, min_version_or_None)."""
    # Strip extras like dask[array]
    clean = re.sub(r'\[.*?\]', '', dep_str).strip()
    try:
        req = Requirement(clean)
        name = req.name
        min_ver = None
        for spec in req.specifier:
            if spec.operator in ('>=', '==', '~='):
                min_ver = str(spec.version)
                break
        return name, min_ver
    except InvalidRequirement:
        # Fallback: split on >= or ==
        m = re.match(r'^([a-zA-Z0-9_.-]+)\s*(?:>=|==|~=)\s*(.+)$', clean)
        if m:
            return m.group(1), m.group(2)
        return clean, None


def _check_package(pkg_name: str, min_version: str | None = None,
                   dep_str: str = '') -> tuple[str, str, str]:
    """Check a single package. Returns (status, installed_ver, message)."""
    # Use the original dep_str for extras handling
    raw_name = dep_str or pkg_name
    dist_name = _DIST_NAMES.get(raw_name, pkg_name)

    # Try metadata lookup
    installed = None
    for try_name in [dist_name, pkg_name, pkg_name.replace('-', '_'),
                     pkg_name.replace('_', '-')]:
        try:
            installed = importlib.metadata.version(try_name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue

    if installed is None:
        # Last resort: check if importable
        import_name = _IMPORT_NAMES.get(raw_name,
                                        pkg_name.replace("-", "_").lower())
        if importlib.util.find_spec(import_name) is not None:
            return ("OK (unknown ver)", "?", "")
        return ("MISSING", "-", f"pip install {raw_name}")

    if min_version is None:
        return ("OK", installed, "")

    try:
        if Version(installed) < Version(min_version):
            return ("OUTDATED", installed,
                    f"pip install --upgrade {raw_name}>={min_version}")
    except InvalidVersion:
        return ("OK (unparsable)", installed, "")

    return ("OK", installed, "")


# ---------------------------------------------------------------------------
# Core Dependency Dialog
# ---------------------------------------------------------------------------

class CoreDependencyDialog(QtWidgets.QDialog):
    """Dialog showing status of core and optional flika dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dependency Check")
        self.resize(750, 600)
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
        core_deps, optional_deps = _get_dependencies()

        lines = []
        all_missing = []  # collect pip install specs for all missing/outdated

        lines.append("=== CORE DEPENDENCIES ===")
        lines.append("")
        lines.append(
            f"{'Package':<22} {'Required':<14} {'Installed':<16} "
            f"{'Status':<18} {'Action'}")
        lines.append("-" * 95)

        issues = 0
        for dep_str in core_deps:
            pkg_name, min_ver = _parse_requirement(dep_str)
            status, installed, action = _check_package(
                pkg_name, min_ver, dep_str)
            min_str = f">={min_ver}" if min_ver else "-"
            lines.append(
                f"{pkg_name:<22} {min_str:<14} {installed:<16} "
                f"{status:<18} {action}")
            if "MISSING" in status or "OUTDATED" in status:
                issues += 1
                all_missing.append(dep_str)

        lines.append("")
        if issues == 0:
            lines.append("All core dependencies are satisfied.")
        else:
            lines.append(
                f"{issues} core issue(s) found.")

        # Optional dependency groups
        if optional_deps:
            lines.append("")
            lines.append("")
            lines.append("=== OPTIONAL DEPENDENCIES ===")
            lines.append("")
            lines.append(
                "These enable extra features. Install a group with: "
                "pip install flika[group_name]")
            lines.append("")

            opt_issues = 0
            for group_name, deps in sorted(optional_deps.items()):
                lines.append(f"--- [{group_name}] ---")
                group_ok = True
                for dep_str in deps:
                    pkg_name, min_ver = _parse_requirement(dep_str)
                    status, installed, action = _check_package(
                        pkg_name, min_ver, dep_str)
                    min_str = f">={min_ver}" if min_ver else "-"
                    marker = " " if "OK" in status else "!"
                    lines.append(
                        f"  {marker} {pkg_name:<20} {min_str:<14} "
                        f"{installed:<16} {status}")
                    if "MISSING" in status or "OUTDATED" in status:
                        group_ok = False
                        opt_issues += 1
                        all_missing.append(dep_str)
                if group_ok:
                    lines.append("  All satisfied.")
                lines.append("")

            if opt_issues == 0:
                lines.append("All installed optional dependencies are satisfied.")
            else:
                lines.append(
                    f"{opt_issues} optional package(s) not installed. "
                    "Install groups as needed.")

        # Combined install command
        if all_missing:
            lines.append("")
            lines.append("")
            lines.append("=== INSTALL ALL MISSING ===")
            lines.append("")
            lines.append("Run this command to install everything at once:")
            lines.append("")
            lines.append("pip install " + " ".join(
                f'"{d}"' for d in all_missing))
            lines.append("")

        self._text.setPlainText("\n".join(lines))


# ---------------------------------------------------------------------------
# Plugin Dependency Dialog (unchanged logic)
# ---------------------------------------------------------------------------

def _scan_plugin_dependencies():
    """Scan ~/.FLIKA/plugins/*/info.xml for <dependency> tags.

    Returns dict: {plugin_name: [dep_name, ...]}
    """
    plugins_dir = os.path.join(os.path.expanduser("~"), ".FLIKA", "plugins")
    if not os.path.isdir(plugins_dir):
        return {}

    skip_dirs = {"NOT_WORKING", "NOT_WORKING_WITH_PY311", "OLD",
                 "EXPERIMENTAL", "backups", "__pycache__"}
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
    """Check if a dependency is importable. Returns (status, ver, action)."""
    base = re.split(r"[><=!]", dep_name)[0].strip()
    import_name = base.replace("-", "_").lower()

    try:
        ver = importlib.metadata.version(base)
        return ("OK", ver, "")
    except importlib.metadata.PackageNotFoundError:
        pass

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
        all_missing = []
        for plugin_name, deps in plugin_deps.items():
            lines.append(f"=== {plugin_name} ===")
            for dep in deps:
                status, ver, action = _check_importable(dep)
                lines.append(
                    f"  {dep:<30} {ver:<12} {status:<12} {action}")
                if "MISSING" in status:
                    total_issues += 1
                    all_missing.append(dep)
            lines.append("")

        if total_issues == 0:
            lines.append("All plugin dependencies are satisfied.")
        else:
            lines.append(
                f"{total_issues} missing plugin dependency(ies).")
            lines.append("")
            lines.append("Run this command to install everything at once:")
            lines.append("")
            lines.append("pip install " + " ".join(
                f'"{d}"' for d in all_missing))
            lines.append("")

        self._text.setPlainText("\n".join(lines))
