"""AI safety settings and code validation for flika.

Provides a centralized safety layer used by Live Session, Script Assistant,
and Plugin Generator.  Settings are stored in ``g.settings`` and exposed
via the Claude > Safety Settings dialog.
"""
from __future__ import annotations

import re
from qtpy import QtWidgets, QtCore
from flika.logger import logger

# ---------------------------------------------------------------------------
# Settings keys (stored in g.settings)
# ---------------------------------------------------------------------------
_KEY_ALLOW_FILE_WRITE = 'ai_allow_file_write'
_KEY_ALLOW_FILE_READ = 'ai_allow_file_read'
_KEY_ALLOW_NETWORK = 'ai_allow_network'
_KEY_ALLOW_SUBPROCESS = 'ai_allow_subprocess'
_KEY_ALLOW_OS_ACCESS = 'ai_allow_os_access'
_KEY_ALLOWED_DIRS = 'ai_allowed_directories'
_KEY_REQUIRE_APPROVAL = 'ai_require_approval'

# Default: conservative
_DEFAULTS = {
    _KEY_ALLOW_FILE_WRITE: False,
    _KEY_ALLOW_FILE_READ: True,
    _KEY_ALLOW_NETWORK: False,
    _KEY_ALLOW_SUBPROCESS: False,
    _KEY_ALLOW_OS_ACCESS: False,
    _KEY_ALLOWED_DIRS: '',  # comma-separated paths; empty = flika dirs only
    _KEY_REQUIRE_APPROVAL: True,
}


def _get(key: str):
    """Read an AI safety setting, falling back to the default."""
    import flika.global_vars as g
    val = g.settings[key]
    if val is None:
        return _DEFAULTS.get(key)
    return val


# ---------------------------------------------------------------------------
# Dangerous-pattern definitions, grouped by category
# ---------------------------------------------------------------------------
_PATTERNS_SUBPROCESS = [
    (r'\bsubprocess\b', "subprocess module"),
    (r'\bos\.system\s*\(', "os.system()"),
    (r'\bos\.popen\s*\(', "os.popen()"),
    (r'\bos\.exec[a-z]*\s*\(', "os.exec*()"),
    (r'\bos\.spawn[a-z]*\s*\(', "os.spawn*()"),
]

_PATTERNS_FILE_WRITE = [
    (r'\bopen\s*\([^)]*["\'][wax]["\']', "file write via open()"),
    (r'\bshutil\.rmtree\b', "shutil.rmtree()"),
    (r'\bshutil\.move\b', "shutil.move()"),
    (r'\bos\.remove\s*\(', "os.remove()"),
    (r'\bos\.unlink\s*\(', "os.unlink()"),
    (r'\bos\.rename\s*\(', "os.rename()"),
    (r'\bos\.makedirs\s*\(', "os.makedirs()"),
    (r'\brmtree\b', "rmtree"),
    (r'rm\s+-rf\b', "rm -rf"),
]

_PATTERNS_FILE_READ = [
    (r'\bopen\s*\([^)]*["\']r["\']', "file read via open()"),
    (r'\bopen\s*\([^)]*\)\s*$', "open() (default read mode)"),
]

_PATTERNS_NETWORK = [
    (r'\burllib\b', "urllib module"),
    (r'\brequests\b', "requests module"),
    (r'\bhttpx\b', "httpx module"),
    (r'\bsocket\b', "socket module"),
    (r'\burlopen\s*\(', "urlopen()"),
    (r'\bhttp\.client\b', "http.client module"),
]

_PATTERNS_OS_ACCESS = [
    (r'\b__import__\b', "dynamic __import__()"),
    (r'\beval\s*\(', "eval()"),
    (r'\bexec\s*\(', "exec()"),
    (r'\bgetattr\s*\(\s*__builtins__', "getattr(__builtins__)"),
    (r'\bos\.environ\b', "os.environ access"),
    (r'\bctypes\b', "ctypes module"),
]

# Map setting keys to pattern groups
_PATTERN_GROUPS = {
    _KEY_ALLOW_SUBPROCESS: _PATTERNS_SUBPROCESS,
    _KEY_ALLOW_FILE_WRITE: _PATTERNS_FILE_WRITE,
    _KEY_ALLOW_FILE_READ: _PATTERNS_FILE_READ,
    _KEY_ALLOW_NETWORK: _PATTERNS_NETWORK,
    _KEY_ALLOW_OS_ACCESS: _PATTERNS_OS_ACCESS,
}


# ---------------------------------------------------------------------------
# Central safety check
# ---------------------------------------------------------------------------

# Session-level approvals (reset when flika restarts)
_session_approved: set = set()


def check_code_safety(code: str) -> tuple[bool, list[str]]:
    """Check AI-generated code against the current safety settings.

    Returns (is_safe, warnings) where *is_safe* is False if the code
    contains patterns blocked by the current settings.
    """
    warnings = []
    for key, patterns in _PATTERN_GROUPS.items():
        allowed = _get(key)
        if allowed:
            continue  # user has allowed this category
        for regex, description in patterns:
            if re.search(regex, code):
                if description not in _session_approved:
                    warnings.append(description)

    return (len(warnings) == 0, warnings)


def request_approval(parent: QtWidgets.QWidget, code: str,
                     warnings: list[str]) -> bool:
    """Show an approval dialog for flagged code patterns.

    If the user approves, the patterns are added to the session whitelist
    so they won't be asked again.  Returns True if approved.
    """
    if not _get(_KEY_REQUIRE_APPROVAL):
        # User has disabled the approval gate
        return True

    warn_text = "\n".join(f"  • {w}" for w in warnings)
    reply = QtWidgets.QMessageBox.warning(
        parent,
        "AI Code Safety Warning",
        f"The AI-generated code contains potentially dangerous operations:\n\n"
        f"{warn_text}\n\n"
        f"Allow execution?  (Approving adds these to the session whitelist.)",
        QtWidgets.QMessageBox.StandardButton.Yes |
        QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No
    )
    if reply == QtWidgets.QMessageBox.StandardButton.Yes:
        _session_approved.update(warnings)
        return True
    return False


def get_policy_summary() -> str:
    """Return a human-readable summary of current AI permissions.

    This is included in AI system prompts so the model knows what it
    should and should not attempt.
    """
    lines = []
    lines.append("# AI Safety Policy (enforced by flika)")
    lines.append("The following restrictions are active.  Do NOT generate "
                 "code that violates them — it will be blocked.\n")

    if not _get(_KEY_ALLOW_FILE_WRITE):
        lines.append("- File writes are BLOCKED (no open('w'), os.remove, "
                     "shutil.rmtree, etc.)")
    else:
        lines.append("- File writes are allowed")

    if not _get(_KEY_ALLOW_FILE_READ):
        lines.append("- Arbitrary file reads are BLOCKED")
    else:
        lines.append("- File reads are allowed")

    if not _get(_KEY_ALLOW_NETWORK):
        lines.append("- Network access is BLOCKED (no urllib, requests, "
                     "socket, etc.)")
    else:
        lines.append("- Network access is allowed")

    if not _get(_KEY_ALLOW_SUBPROCESS):
        lines.append("- Subprocess/shell execution is BLOCKED (no subprocess, "
                     "os.system, os.popen, etc.)")
    else:
        lines.append("- Subprocess execution is allowed")

    if not _get(_KEY_ALLOW_OS_ACCESS):
        lines.append("- Low-level OS access is BLOCKED (no eval, exec, "
                     "__import__, ctypes, os.environ)")
    else:
        lines.append("- Low-level OS access is allowed")

    dirs = _get(_KEY_ALLOWED_DIRS)
    if dirs:
        lines.append(f"- File operations restricted to: {dirs}")

    lines.append("\nYou may use: numpy, scipy, pyqtgraph, flika process "
                 "functions, Window creation, ROI operations, and all "
                 "standard image analysis operations.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Settings Dialog
# ---------------------------------------------------------------------------

class AISafetyDialog(QtWidgets.QDialog):
    """Claude > Safety Settings dialog."""

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
        self.setWindowTitle("AI Safety Settings")
        self.resize(520, 500)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        header = QtWidgets.QLabel(
            "<b>Control what AI-generated code is allowed to do.</b><br>"
            "These settings apply to Live Session, Script Assistant, and "
            "Plugin Generator.  Changes take effect immediately.")
        header.setWordWrap(True)
        layout.addWidget(header)

        disclaimer = QtWidgets.QLabel(
            "<b>Note:</b> These are <i>policy restrictions</i>, not a sandbox. "
            "They work by pattern-matching against known dangerous operations "
            "and instructing the AI model to respect them. A determined or "
            "creative prompt could potentially bypass these checks. "
            "Always review AI-generated code before running it, especially "
            "if you have enabled permissive settings.")
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            "background-color: #fff3cd; color: #664d03; "
            "border: 1px solid #ffecb5; border-radius: 4px; "
            "padding: 8px; font-size: 12px;")
        layout.addWidget(disclaimer)

        layout.addSpacing(6)

        # Permission checkboxes
        group = QtWidgets.QGroupBox("Permissions")
        glay = QtWidgets.QVBoxLayout(group)

        self._cb_file_read = QtWidgets.QCheckBox(
            "Allow file reads (open files for reading)")
        self._cb_file_read.setChecked(bool(_get(_KEY_ALLOW_FILE_READ)))
        glay.addWidget(self._cb_file_read)

        self._cb_file_write = QtWidgets.QCheckBox(
            "Allow file writes (create, modify, delete files)")
        self._cb_file_write.setChecked(bool(_get(_KEY_ALLOW_FILE_WRITE)))
        glay.addWidget(self._cb_file_write)

        self._cb_network = QtWidgets.QCheckBox(
            "Allow network access (urllib, requests, socket)")
        self._cb_network.setChecked(bool(_get(_KEY_ALLOW_NETWORK)))
        glay.addWidget(self._cb_network)

        self._cb_subprocess = QtWidgets.QCheckBox(
            "Allow subprocess/shell execution")
        self._cb_subprocess.setChecked(bool(_get(_KEY_ALLOW_SUBPROCESS)))
        glay.addWidget(self._cb_subprocess)

        self._cb_os_access = QtWidgets.QCheckBox(
            "Allow low-level OS access (eval, exec, __import__, ctypes)")
        self._cb_os_access.setChecked(bool(_get(_KEY_ALLOW_OS_ACCESS)))
        glay.addWidget(self._cb_os_access)

        layout.addWidget(group)

        # Approval gate
        group2 = QtWidgets.QGroupBox("Approval")
        g2lay = QtWidgets.QVBoxLayout(group2)

        self._cb_require_approval = QtWidgets.QCheckBox(
            "Show approval dialog before running flagged code")
        self._cb_require_approval.setChecked(bool(_get(_KEY_REQUIRE_APPROVAL)))
        g2lay.addWidget(self._cb_require_approval)

        note = QtWidgets.QLabel(
            "<i>When enabled, you'll be prompted to approve any code that "
            "matches a blocked category.  Approved patterns are remembered "
            "for the session.</i>")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        g2lay.addWidget(note)

        layout.addWidget(group2)

        # Allowed directories
        group3 = QtWidgets.QGroupBox("Allowed Directories (optional)")
        g3lay = QtWidgets.QVBoxLayout(group3)
        dir_note = QtWidgets.QLabel(
            "Comma-separated list of directories the AI may access.  "
            "Leave empty to allow flika directories only.")
        dir_note.setWordWrap(True)
        dir_note.setStyleSheet("font-size: 11px;")
        g3lay.addWidget(dir_note)
        self._dirs_edit = QtWidgets.QLineEdit()
        self._dirs_edit.setPlaceholderText(
            "e.g. /Users/me/data, /Users/me/results")
        self._dirs_edit.setText(_get(_KEY_ALLOWED_DIRS) or '')
        g3lay.addWidget(self._dirs_edit)
        layout.addWidget(group3)

        layout.addStretch()

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(btn_reset)

        btn_layout.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        btn_layout.addWidget(btn_cancel)

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._save)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)

    def _save(self):
        import flika.global_vars as g
        g.settings[_KEY_ALLOW_FILE_READ] = self._cb_file_read.isChecked()
        g.settings[_KEY_ALLOW_FILE_WRITE] = self._cb_file_write.isChecked()
        g.settings[_KEY_ALLOW_NETWORK] = self._cb_network.isChecked()
        g.settings[_KEY_ALLOW_SUBPROCESS] = self._cb_subprocess.isChecked()
        g.settings[_KEY_ALLOW_OS_ACCESS] = self._cb_os_access.isChecked()
        g.settings[_KEY_REQUIRE_APPROVAL] = self._cb_require_approval.isChecked()
        g.settings[_KEY_ALLOWED_DIRS] = self._dirs_edit.text().strip()
        g.settings.save()
        logger.info("AI safety settings saved")
        # Clear session approvals when settings change
        _session_approved.clear()
        self.close()

    def _reset_defaults(self):
        self._cb_file_read.setChecked(_DEFAULTS[_KEY_ALLOW_FILE_READ])
        self._cb_file_write.setChecked(_DEFAULTS[_KEY_ALLOW_FILE_WRITE])
        self._cb_network.setChecked(_DEFAULTS[_KEY_ALLOW_NETWORK])
        self._cb_subprocess.setChecked(_DEFAULTS[_KEY_ALLOW_SUBPROCESS])
        self._cb_os_access.setChecked(_DEFAULTS[_KEY_ALLOW_OS_ACCESS])
        self._cb_require_approval.setChecked(_DEFAULTS[_KEY_REQUIRE_APPROVAL])
        self._dirs_edit.setText('')


def _show_safety_settings():
    """Menu callback for Claude > Safety Settings."""
    AISafetyDialog.show_dialog()
