"""Rich AI Assistant dialog for flika."""
from __future__ import annotations

import traceback
from qtpy import QtWidgets, QtCore, QtGui
from ..logger import logger


_INSTRUCTIONS = """\
<h3>AI Assistant — How to Use</h3>
<p>The AI Assistant generates Python scripts for flika based on your
natural-language descriptions. It has full knowledge of flika's API.</p>

<h4>What you can ask:</h4>
<ul>
<li><b>Image processing:</b> "Apply a Gaussian blur with sigma 3 to the current image"</li>
<li><b>Analysis workflows:</b> "Threshold the image, find connected components, and measure their areas"</li>
<li><b>File operations:</b> "Open all TIFF files in a folder and create a max projection of each"</li>
<li><b>Visualization:</b> "Create an RGB overlay of the first three channels"</li>
<li><b>ROI operations:</b> "Draw a rectangle ROI and extract the intensity trace"</li>
<li><b>Complex pipelines:</b> "Detect particles, link them into tracks, and calculate diffusion coefficients"</li>
<li><b>Plugin creation:</b> Use AI > Claude > Generate Plugin for reusable tools</li>
</ul>

<h4>Tips:</h4>
<ul>
<li>Be specific about parameters (sigma values, threshold levels, etc.)</li>
<li>The assistant knows about the current window's dimensions and data type</li>
<li>You can have a multi-turn conversation to refine scripts</li>
<li>Review generated code before running — click "Insert" to edit first</li>
<li>Scripts run in the Script Editor's IPython namespace with numpy, scipy, pyqtgraph pre-imported</li>
</ul>
"""


class _AssistantWorker(QtCore.QThread):
    """Background thread for AI API calls."""
    sig_result = QtCore.Signal(str)
    sig_error = QtCore.Signal(str)

    def __init__(self, assistant, messages):
        super().__init__()
        self.assistant = assistant
        self.messages = messages

    def run(self):
        try:
            result = self.assistant.generate_with_history(self.messages)
            self.sig_result.emit(result)
        except Exception as e:
            logger.error("AI assistant worker error: %s\n%s",
                         e, traceback.format_exc())
            self.sig_error.emit(f"{type(e).__name__}: {e}")


class AIAssistantDialog(QtWidgets.QDialog):
    """Multi-turn AI assistant dialog for flika scripting.

    Provides a chat-style interface for generating flika scripts via the
    Anthropic API.  The dialog is a singleton — calling ``show_dialog()``
    will re-use an existing instance so that conversation history is
    preserved across invocations.

    Features
    --------
    * Multi-turn conversation with full message history
    * Automatic context injection (current window shape, dtype, range)
    * Background API calls via ``_AssistantWorker`` (no UI blocking)
    * Insert / Run / Copy action buttons for generated code
    * Collapsible instructions panel
    * Status bar showing the active model
    """

    _instance = None

    @classmethod
    def show_dialog(cls):
        """Show the singleton dialog, creating it if necessary."""
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent=None)
        cls._instance.show()
        cls._instance.raise_()
        cls._instance.activateWindow()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Assistant")
        self.resize(800, 700)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self._messages = []  # conversation history: [{role, content}, ...]
        self._assistant = None
        self._worker = None
        self._last_code = ""

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # --- Instructions (collapsible) ---
        self._instructions_btn = QtWidgets.QPushButton("Show Instructions")
        self._instructions_btn.setCheckable(True)
        self._instructions_btn.toggled.connect(self._toggle_instructions)
        layout.addWidget(self._instructions_btn)

        self._instructions = QtWidgets.QTextBrowser()
        self._instructions.setHtml(_INSTRUCTIONS)
        self._instructions.setMaximumHeight(300)
        self._instructions.setVisible(False)
        layout.addWidget(self._instructions)

        # --- Chat history ---
        self._chat_display = QtWidgets.QTextBrowser()
        self._chat_display.setOpenExternalLinks(True)
        self._chat_display.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Courier New', Courier, monospace;
                font-size: 13px;
                padding: 8px;
            }
        """)
        layout.addWidget(self._chat_display, 1)

        # --- Context info label ---
        self._context_label = QtWidgets.QLabel()
        self._context_label.setStyleSheet(
            "color: gray; font-size: 11px; padding: 2px;")
        layout.addWidget(self._context_label)
        self._update_context_label()

        # --- Input area ---
        input_layout = QtWidgets.QHBoxLayout()

        self._input = QtWidgets.QPlainTextEdit()
        self._input.setMaximumHeight(100)
        self._input.setPlaceholderText(
            "Describe what you want to do... (Ctrl+Enter to send)")
        input_layout.addWidget(self._input, 1)

        self._send_btn = QtWidgets.QPushButton("Send")
        self._send_btn.setMinimumHeight(60)
        self._send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(self._send_btn)

        layout.addLayout(input_layout)

        # --- Action buttons ---
        btn_layout = QtWidgets.QHBoxLayout()

        self._btn_insert = QtWidgets.QPushButton("Insert into Script Editor")
        self._btn_insert.setToolTip(
            "Open the generated script in the Script Editor for review and editing")
        self._btn_insert.clicked.connect(self._insert_into_editor)
        self._btn_insert.setEnabled(False)
        btn_layout.addWidget(self._btn_insert)

        self._btn_run = QtWidgets.QPushButton("Run Script")
        self._btn_run.setToolTip(
            "Execute the most recent script directly in the Script Editor")
        self._btn_run.clicked.connect(self._run_script)
        self._btn_run.setEnabled(False)
        btn_layout.addWidget(self._btn_run)

        self._btn_copy = QtWidgets.QPushButton("Copy to Clipboard")
        self._btn_copy.setToolTip(
            "Copy the most recent generated code to the clipboard")
        self._btn_copy.clicked.connect(self._copy_to_clipboard)
        self._btn_copy.setEnabled(False)
        btn_layout.addWidget(self._btn_copy)

        btn_layout.addStretch()

        self._btn_new = QtWidgets.QPushButton("New Conversation")
        self._btn_new.setToolTip(
            "Clear the conversation history and start fresh")
        self._btn_new.clicked.connect(self._new_conversation)
        btn_layout.addWidget(self._btn_new)

        layout.addLayout(btn_layout)

        # --- Status bar ---
        self._status = QtWidgets.QLabel()
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._status)
        self._update_status()

        # --- Keyboard shortcut: Ctrl+Enter to send ---
        shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Return"), self._input)
        shortcut.activated.connect(self._on_send)

    # ------------------------------------------------------------------
    # Instructions toggle
    # ------------------------------------------------------------------

    def _toggle_instructions(self, checked):
        self._instructions.setVisible(checked)
        self._instructions_btn.setText(
            "Hide Instructions" if checked else "Show Instructions")

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _update_context_label(self):
        """Update the label showing info about the current window."""
        try:
            from .. import global_vars as g
            if g.win is not None and hasattr(g.win, 'image'):
                img = g.win.image
                name = getattr(g.win, 'name', 'unknown')
                self._context_label.setText(
                    f"Current window: '{name}' | Shape: {img.shape} | "
                    f"Dtype: {img.dtype} | "
                    f"Range: [{img.min():.1f}, {img.max():.1f}]")
            else:
                self._context_label.setText("No window selected")
        except Exception:
            self._context_label.setText("No window selected")

    def _update_status(self):
        """Update the status bar with the current model name."""
        try:
            from .. import global_vars as g
            model = g.settings['ai_model'] or 'claude-sonnet-4-20250514'
        except Exception:
            model = 'claude-sonnet-4-20250514'
        self._status.setText(f"Model: {model}")

    def _get_context_prompt(self):
        """Build a context string about the current flika state."""
        parts = []
        try:
            from .. import global_vars as g
            if g.win is not None and hasattr(g.win, 'image'):
                img = g.win.image
                name = getattr(g.win, 'name', 'unknown')
                parts.append(
                    f"[Context: Current window '{name}', shape={img.shape}, "
                    f"dtype={img.dtype}, "
                    f"range=[{img.min():.2f}, {img.max():.2f}]]")
            n_windows = (len(g.m.windows)
                         if hasattr(g, 'm') and g.m else 0)
            if n_windows > 0:
                parts.append(f"[{n_windows} window(s) open]")
        except Exception:
            pass
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    def _on_send(self):
        """Send the current input to the assistant."""
        text = self._input.toPlainText().strip()
        if not text:
            return

        self._update_context_label()

        # Prepend context on the first message of the conversation
        user_display = text
        if not self._messages:
            context = self._get_context_prompt()
            if context:
                text = context + "\n\n" + text

        self._messages.append({"role": "user", "content": text})
        self._append_chat("You", user_display)
        self._input.clear()

        self._send_btn.setEnabled(False)
        self._send_btn.setText("Generating...")

        try:
            if self._assistant is None:
                from .assistant import FlikaAssistant
                self._assistant = FlikaAssistant()

            self._worker = _AssistantWorker(
                self._assistant, list(self._messages))
            self._worker.sig_result.connect(self._on_result)
            self._worker.sig_error.connect(self._on_error)
            self._worker.finished.connect(self._on_finished)
            self._worker.start()
        except Exception as e:
            self._on_error(str(e))

    def _on_result(self, text):
        """Handle a successful response from the assistant."""
        self._messages.append({"role": "assistant", "content": text})
        self._last_code = text
        self._append_chat("Assistant", text, is_code=True)
        self._btn_insert.setEnabled(True)
        self._btn_run.setEnabled(True)
        self._btn_copy.setEnabled(True)

    def _on_error(self, msg):
        """Handle an error from the assistant worker."""
        self._append_chat("Error", msg)
        self._on_finished()

    def _on_finished(self):
        """Re-enable the send button after a request completes."""
        self._send_btn.setEnabled(True)
        self._send_btn.setText("Send")

    # ------------------------------------------------------------------
    # Chat display
    # ------------------------------------------------------------------

    def _append_chat(self, sender, text, is_code=False):
        """Append a message to the chat display."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)

        if sender == "You":
            escaped = (text.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace("\n", "<br>"))
            html = (f'<div style="margin: 8px 0;">'
                    f'<b style="color: #569cd6;">You:</b><br>'
                    f'<span style="color: #d4d4d4;">{escaped}</span></div>')
        elif sender == "Error":
            escaped = (text.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))
            html = (f'<div style="margin: 8px 0;">'
                    f'<b style="color: #f44747;">Error:</b> '
                    f'<span style="color: #f44747;">{escaped}</span></div>')
        else:
            # Assistant (code) response
            escaped = (text.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))
            html = (f'<div style="margin: 8px 0;">'
                    f'<b style="color: #4ec9b0;">Assistant:</b><br>'
                    f'<pre style="background: #2d2d2d; padding: 8px; '
                    f'border-radius: 4px; color: #ce9178; '
                    f'white-space: pre-wrap;">{escaped}</pre></div>')

        html += '<hr style="border: 1px solid #333;">'
        cursor.insertHtml(html)

        # Scroll to the bottom
        sb = self._chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Action buttons
    # ------------------------------------------------------------------

    def _insert_into_editor(self):
        """Insert the last generated code into the Script Editor."""
        if not self._last_code:
            return
        try:
            from ..app.script_editor import ScriptEditor
            ScriptEditor.show()
            editor = ScriptEditor.gui.addEditor()
            editor.setPlainText(self._last_code)
            from .. import global_vars as g
            if g.m is not None:
                g.m.statusBar().showMessage(
                    "AI script inserted — review before running.")
        except Exception as e:
            logger.error("Failed to insert into Script Editor: %s", e)

    def _run_script(self):
        """Insert the code and immediately run it in the Script Editor."""
        if not self._last_code:
            return
        # Safety check before running AI-generated code
        from .safety import check_code_safety, request_approval
        is_safe, warnings = check_code_safety(self._last_code)
        if not is_safe:
            if not request_approval(self, self._last_code, warnings):
                return
        try:
            from ..app.script_editor import ScriptEditor
            ScriptEditor.show()
            editor = ScriptEditor.gui.addEditor()
            editor.setPlainText(self._last_code)
            ScriptEditor.gui.runScript()
        except Exception as e:
            logger.error("Failed to run script: %s", e)

    def _copy_to_clipboard(self):
        """Copy the last generated code to the system clipboard."""
        if not self._last_code:
            return
        QtWidgets.QApplication.clipboard().setText(self._last_code)
        self._status.setText("Copied to clipboard!")
        QtCore.QTimer.singleShot(2000, self._update_status)

    def _new_conversation(self):
        """Clear conversation history and reset the UI."""
        self._messages.clear()
        self._chat_display.clear()
        self._btn_insert.setEnabled(False)
        self._btn_run.setEnabled(False)
        self._btn_copy.setEnabled(False)
        self._last_code = ""
        self._update_context_label()
        self._update_status()
