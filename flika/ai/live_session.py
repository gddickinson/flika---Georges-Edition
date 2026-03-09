"""Live AI Session for flika.

An agentic mode where Claude can directly execute actions in flika.
Uses the Anthropic tool-use API so Claude decides when to run code,
inspect the environment, or respond with text.  All actions are logged
to an in-session action log that both the user and the AI can review.
"""
from __future__ import annotations

import datetime
import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from qtpy import QtWidgets, QtCore, QtGui
from ..logger import logger


# ---------------------------------------------------------------------------
# Tool definitions for the Anthropic API
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "execute_code",
        "description": (
            "Execute Python code in flika's script editor namespace. "
            "The namespace has numpy (np), scipy, pyqtgraph (pg), and all "
            "flika process functions pre-imported. Use this to perform "
            "image processing, analysis, file operations, create windows, "
            "apply filters, and any other flika operation. "
            "The code runs on the main thread and can interact with the GUI. "
            "Returns stdout output and any return value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in flika's namespace"
                },
                "explanation": {
                    "type": "string",
                    "description": (
                        "Brief explanation of what this code does, shown to the user"
                    )
                }
            },
            "required": ["code", "explanation"]
        }
    },
    {
        "name": "get_window_info",
        "description": (
            "Get information about the currently selected window in flika. "
            "Returns the window name, image shape, dtype, value range, "
            "number of ROIs, and metadata keys."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "list_windows",
        "description": (
            "List all open windows in flika with their names, shapes, "
            "and dtypes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_image_stats",
        "description": (
            "Get detailed statistics of the current window's image: "
            "min, max, mean, std, median, shape, dtype, and histogram info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_action_log",
        "description": (
            "Retrieve the action log for this session. Shows all actions "
            "taken so far with timestamps, code executed, and results. "
            "Use this to review what has been done or to recall previous "
            "results when the user refers to earlier steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "last_n": {
                    "type": "integer",
                    "description": (
                        "Number of most recent entries to return. "
                        "0 or omitted = all entries."
                    )
                }
            },
            "required": []
        }
    },
]

_LIVE_SYSTEM_PROMPT = """\
You are an AI assistant embedded in *flika*, a PyQt6-based microscopy \
image-processing application for biologists. You are in LIVE SESSION mode, \
which means you can directly execute actions in flika.

You have tools available to:
- **execute_code**: Run Python code directly in flika (apply filters, \
open files, create windows, analyze data, etc.)
- **get_window_info**: Inspect the current window
- **list_windows**: See all open windows
- **get_image_stats**: Get detailed image statistics
- **get_action_log**: Review the session action log (all actions taken \
so far with timestamps, code, and results)

# When to use tools vs. respond with text
- If the user asks you to DO something (apply a filter, open a file, \
threshold an image, analyze data, etc.) → use `execute_code` to do it
- If the user asks a QUESTION about their data → use inspection tools \
first, then answer
- If the user asks a general question → respond with text
- If asked to create a script → use `execute_code` to define it, or just \
respond with the code
- If asked to create a plugin → respond with the code (suggest using \
AI > Claude > Generate Plugin for proper plugin creation)

# Important guidelines
- Always use `keepSourceWindow=True` when applying filters/processing so \
the user doesn't lose their original data
- After processing, briefly describe what was done and the result
- If an operation fails, explain the error and suggest alternatives
- For multi-step workflows, execute each step and verify before continuing
- You can chain multiple tool calls in sequence for complex operations

# flika API Reference
Source code: {source_url}

## Key imports (already in namespace)
- numpy as np, scipy, pyqtgraph as pg
- All flika.process functions: gaussian_blur, threshold, zproject, etc.

## Core API
- `g.win` — current Window; `g.win.image` — numpy array
- `g.m.windows` — list of all Windows
- Image dimensions: 2D=(Y,X), 3D=(T,Y,X), 4D=(T,Y,X,Z)
- Process functions: `func(params, keepSourceWindow=False)`
- File I/O: `open_file('path')`, `save_file('path')`
- ROIs: `from flika.roi import makeROI`
- Window creation: `from flika.window import Window; Window(array, name='...')`

## Available process functions
Filters: gaussian_blur, mean_filter, median_filter, bilateral_filter, \
butterworth_filter, fourier_filter, wavelet_filter, \
difference_of_gaussians, sobel_filter, laplacian_filter, \
gaussian_laplace_filter, tv_denoise, bleach_correction, \
maximum_filter, minimum_filter, percentile_filter, \
sato_tubeness, meijering_neuriteness, hessian_filter, gabor_filter

Binary: threshold, adaptive_threshold, hysteresis_threshold, \
multi_otsu_threshold, canny_edge_detector, \
binary_erosion, binary_dilation, remove_small_blobs, \
remove_small_holes, logically_combine, generate_rois

Math: subtract, multiply, divide, power, sqrt, ratio, \
absolute_value, normalize, histogram_equalize, image_calculator

Stacks: duplicate, trim, zproject, pixel_binning, frame_binning, \
resize, concatenate_stacks, change_datatype, motion_correction

Segmentation: connected_components, region_properties, \
watershed_segmentation, find_boundaries, find_contours_process

Detection: blob_detection_log, blob_detection_doh, peak_local_max, \
template_match, analyze_particles

Deconvolution: richardson_lucy, wiener_deconvolution, generate_psf
Color: split_channels, blend_channels, convert_color_space, grayscale
Dynamics: frap_analysis, fret_analysis, calcium_analysis, spectral_unmixing
SPT: spt_analysis, detect_particles, link_particles_process
Simulation: simulate.run(preset='...')
"""

_LIVE_INSTRUCTIONS = """\
<h3>Live Session — How to Use</h3>
<p>In Live Session mode, Claude can <b>directly perform actions</b> in flika. \
Just describe what you want done in natural language.</p>

<h4>Examples:</h4>
<ul>
<li>"Apply a Gaussian blur with sigma 3"</li>
<li>"Threshold the image at 150 and remove small objects"</li>
<li>"What's the mean intensity of the current image?"</li>
<li>"Create a max projection of this stack"</li>
<li>"Open all TIFF files in /path/to/folder"</li>
<li>"Detect bright spots and count them"</li>
<li>"Apply a median filter, then threshold, then measure connected components"</li>
<li>"Compare the histograms of the two open windows"</li>
<li>"Run Richardson-Lucy deconvolution with 20 iterations"</li>
</ul>

<h4>How it works:</h4>
<ul>
<li>Claude analyzes your request and decides what actions to take</li>
<li>Code is executed <b>directly</b> in flika — results appear immediately</li>
<li>Claude reports back on what was done and any results</li>
<li>You can give follow-up instructions to refine the workflow</li>
<li>Original windows are preserved (keepSourceWindow=True)</li>
</ul>

<h4>Tips:</h4>
<ul>
<li>Be as specific or as vague as you like — Claude will figure it out</li>
<li>Ask questions about your data — Claude will inspect it first</li>
<li>Chain multiple operations: "blur, threshold, and count objects"</li>
<li>All actions are logged — click <b>Show Log</b> to review or export</li>
<li>You can undo operations with Edit > Undo</li>
</ul>
"""


# ---------------------------------------------------------------------------
# Action Log
# ---------------------------------------------------------------------------

class ActionLog:
    """In-session log of all actions taken by the AI assistant."""

    def __init__(self):
        self._entries = []

    def add(self, action_type, code="", explanation="", result="",
            error=""):
        """Add an entry to the log."""
        entry = {
            "timestamp": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"),
            "action_type": action_type,
            "explanation": explanation,
            "code": code,
            "result": result[:500] if result else "",
            "error": error,
            "index": len(self._entries) + 1,
        }
        self._entries.append(entry)
        return entry

    def get(self, last_n=0):
        """Return log entries. last_n=0 means all."""
        if last_n and last_n > 0:
            return self._entries[-last_n:]
        return list(self._entries)

    def clear(self):
        self._entries.clear()

    def to_text(self, last_n=0):
        """Format the log as readable text."""
        entries = self.get(last_n)
        if not entries:
            return "Action log is empty."
        lines = [f"=== Action Log ({len(entries)} entries) ===\n"]
        for e in entries:
            lines.append(f"--- Step {e['index']} [{e['timestamp']}] ---")
            lines.append(f"Action: {e['action_type']}")
            if e['explanation']:
                lines.append(f"Description: {e['explanation']}")
            if e['code']:
                lines.append(f"Code:\n{e['code']}")
            if e['result']:
                lines.append(f"Result: {e['result']}")
            if e['error']:
                lines.append(f"ERROR: {e['error']}")
            lines.append("")
        return "\n".join(lines)

    def to_script(self):
        """Export all executed code as a single script."""
        lines = [
            "# Auto-generated script from Claude Live Session",
            f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Total actions: {len(self._entries)}",
            "",
        ]
        for e in self._entries:
            if e['action_type'] == 'execute_code' and e['code']:
                if e['explanation']:
                    lines.append(f"# {e['explanation']}")
                lines.append(e['code'])
                lines.append("")
        return "\n".join(lines)

    def __len__(self):
        return len(self._entries)


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------

# Module-level action log instance (shared by dialog)
_action_log = ActionLog()


def _get_namespace():
    """Get or create the script editor namespace for code execution."""
    from ..app.script_namespace import getnamespace
    if not hasattr(_get_namespace, '_ns'):
        _get_namespace._ns = getnamespace()
    return _get_namespace._ns


def _execute_tool_call(tool_name, tool_input):
    """Execute a tool call and return the result string."""
    if tool_name == "execute_code":
        return _tool_execute_code(
            tool_input.get("code", ""),
            tool_input.get("explanation", ""))
    elif tool_name == "get_window_info":
        return _tool_get_window_info()
    elif tool_name == "list_windows":
        return _tool_list_windows()
    elif tool_name == "get_image_stats":
        return _tool_get_image_stats()
    elif tool_name == "get_action_log":
        return _tool_get_action_log(tool_input.get("last_n", 0))
    else:
        return f"Unknown tool: {tool_name}"


def _tool_execute_code(code, explanation):
    """Execute Python code in flika's namespace and log the action."""
    ns = _get_namespace()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, ns)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        result_parts = []
        if stdout.strip():
            result_parts.append(f"Output:\n{stdout.strip()}")
        if stderr.strip():
            result_parts.append(f"Warnings:\n{stderr.strip()}")
        if not result_parts:
            result_parts.append("Code executed successfully (no output).")
        result = "\n".join(result_parts)

        _action_log.add("execute_code", code=code,
                        explanation=explanation, result=result)
        return result

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Error executing code:\n{tb}"
        _action_log.add("execute_code", code=code,
                        explanation=explanation, error=error_msg)
        return error_msg


def _tool_get_action_log(last_n=0):
    """Return the session action log."""
    return _action_log.to_text(last_n)


def _tool_get_window_info():
    """Get info about the current window."""
    try:
        from .. import global_vars as g
        if g.win is None:
            return "No window is currently selected."

        img = g.win.image
        info = {
            "name": getattr(g.win, 'name', 'unknown'),
            "shape": str(img.shape),
            "dtype": str(img.dtype),
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
            "n_rois": len(g.win.rois) if hasattr(g.win, 'rois') else 0,
            "metadata_keys": list(g.win.metadata.keys()) if hasattr(g.win, 'metadata') else [],
        }
        if img.ndim >= 3:
            info["n_frames"] = img.shape[0]
            info["current_frame"] = getattr(g.win, 'currentIndex', 0)

        lines = [f"{k}: {v}" for k, v in info.items()]
        return "Current window info:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error getting window info: {e}"


def _tool_list_windows():
    """List all open windows."""
    try:
        from .. import global_vars as g
        if not hasattr(g, 'm') or g.m is None or not g.m.windows:
            return "No windows are currently open."

        lines = []
        for i, w in enumerate(g.m.windows):
            name = getattr(w, 'name', f'Window {i}')
            shape = str(w.image.shape) if hasattr(w, 'image') else 'unknown'
            dtype = str(w.image.dtype) if hasattr(w, 'image') else 'unknown'
            selected = " (selected)" if w is g.win else ""
            lines.append(f"  [{i}] '{name}' — shape={shape}, dtype={dtype}{selected}")

        return f"{len(g.m.windows)} window(s) open:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing windows: {e}"


def _tool_get_image_stats():
    """Get detailed image statistics."""
    try:
        import numpy as np
        from .. import global_vars as g

        if g.win is None:
            return "No window is currently selected."

        img = g.win.image
        stats = {
            "shape": str(img.shape),
            "dtype": str(img.dtype),
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
            "std": float(img.std()),
            "median": float(np.median(img)),
        }

        if img.ndim == 2:
            dims = f"2D image: {img.shape[0]}(Y) x {img.shape[1]}(X)"
        elif img.ndim == 3:
            dims = (f"3D stack: {img.shape[0]}(T) x {img.shape[1]}(Y) "
                    f"x {img.shape[2]}(X)")
        elif img.ndim == 4:
            dims = (f"4D stack: {img.shape[0]}(T) x {img.shape[1]}(Y) "
                    f"x {img.shape[2]}(X) x {img.shape[3]}(Z)")
        else:
            dims = f"{img.ndim}D array"
        stats["dimensions"] = dims

        # Histogram summary
        hist, bin_edges = np.histogram(img.ravel(), bins=10)
        hist_lines = []
        for i in range(len(hist)):
            hist_lines.append(
                f"  [{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}]: {hist[i]}")
        stats["histogram_10bin"] = "\n".join(hist_lines)

        lines = []
        for k, v in stats.items():
            if k == "histogram_10bin":
                lines.append(f"Histogram (10 bins):\n{v}")
            else:
                lines.append(f"{k}: {v}")
        return "Image statistics:\n" + "\n".join(lines)
    except Exception as e:
        return f"Error getting image stats: {e}"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _LiveSessionWorker(QtCore.QThread):
    """Background thread for Anthropic API calls with tool use."""
    sig_text = QtCore.Signal(str)        # Text response from Claude
    sig_tool_call = QtCore.Signal(str, str, dict)  # id, name, input
    sig_error = QtCore.Signal(str)
    sig_done = QtCore.Signal()

    def __init__(self, client, model, system, messages, tools):
        super().__init__()
        self.client = client
        self.model = model
        self.system = system
        self.messages = messages
        self.tools = tools

    def run(self):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system,
                messages=self.messages,
                tools=self.tools,
            )

            # Process the response content blocks
            for block in response.content:
                if block.type == "text":
                    self.sig_text.emit(block.text)
                elif block.type == "tool_use":
                    self.sig_tool_call.emit(
                        block.id, block.name, block.input)

            # If the model wants to use tools, we need to signal that
            # we're not done yet (the main thread will handle tool
            # execution and continue the loop)
            if response.stop_reason == "tool_use":
                pass  # Main thread handles continuation
            self.sig_done.emit()

        except Exception as e:
            logger.error("Live session API error: %s\n%s",
                         e, traceback.format_exc())
            self.sig_error.emit(f"{type(e).__name__}: {e}")
            self.sig_done.emit()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class LiveSessionDialog(QtWidgets.QDialog):
    """Agentic AI session that can directly execute actions in flika."""

    _instance = None

    @classmethod
    def show_dialog(cls):
        """Show the singleton dialog."""
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent=None)
        cls._instance.show()
        cls._instance.raise_()
        cls._instance.activateWindow()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Claude Live Session")
        self.resize(850, 750)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

        self._messages = []
        self._client = None
        self._model = None
        self._worker = None
        self._pending_tool_calls = []
        self._tool_results = []

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Instructions (collapsible)
        self._instructions_btn = QtWidgets.QPushButton("Show Instructions")
        self._instructions_btn.setCheckable(True)
        self._instructions_btn.toggled.connect(self._toggle_instructions)
        layout.addWidget(self._instructions_btn)

        self._instructions = QtWidgets.QTextBrowser()
        self._instructions.setHtml(_LIVE_INSTRUCTIONS)
        self._instructions.setMaximumHeight(280)
        self._instructions.setVisible(False)
        layout.addWidget(self._instructions)

        # Chat display
        self._chat = QtWidgets.QTextBrowser()
        self._chat.setOpenExternalLinks(True)
        self._chat.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Courier New', Courier, monospace;
                font-size: 13px;
                padding: 8px;
            }
        """)
        layout.addWidget(self._chat, 1)

        # Context label
        self._context_label = QtWidgets.QLabel()
        self._context_label.setStyleSheet(
            "color: gray; font-size: 11px; padding: 2px;")
        layout.addWidget(self._context_label)
        self._update_context_label()

        # Input area
        input_layout = QtWidgets.QHBoxLayout()

        self._input = QtWidgets.QPlainTextEdit()
        self._input.setMaximumHeight(80)
        self._input.setPlaceholderText(
            "Tell Claude what to do... (Ctrl+Enter to send)")
        input_layout.addWidget(self._input, 1)

        self._send_btn = QtWidgets.QPushButton("Send")
        self._send_btn.setMinimumHeight(50)
        self._send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(self._send_btn)
        layout.addLayout(input_layout)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.setToolTip("Stop the current operation")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)
        btn_layout.addWidget(self._btn_stop)

        self._btn_log = QtWidgets.QPushButton("Show Log")
        self._btn_log.setToolTip("View the action log for this session")
        self._btn_log.clicked.connect(self._show_log)
        btn_layout.addWidget(self._btn_log)

        self._btn_export = QtWidgets.QPushButton("Export as Script")
        self._btn_export.setToolTip(
            "Export all executed code as a single Python script")
        self._btn_export.clicked.connect(self._export_script)
        btn_layout.addWidget(self._btn_export)

        btn_layout.addStretch()

        self._btn_new = QtWidgets.QPushButton("New Session")
        self._btn_new.setToolTip("Clear history and start fresh")
        self._btn_new.clicked.connect(self._new_session)
        btn_layout.addWidget(self._btn_new)

        layout.addLayout(btn_layout)

        # Status
        self._status = QtWidgets.QLabel()
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._status)
        self._update_status()

        # Ctrl+Enter shortcut
        shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Return"), self._input)
        shortcut.activated.connect(self._on_send)

    def _toggle_instructions(self, checked):
        self._instructions.setVisible(checked)
        self._instructions_btn.setText(
            "Hide Instructions" if checked else "Show Instructions")

    def _update_context_label(self):
        try:
            from .. import global_vars as g
            if g.win is not None and hasattr(g.win, 'image'):
                img = g.win.image
                name = getattr(g.win, 'name', 'unknown')
                n_win = len(g.m.windows) if hasattr(g, 'm') and g.m else 0
                self._context_label.setText(
                    f"Current: '{name}' | Shape: {img.shape} | "
                    f"Dtype: {img.dtype} | "
                    f"Windows: {n_win}")
            else:
                self._context_label.setText("No window selected")
        except Exception:
            self._context_label.setText("No window selected")

    def _update_status(self, msg=None):
        if msg:
            self._status.setText(msg)
            return
        try:
            from .. import global_vars as g
            model = g.settings['ai_model'] or 'claude-sonnet-4-20250514'
        except Exception:
            model = 'claude-sonnet-4-20250514'
        self._status.setText(f"Model: {model} | Mode: Live Session (agentic)")

    def _init_client(self):
        """Initialize the Anthropic client if needed."""
        if self._client is not None:
            return

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required. "
                "Install with: pip install anthropic")

        from ..app.settings_editor import get_api_key
        api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY or "
                "enter your key in Edit > Settings.")

        self._client = anthropic.Anthropic(api_key=api_key)

        try:
            from .. import global_vars as g
            self._model = g.settings['ai_model'] or 'claude-sonnet-4-20250514'
        except Exception:
            self._model = 'claude-sonnet-4-20250514'

    def _get_system_prompt(self):
        try:
            from .. import global_vars as g
            url = (g.settings['flika_source_url']
                   or 'https://github.com/gddickinson/flika---Georges-Edition')
        except Exception:
            url = 'https://github.com/gddickinson/flika---Georges-Edition'
        return _LIVE_SYSTEM_PROMPT.format(source_url=url)

    # ------------------------------------------------------------------
    # Send / receive loop
    # ------------------------------------------------------------------

    def _on_send(self):
        text = self._input.toPlainText().strip()
        if not text:
            return

        self._update_context_label()
        self._append_chat("You", text)
        self._input.clear()

        # Add context on first message
        user_content = text
        if not self._messages:
            context = self._build_context()
            if context:
                user_content = context + "\n\n" + text

        self._messages.append({"role": "user", "content": user_content})

        self._set_busy(True)
        self._call_api()

    def _build_context(self):
        """Build context about current flika state."""
        parts = []
        try:
            from .. import global_vars as g
            if g.win is not None and hasattr(g.win, 'image'):
                img = g.win.image
                name = getattr(g.win, 'name', 'unknown')
                parts.append(
                    f"[Current window: '{name}', shape={img.shape}, "
                    f"dtype={img.dtype}, "
                    f"range=[{img.min():.2f}, {img.max():.2f}]]")
            n_win = len(g.m.windows) if hasattr(g, 'm') and g.m else 0
            if n_win > 0:
                parts.append(f"[{n_win} window(s) open]")
        except Exception:
            pass
        return "\n".join(parts)

    def _call_api(self):
        """Make an API call with the current messages."""
        try:
            self._init_client()
        except Exception as e:
            self._append_chat("Error", str(e))
            self._set_busy(False)
            return

        self._pending_tool_calls = []
        self._pending_text_blocks = []

        self._worker = _LiveSessionWorker(
            self._client, self._model,
            self._get_system_prompt(),
            list(self._messages), _TOOLS)
        self._worker.sig_text.connect(self._on_text_response)
        self._worker.sig_tool_call.connect(self._on_tool_call)
        self._worker.sig_error.connect(self._on_error)
        self._worker.sig_done.connect(self._on_api_done)
        self._worker.start()

    def _on_text_response(self, text):
        """Handle a text block from Claude."""
        self._append_chat("Claude", text)
        self._pending_text_blocks.append({"type": "text", "text": text})

    def _on_tool_call(self, tool_id, tool_name, tool_input):
        """Queue a tool call for execution on the main thread."""
        self._pending_tool_calls.append((tool_id, tool_name, tool_input))

    def _on_api_done(self):
        """API call finished. Execute any pending tool calls."""
        if not self._pending_tool_calls:
            # No tool calls — just text response.
            # Add text-only assistant message to history if we have text.
            if self._pending_text_blocks:
                self._messages.append({
                    "role": "assistant",
                    "content": self._pending_text_blocks,
                })
                self._pending_text_blocks = []
            self._set_busy(False)
            self._update_context_label()
            return

        # Build the assistant content: text blocks + tool_use blocks
        assistant_content = list(self._pending_text_blocks)
        self._pending_text_blocks = []

        # Execute each tool call on the main thread
        tool_result_blocks = []

        for tool_id, tool_name, tool_input in self._pending_tool_calls:
            # Show what's being executed
            explanation = tool_input.get('explanation', '')
            code = tool_input.get('code', '')

            if tool_name == "execute_code" and explanation:
                self._append_chat("Action", explanation)
            if tool_name == "execute_code" and code:
                self._append_chat("Code", code)
            elif tool_name != "execute_code":
                self._append_chat(
                    "Action", f"Inspecting: {tool_name}")

            self._update_status(f"Executing: {tool_name}...")
            QtWidgets.QApplication.processEvents()

            # Execute the tool
            result = _execute_tool_call(tool_name, tool_input)

            # Show result
            if result and result != "Code executed successfully (no output).":
                self._append_chat("Result", result)

            # Build the tool_use and tool_result content blocks
            assistant_content.append({
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            })
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result,
            })

        self._pending_tool_calls = []

        # Append assistant message (text + tool_use blocks)
        # and user message (tool_result blocks) to history
        self._messages.append({
            "role": "assistant",
            "content": assistant_content,
        })
        self._messages.append({
            "role": "user",
            "content": tool_result_blocks,
        })

        # Continue the conversation — Claude may want to use more tools
        # or provide a final text response
        self._update_status("Continuing...")
        self._call_api()

    def _on_error(self, msg):
        self._append_chat("Error", msg)
        self._set_busy(False)

    def _on_stop(self):
        """Stop the current operation."""
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self._pending_tool_calls = []
        self._set_busy(False)
        self._append_chat("System", "Operation stopped by user.")

    def _set_busy(self, busy):
        self._send_btn.setEnabled(not busy)
        self._send_btn.setText("Working..." if busy else "Send")
        self._btn_stop.setEnabled(busy)
        self._input.setEnabled(not busy)
        if not busy:
            self._update_status()

    # ------------------------------------------------------------------
    # Chat display
    # ------------------------------------------------------------------

    def _append_chat(self, sender, text):
        """Append a message to the chat display."""
        escaped = (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))

        colors = {
            "You": "#569cd6",
            "Claude": "#4ec9b0",
            "Action": "#dcdcaa",
            "Code": "#ce9178",
            "Result": "#9cdcfe",
            "Error": "#f44747",
            "System": "#808080",
        }
        color = colors.get(sender, "#d4d4d4")

        if sender == "Code":
            html = (f'<div style="margin: 4px 0;">'
                    f'<pre style="background: #2d2d2d; padding: 6px; '
                    f'border-radius: 4px; color: {color}; '
                    f'white-space: pre-wrap; font-size: 12px;">'
                    f'{escaped}</pre></div>')
        elif sender == "Result":
            html = (f'<div style="margin: 4px 0; padding-left: 12px; '
                    f'border-left: 2px solid #444;">'
                    f'<span style="color: {color}; font-size: 12px;">'
                    f'{escaped.replace(chr(10), "<br>")}</span></div>')
        elif sender == "Action":
            html = (f'<div style="margin: 4px 0;">'
                    f'<span style="color: {color};">▶ {escaped}</span>'
                    f'</div>')
        elif sender == "Claude":
            html = (f'<div style="margin: 8px 0;">'
                    f'<b style="color: {color};">Claude:</b><br>'
                    f'<span style="color: #d4d4d4;">'
                    f'{escaped.replace(chr(10), "<br>")}</span></div>'
                    f'<hr style="border: 1px solid #333;">')
        elif sender == "You":
            html = (f'<div style="margin: 8px 0;">'
                    f'<b style="color: {color};">You:</b><br>'
                    f'<span style="color: #d4d4d4;">'
                    f'{escaped.replace(chr(10), "<br>")}</span></div>'
                    f'<hr style="border: 1px solid #333;">')
        else:
            html = (f'<div style="margin: 4px 0;">'
                    f'<span style="color: {color};">'
                    f'[{sender}] {escaped}</span></div>')

        cursor = self._chat.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertHtml(html)

        sb = self._chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_log(self):
        """Show the action log in a popup window."""
        log_text = _action_log.to_text()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Action Log ({len(_action_log)} entries)")
        dlg.resize(700, 500)
        layout = QtWidgets.QVBoxLayout(dlg)

        text_edit = QtWidgets.QPlainTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(log_text)
        text_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e; color: #d4d4d4;
                font-family: 'Courier New', monospace; font-size: 12px;
            }
        """)
        layout.addWidget(text_edit)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(log_text))
        btn_layout.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("Save to File")
        def _save_log():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save Action Log", "flika_session_log.txt",
                "Text Files (*.txt)")
            if path:
                with open(path, 'w') as f:
                    f.write(log_text)
        btn_save.clicked.connect(_save_log)
        btn_layout.addWidget(btn_save)

        btn_layout.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        dlg.show()

    def _export_script(self):
        """Export all executed code as a single Python script."""
        if len(_action_log) == 0:
            self._update_status("No actions to export.")
            return

        script = _action_log.to_script()

        # Open in Script Editor
        try:
            from ..app.script_editor import ScriptEditor
            ScriptEditor.show()
            editor = ScriptEditor.gui.addEditor()
            editor.setPlainText(script)
            self._update_status(
                f"Exported {len(_action_log)} actions to Script Editor.")
        except Exception as e:
            self._update_status(f"Export failed: {e}")

    def _new_session(self):
        """Clear history and start fresh."""
        self._messages.clear()
        self._chat.clear()
        self._pending_tool_calls = []
        _action_log.clear()
        self._update_context_label()
        self._update_status()
        self._set_busy(False)


def _show_live_session():
    """Menu callback for AI > Claude > Live Session."""
    LiveSessionDialog.show_dialog()
