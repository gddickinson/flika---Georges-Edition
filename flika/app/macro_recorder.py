"""Macro recorder for flika.

Records BaseProcess commands as they run, producing a Python script that
can replay the exact sequence of operations.

Usage::

    from flika.app.macro_recorder import macro_recorder

    macro_recorder.start()
    # … user performs operations …
    macro_recorder.stop()
    script = macro_recorder.get_script()
    macro_recorder.save('my_macro.py')
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List

from ..logger import logger


class MacroRecorder:
    """Singleton that accumulates flika commands while recording is active."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._recording = False
            cls._instance._commands: List[str] = []
            cls._instance._structured_records: List[dict] = []
        return cls._instance

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self):
        """Begin recording."""
        self._commands.clear()
        self._structured_records.clear()
        self._recording = True
        logger.info("Macro recording started")

    def stop(self):
        """Stop recording."""
        self._recording = False
        logger.info("Macro recording stopped (%d commands)", len(self._commands))

    def record(self, command: str):
        """Append *command* if currently recording."""
        if self._recording:
            self._commands.append(command)

    def get_script(self) -> str:
        """Return the accumulated commands as a runnable Python script."""
        header = (
            f"# Flika macro recorded {datetime.now().isoformat()}\n"
            "from flika.process import *\n"
            "from flika.process.file_ import open_file\n"
            "\n"
        )
        return header + '\n'.join(self._commands) + '\n'

    def save(self, path: str):
        """Write the recorded script to *path*."""
        with open(path, 'w') as f:
            f.write(self.get_script())
        logger.info("Macro saved to %s", path)

    def clear(self):
        """Discard all recorded commands."""
        self._commands.clear()
        self._structured_records.clear()

    def record_structured(self, command: str, duration_seconds: float = 0.0,
                          input_window: str = '', input_shape: tuple = (),
                          output_window: str = '', output_shape: tuple = ()):
        """Record a structured command entry with timing and shape info."""
        from datetime import datetime
        entry = {
            'command': command,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'input_window': input_window,
            'input_shape': tuple(input_shape),
            'output_window': output_window,
            'output_shape': tuple(output_shape),
        }
        self._structured_records.append(entry)


# Singleton instance
macro_recorder = MacroRecorder()


def _toggle_recording():
    """Menu callback: start or stop recording."""
    from .. import global_vars as g
    if macro_recorder.is_recording:
        macro_recorder.stop()
        g.m.statusBar().showMessage("Macro recording stopped.")
    else:
        macro_recorder.start()
        g.m.statusBar().showMessage("Macro recording started...")


def _save_macro():
    """Menu callback: save the recorded macro to a file."""
    from .. import global_vars as g
    from ..utils.misc import save_file_gui
    if not macro_recorder._commands:
        g.alert("No commands recorded.")
        return
    path = save_file_gui("Save Macro", filetypes='Python scripts (*.py)')
    if path:
        macro_recorder.save(path)
        g.m.statusBar().showMessage(f"Macro saved to {os.path.basename(path)}")


def _run_macro():
    """Menu callback: open and execute a macro .py file."""
    from .. import global_vars as g
    from ..utils.misc import open_file_gui
    path = open_file_gui("Run Macro", filetypes='Python scripts (*.py)')
    if path:
        try:
            with open(path, 'r') as f:
                script = f.read()
            exec(compile(script, path, 'exec'))
            g.m.statusBar().showMessage(f"Macro executed: {os.path.basename(path)}")
        except Exception as e:
            logger.error("Macro execution failed: %s", e)
            g.alert(f"Macro execution failed:\n{e}")
