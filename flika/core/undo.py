"""Undo/redo system for flika image processing operations.

Each :class:`ProcessCommand` stores enough state to reverse a single
BaseProcess invocation: the old window's image, name, and metadata.

The :data:`undo_stack` singleton manages the history with a configurable
maximum size and memory guard.

Usage::

    from flika.core.undo import undo_stack
    undo_stack.undo()
    undo_stack.redo()
"""
from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np

from flika.logger import logger


# Maximum number of operations to keep.  Each entry stores a full copy of
# the image array so this can consume a lot of RAM.
_MAX_STACK_SIZE = 20
_WARN_BYTES = 500 * 1024 * 1024  # 500 MB


class ProcessCommand:
    """Stores the state needed to undo/redo one BaseProcess operation."""

    __slots__ = (
        'old_image', 'old_name', 'old_filename', 'old_commands', 'old_metadata',
        'old_volume',
        'new_image', 'new_name', 'new_filename', 'new_commands', 'new_metadata',
        'new_volume',
        'command_str', 'keep_source',
        '_restored_window', '_result_window',
    )

    def __init__(self, old_image, old_name, old_filename, old_commands, old_metadata,
                 new_window, command_str, keep_source, old_volume=None):
        self.old_image = old_image
        self.old_name = old_name
        self.old_filename = old_filename
        self.old_commands = old_commands
        self.old_metadata = old_metadata
        self.old_volume = old_volume
        self.command_str = command_str
        self.keep_source = keep_source
        # Filled in when new_window is set
        self.new_image = None
        self.new_name = None
        self.new_filename = None
        self.new_commands = None
        self.new_metadata = None
        self.new_volume = None
        self._restored_window = None  # window created by undo
        self._result_window = new_window  # window created by the operation

    def capture_new_state(self, window):
        """Capture the new window's state for redo."""
        self.new_image = window.image.copy()
        self.new_name = window.name
        self.new_filename = window.filename
        self.new_commands = window.commands[:]
        self.new_metadata = window.metadata.copy() if hasattr(window.metadata, 'copy') else dict(window.metadata)
        self.new_volume = window.volume.copy() if window.volume is not None else None
        self._result_window = window

    @property
    def nbytes(self) -> int:
        total = self.old_image.nbytes if self.old_image is not None else 0
        total += self.new_image.nbytes if self.new_image is not None else 0
        total += self.old_volume.nbytes if self.old_volume is not None else 0
        total += self.new_volume.nbytes if self.new_volume is not None else 0
        return total


class UndoStack:
    """Fixed-size undo/redo stack."""

    def __init__(self, max_size: int = _MAX_STACK_SIZE):
        self._undo: List[ProcessCommand] = []
        self._redo: List[ProcessCommand] = []
        self.max_size = max_size

    def push(self, cmd: ProcessCommand):
        """Push a new command.  Clears the redo stack."""
        if cmd.nbytes > _WARN_BYTES:
            logger.warning(
                "Undo entry for %r uses %.1f MB of memory",
                cmd.command_str, cmd.nbytes / 1024 / 1024,
            )
        self._undo.append(cmd)
        self._redo.clear()
        # Enforce size limit
        while len(self._undo) > self.max_size:
            self._undo.pop(0)

    def undo(self):
        """Undo the most recent operation."""
        import flika.global_vars as g
        from flika.window import Window
        if not self._undo:
            if g.m:
                g.m.statusBar().showMessage('Nothing to undo')
            return
        cmd = self._undo.pop()
        self._redo.append(cmd)
        # Close the result window (or the previously restored window on repeated undo)
        target = cmd._result_window
        if target is not None and not target.closed:
            # Capture new state for redo before closing
            if cmd.new_image is None:
                cmd.capture_new_state(target)
            target.close()
        # Close any previously restored window from a prior undo of this cmd
        if cmd._restored_window is not None and not cmd._restored_window.closed:
            cmd._restored_window.close()
        # Restore old window
        old_vol = cmd.old_volume
        restore_data = old_vol.copy() if old_vol is not None else cmd.old_image.copy()
        restored = Window(
            restore_data, cmd.old_name, cmd.old_filename,
            cmd.old_commands[:], cmd.old_metadata,
        )
        cmd._restored_window = restored
        if g.m:
            g.m.statusBar().showMessage(f'Undid: {cmd.command_str}')
        logger.info("Undo: %s", cmd.command_str)
        return restored

    def redo(self):
        """Re-apply the most recently undone operation."""
        import flika.global_vars as g
        from flika.window import Window
        if not self._redo:
            if g.m:
                g.m.statusBar().showMessage('Nothing to redo')
            return
        cmd = self._redo.pop()
        self._undo.append(cmd)
        # Close the restored window from undo
        if cmd._restored_window is not None and not cmd._restored_window.closed:
            cmd._restored_window.close()
            cmd._restored_window = None
        # Re-create the result window
        if cmd.new_image is not None:
            new_vol = cmd.new_volume
            redo_data = new_vol.copy() if new_vol is not None else cmd.new_image.copy()
            result = Window(
                redo_data, cmd.new_name, cmd.new_filename,
                cmd.new_commands[:], cmd.new_metadata,
            )
            cmd._result_window = result
        if g.m:
            g.m.statusBar().showMessage(f'Redo: {cmd.command_str}')
        logger.info("Redo: %s", cmd.command_str)

    def clear(self):
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    @property
    def total_bytes(self) -> int:
        return sum(c.nbytes for c in self._undo) + sum(c.nbytes for c in self._redo)


# Singleton
undo_stack = UndoStack()
