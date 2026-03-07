"""Tests for the undo/redo stack."""
import numpy as np
import pytest

from ..core.undo import UndoStack, ProcessCommand


def _make_cmd(label="test", shape=(10, 10)):
    """Helper to create a dummy ProcessCommand."""
    return ProcessCommand(
        old_image=np.zeros(shape, dtype=np.float32),
        old_name="old",
        old_filename="old.tif",
        old_commands=[],
        old_metadata={},
        new_window=None,
        command_str=label,
        keep_source=False,
    )


class TestUndoStack:
    def setup_method(self):
        self.stack = UndoStack(max_size=5)

    def test_initially_empty(self):
        assert not self.stack.can_undo
        assert not self.stack.can_redo

    def test_push_enables_undo(self):
        self.stack.push(_make_cmd())
        assert self.stack.can_undo
        assert not self.stack.can_redo

    def test_push_clears_redo(self):
        self.stack.push(_make_cmd("a"))
        # Simulate undo manually
        self.stack._redo.append(self.stack._undo.pop())
        assert self.stack.can_redo
        self.stack.push(_make_cmd("b"))
        assert not self.stack.can_redo

    def test_max_size_enforced(self):
        for i in range(10):
            self.stack.push(_make_cmd(f"cmd{i}"))
        assert len(self.stack._undo) == 5

    def test_clear(self):
        self.stack.push(_make_cmd())
        self.stack.clear()
        assert not self.stack.can_undo
        assert not self.stack.can_redo

    def test_total_bytes(self):
        cmd = _make_cmd(shape=(100, 100))
        self.stack.push(cmd)
        assert self.stack.total_bytes == 100 * 100 * 4  # float32

    def test_nbytes_property(self):
        cmd = _make_cmd(shape=(50, 50))
        assert cmd.nbytes == 50 * 50 * 4
