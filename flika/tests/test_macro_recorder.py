"""Tests for the macro recorder."""
import os
import tempfile

import pytest

from flika.app.macro_recorder import MacroRecorder


class TestMacroRecorder:
    def setup_method(self):
        self.rec = MacroRecorder.__new__(MacroRecorder)
        self.rec._recording = False
        self.rec._commands = []

    def test_not_recording_by_default(self):
        assert not self.rec.is_recording

    def test_start_stop(self):
        self.rec.start()
        assert self.rec.is_recording
        self.rec.stop()
        assert not self.rec.is_recording

    def test_record_only_when_active(self):
        self.rec.record("gaussian_blur(sigma=1)")
        assert len(self.rec._commands) == 0

        self.rec.start()
        self.rec.record("gaussian_blur(sigma=1)")
        assert len(self.rec._commands) == 1

    def test_start_clears_previous(self):
        self.rec.start()
        self.rec.record("cmd1()")
        self.rec.start()
        assert len(self.rec._commands) == 0

    def test_get_script(self):
        self.rec.start()
        self.rec.record("threshold(value=128)")
        self.rec.record("gaussian_blur(sigma=2)")
        script = self.rec.get_script()
        assert "threshold(value=128)" in script
        assert "gaussian_blur(sigma=2)" in script
        assert "from flika.process import *" in script

    def test_save(self):
        self.rec.start()
        self.rec.record("test_command()")
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            path = f.name
        try:
            self.rec.save(path)
            with open(path) as f:
                content = f.read()
            assert "test_command()" in content
        finally:
            os.unlink(path)

    def test_clear(self):
        self.rec.start()
        self.rec.record("a()")
        self.rec.record("b()")
        self.rec.clear()
        assert len(self.rec._commands) == 0
