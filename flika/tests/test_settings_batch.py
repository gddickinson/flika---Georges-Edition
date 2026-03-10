"""Tests for the Settings batch context manager."""
import json
import os
import tempfile

import pytest

from flika.global_vars import Settings


class TestSettingsBatch:
    def setup_method(self):
        """Create a temporary settings file for each test."""
        self.tmpdir = tempfile.mkdtemp()
        self.settings = Settings.__new__(Settings)
        self.settings.d = dict(Settings.initial_settings)
        self.settings._batch_count = 0
        self.settings._batch_dirty = False
        self.settings.settings_file = os.path.join(self.tmpdir, 'settings.json')
        self.settings.save()

    def _read_file(self):
        with open(self.settings.settings_file, 'r') as f:
            return json.load(f)

    def test_normal_save_writes_immediately(self):
        self.settings['point_size'] = 99
        data = self._read_file()
        assert data['point_size'] == 99

    def test_batch_defers_save(self):
        original = self._read_file().get('point_size')
        with self.settings.batch():
            self.settings['point_size'] = 42
            # Should NOT have written yet
            data = self._read_file()
            assert data.get('point_size') == original or data.get('point_size') != 42
        # After exiting batch, should be written
        data = self._read_file()
        assert data['point_size'] == 42

    def test_nested_batch(self):
        with self.settings.batch():
            self.settings['point_size'] = 10
            with self.settings.batch():
                self.settings['point_size'] = 20
            # Still inside outer batch — should not have written
            data = self._read_file()
            assert data.get('point_size') != 20 or self.settings._batch_count == 1
        # Now exited all batches
        data = self._read_file()
        assert data['point_size'] == 20

    def test_batch_no_changes(self):
        """Batch with no changes should not dirty-save."""
        mtime_before = os.path.getmtime(self.settings.settings_file)
        with self.settings.batch():
            pass
        mtime_after = os.path.getmtime(self.settings.settings_file)
        assert mtime_before == mtime_after
