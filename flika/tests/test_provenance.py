"""Tests for workflow provenance (flika.utils.provenance)."""
import json
import os
import tempfile

import numpy as np
import pytest

from flika.utils.provenance import (
    CommandRecord,
    ProvenanceRecord,
    _hash_file_head,
    export_provenance,
)


class TestCommandRecord:
    def test_defaults(self):
        rec = CommandRecord()
        assert rec.command == ''
        assert rec.duration_seconds == 0.0
        assert rec.input_shape == ()

    def test_with_values(self):
        rec = CommandRecord(command='blur(3)', duration_seconds=1.5,
                           input_window='A', input_shape=(10, 20))
        assert rec.command == 'blur(3)'
        assert rec.duration_seconds == 1.5


class TestProvenanceRecord:
    def test_defaults(self):
        rec = ProvenanceRecord()
        assert rec.software == 'flika'
        assert rec.commands == []

    def test_with_commands(self):
        cmd = CommandRecord(command='open_file("test.tif")')
        rec = ProvenanceRecord(commands=[cmd])
        assert len(rec.commands) == 1


class TestHashFileHead:
    def test_hash_returns_hex(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            f.write(b'test data for hashing')
            path = f.name
        try:
            h = _hash_file_head(path)
            assert len(h) == 64  # SHA256 hex length
            assert all(c in '0123456789abcdef' for c in h)
        finally:
            os.unlink(path)

    def test_hash_nonexistent_file(self):
        h = _hash_file_head('/nonexistent/file.bin')
        assert h == ''


class TestExportProvenance:
    def test_creates_valid_json(self):
        """Test that export_provenance creates a valid JSON file."""
        # Clear any structured records from prior tests
        from ..app.macro_recorder import macro_recorder
        macro_recorder._structured_records.clear()

        # Create a mock window
        class MockWindow:
            image = np.zeros((10, 20), dtype=np.float32)
            filename = ''
            commands = ['open_file("test.tif")', 'gaussian_blur(sigma=2)']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            path = f.name
        try:
            export_provenance(MockWindow(), path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data['software'] == 'flika'
            assert 'commands' in data
            assert len(data['commands']) == 2
            assert data['output_shape'] == [10, 20]
        finally:
            os.unlink(path)
