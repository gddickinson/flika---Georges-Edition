"""Workflow provenance tracking and export for flika.

Captures processing history, input file hashes, software versions,
and timing information to produce reproducible JSON sidecar files.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np

from flika.logger import logger


@dataclass
class CommandRecord:
    """A single processing step in the provenance chain."""
    command: str = ''
    timestamp: str = ''
    duration_seconds: float = 0.0
    input_window: str = ''
    input_shape: tuple = ()
    output_window: str = ''
    output_shape: tuple = ()


@dataclass
class ProvenanceRecord:
    """Full provenance metadata for a window's processing history."""
    software: str = 'flika'
    software_version: str = ''
    python_version: str = ''
    platform: str = ''
    numpy_version: str = ''
    input_file: str = ''
    input_file_hash: str = ''
    commands: List[CommandRecord] = field(default_factory=list)
    output_shape: tuple = ()
    output_dtype: str = ''
    created: str = ''
    exported: str = ''


def _hash_file_head(path: str, nbytes: int = 1024 * 1024) -> str:
    """Compute SHA256 of the first *nbytes* of *path*."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            h.update(f.read(nbytes))
    except (OSError, IOError):
        return ''
    return h.hexdigest()


def build_provenance(window) -> ProvenanceRecord:
    """Build a provenance record from a window's processing history."""
    from flika.version import __version__
    from flika.app.macro_recorder import macro_recorder

    record = ProvenanceRecord(
        software_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        numpy_version=np.__version__,
        output_shape=tuple(window.image.shape),
        output_dtype=str(window.image.dtype),
        created=datetime.now().isoformat(),
    )

    # Input file info
    if hasattr(window, 'filename') and window.filename:
        record.input_file = str(window.filename)
        if os.path.isfile(window.filename):
            record.input_file_hash = _hash_file_head(window.filename)

    # Build command records from structured data if available
    if hasattr(macro_recorder, '_structured_records') and macro_recorder._structured_records:
        for sr in macro_recorder._structured_records:
            record.commands.append(CommandRecord(**sr))
    else:
        # Fall back to window.commands
        for cmd in (window.commands or []):
            record.commands.append(CommandRecord(command=cmd))

    return record


def export_provenance(window, path: str) -> str:
    """Export provenance for *window* as a JSON sidecar file.

    Returns the path of the written file.
    """
    record = build_provenance(window)
    record.exported = datetime.now().isoformat()

    # Convert to serializable dict
    data = asdict(record)
    # Convert tuples to lists for JSON
    for cmd in data.get('commands', []):
        if isinstance(cmd.get('input_shape'), tuple):
            cmd['input_shape'] = list(cmd['input_shape'])
        if isinstance(cmd.get('output_shape'), tuple):
            cmd['output_shape'] = list(cmd['output_shape'])
    if isinstance(data.get('output_shape'), tuple):
        data['output_shape'] = list(data['output_shape'])

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Provenance exported to %s", path)
    return path
