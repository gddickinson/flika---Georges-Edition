"""LLM-powered scripting assistant for flika.

Uses the Anthropic Python SDK to turn natural-language descriptions into
executable flika scripts.

Requires:
    pip install anthropic   (or ``pip install flika[ai]``)

The API key is read from the ``ANTHROPIC_API_KEY`` environment variable or
from ``g.settings['anthropic_api_key']``.
"""
from __future__ import annotations

import os
from typing import Optional

from ..logger import logger

_SYSTEM_PROMPT = """\
You are an expert Python programmer who writes scripts for *flika*, a PyQt-based
image-processing application for biologists.

Key APIs:
  - ``from flika.process.file_ import open_file``
  - ``from flika.process.filters import gaussian_blur, mean_filter, butterworth_filter``
  - ``from flika.process.binary import threshold``
  - ``from flika.process.stacks import zproject, trim, pixel_binning``
  - ``from flika.process.math_ import subtract, multiply, ratio``
  - ``from flika import global_vars as g``
  - ``g.win`` is the current window; ``g.win.image`` is the numpy array.
  - ROIs: ``from flika.roi import makeROI``

Return ONLY valid Python code (no markdown fences).  The script will be
executed inside flika's script editor.
"""


class FlikaAssistant:
    """Natural-language → flika script generator."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic  # noqa: F811
        except ImportError:
            raise ImportError(
                "The anthropic package is required for AI features. "
                "Install with:  pip install anthropic  (or  pip install flika[ai])"
            )
        if api_key is None:
            from .. import global_vars as g
            api_key = os.environ.get("ANTHROPIC_API_KEY") or g.settings.get("anthropic_api_key")
        if not api_key:
            raise ValueError(
                "No API key found.  Set ANTHROPIC_API_KEY or add "
                "'anthropic_api_key' to your flika settings."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate_script(self, description: str) -> str:
        """Turn a natural-language *description* into a flika Python script."""
        logger.info("AI assistant: generating script for %r", description)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": description}],
        )
        return message.content[0].text


def _show_generate_script_dialog():
    """Menu callback: pop up a dialog, call the assistant, open in script editor."""
    from qtpy import QtWidgets
    from .. import global_vars as g

    description, ok = QtWidgets.QInputDialog.getMultiLineText(
        g.m, "AI Script Generator",
        "Describe what you want to do with your image data:"
    )
    if not ok or not description.strip():
        return

    try:
        assistant = FlikaAssistant()
        script = assistant.generate_script(description)
    except Exception as e:
        g.alert(f"AI assistant error: {e}")
        return

    from ..app.script_editor import ScriptEditor
    ScriptEditor.show()
    ScriptEditor.gui.editor.setPlainText(script)
    g.m.statusBar().showMessage("AI script generated — review before running.")
