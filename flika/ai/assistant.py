"""LLM-powered scripting assistant for flika.

Uses the Anthropic Python SDK to turn natural-language descriptions into
executable flika scripts.

Requires:
    pip install anthropic   (or ``pip install flika[ai]``)

The API key is read from the system keyring, the ``ANTHROPIC_API_KEY``
environment variable, or from Edit > Settings.
"""
from __future__ import annotations

import os
from typing import Optional

from ..logger import logger

_SYSTEM_PROMPT = """\
You are an expert Python programmer who writes scripts for *flika*, a PyQt6-based
microscopy image-processing application for biologists.

Source code: {source_url}
Plugin repository: https://github.com/gddickinson/flika_plugins

# Architecture
- flika uses qtpy (bridging PyQt5/PyQt6), pyqtgraph, numpy 2.0, scipy 1.15+
- Images are numpy arrays. Dimensions: 2D=(Y,X), 3D=(T,Y,X), 4D=(T,Y,X,Z)
- Every processing function creates a new Window with the result
- ``g.win`` is the currently selected Window; ``g.win.image`` is its numpy array
- ``g.m.windows`` is the list of all open Window objects
- Metadata is stored in ``window.metadata`` (dict)

# Core Imports & Functions
## File I/O
  from flika.process.file_ import open_file, save_file, save_movie_gui
  open_file('path/to/file.tif')  # opens in a new Window

## Filters (each is a callable object with .gui() for dialog)
  from flika.process.filters import (
      gaussian_blur, mean_filter, median_filter, bilateral_filter,
      butterworth_filter, fourier_filter, wavelet_filter,
      difference_of_gaussians, sobel_filter, laplacian_filter,
      gaussian_laplace_filter, tv_denoise, bleach_correction,
      maximum_filter, minimum_filter, percentile_filter,
      sato_tubeness, meijering_neuriteness, hessian_filter, gabor_filter,
  )
  gaussian_blur(sigma)  # applies to g.win, returns new Window

## Binary / Thresholding
  from flika.process.binary import (
      threshold, adaptive_threshold, hysteresis_threshold,
      multi_otsu_threshold, canny_edge_detector,
      binary_erosion, binary_dilation, remove_small_blobs,
      remove_small_holes, logically_combine, generate_rois,
      grayscale_opening, grayscale_closing, morphological_gradient,
  )
  threshold(value)  # binary threshold on g.win

## Math
  from flika.process.math_ import (
      subtract, multiply, divide, power, sqrt, ratio,
      absolute_value, subtract_trace, divide_trace,
      histogram_equalize, normalize, image_calculator,
  )

## Stacks
  from flika.process.stacks import (
      duplicate, trim, frame_remover, deinterleave, zproject,
      pixel_binning, frame_binning, resize, concatenate_stacks,
      change_datatype, shear_transform, motion_correction,
      generate_random_image, generate_phantom_volume,
  )
  zproject(method='Average')  # methods: Average, Max, Min, Sum, Std

## Segmentation
  from flika.process.segmentation import (
      connected_components, region_properties, clear_border,
      expand_labels, watershed_segmentation, random_walker_seg,
      slic_superpixels, find_boundaries, find_contours_process,
  )

## Detection
  from flika.process.detection import (
      blob_detection_log, blob_detection_doh, peak_local_max,
      template_match, local_maxima_detect, analyze_particles,
  )

## Deconvolution
  from flika.process.deconvolution import richardson_lucy, wiener_deconvolution, generate_psf

## Color
  from flika.process.color import (
      split_channels, blend_channels, convert_color_space, grayscale,
  )

## Dynamics Analysis
  from flika.process.frap import frap_analysis
  from flika.process.fret import fret_analysis
  from flika.process.calcium import calcium_analysis
  from flika.process.spectral import spectral_unmixing

## SPT (Single Particle Tracking)
  from flika.process.spt import spt_analysis, detect_particles, link_particles_process

## Simulation
  from flika.process.simulation import simulate
  simulate.run(preset='Beads - PSF Calibration')  # generates synthetic data

## ROIs
  from flika.roi import makeROI
  roi = makeROI('rectangle', [x, y, w, h])
  trace = roi.getTrace()  # 1D intensity trace as numpy array

## Global Variables
  from flika import global_vars as g
  g.win           # currently selected Window
  g.win.image     # numpy array of current image data
  g.win.name      # window name string
  g.win.metadata  # dict for metadata storage
  g.m             # main application window (FlikaApplication)
  g.m.windows     # list of all Window objects

# Calling Conventions
- Process functions: ``func(param1, param2, keepSourceWindow=False)``
  - Set keepSourceWindow=True to keep the original window open
- GUI dialogs: ``func.gui()`` opens the interactive parameter dialog
- Most functions operate on ``g.win`` (the active window)

# Script Editor Environment
The script runs in an IPython namespace with numpy (np), scipy, pyqtgraph (pg),
and all process functions pre-imported.

Return ONLY valid Python code (no markdown fences). The script will be
executed inside flika's script editor.
"""


def _get_model():
    """Return the AI model from settings, with fallback."""
    try:
        from .. import global_vars as g
        return g.settings['ai_model'] or 'claude-sonnet-4-20250514'
    except Exception:
        return 'claude-sonnet-4-20250514'


def _get_source_url():
    """Return the flika source URL from settings."""
    try:
        from .. import global_vars as g
        return (g.settings['flika_source_url']
                or 'https://github.com/gddickinson/flika---Georges-Edition')
    except Exception:
        return 'https://github.com/gddickinson/flika---Georges-Edition'


class FlikaAssistant:
    """Natural-language → flika script generator."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            import anthropic  # noqa: F811
        except ImportError:
            raise ImportError(
                "The anthropic package is required for AI features. "
                "Install with:  pip install anthropic  (or  pip install flika[ai])"
            )
        if api_key is None:
            from ..app.settings_editor import get_api_key
            api_key = get_api_key()
        if not api_key:
            raise ValueError(
                "No API key found.  Set ANTHROPIC_API_KEY environment variable "
                "or enter your key in Edit > Settings."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or _get_model()

    def generate_script(self, description: str) -> str:
        """Turn a natural-language *description* into a flika Python script."""
        logger.info("AI assistant: generating script for %r", description)
        system = _SYSTEM_PROMPT.format(source_url=_get_source_url())
        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": description}],
        )
        return message.content[0].text

    def generate_with_history(self, messages: list[dict]) -> str:
        """Generate a script using multi-turn conversation history."""
        logger.info("AI assistant: generating with %d messages", len(messages))
        system = _SYSTEM_PROMPT.format(source_url=_get_source_url())
        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        return message.content[0].text


def _show_generate_script_dialog():
    """Menu callback for AI > Generate Script."""
    from .assistant_dialog import AIAssistantDialog
    AIAssistantDialog.show_dialog()
