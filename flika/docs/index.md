# Flika Documentation

**Flika** is a PyQt-based image processing application designed for biologists. It provides
an interactive GUI for loading, processing, and analyzing microscopy image stacks, with
built-in support for single particle tracking, AI-assisted segmentation, and publication-ready
figure composition.

**Version:** 0.3.0
**Python:** 3.12 | **GUI:** PyQt6 (via qtpy) | **Visualization:** pyqtgraph 0.14

---

## Table of Contents

### Getting Started

- [Getting Started](getting_started.md) -- Installation, first launch, opening images, basic navigation
- [User Interface](user_interface.md) -- Main window, menu bar, status bar, image windows, ROIs
- [Keyboard Shortcuts](keyboard_shortcuts.md) -- Complete list of keyboard shortcuts

### Menus and Operations

- [File Operations](file_operations.md) -- Supported file formats, opening, saving, recent files
- [Image Menu](image_menu.md) -- Stacks, alignment, color, measurement, overlays
- [Process Menu](process_menu.md) -- Binary, math, filters, segmentation, detection, colocalization

### Regions of Interest

- [ROI Guide](roi_guide.md) -- ROI types, ROI Manager, measurements, operations

### Advanced Features

- [Single Particle Tracking](spt_guide.md) -- Detection, linking, analysis, classification, batch processing
- [AI Tools](ai_tools.md) -- Pixel classifier, SAM, denoiser, model zoo, Cellpose, StarDist, Claude Live Session, Safety Settings
- [Interoperability](interoperability.md) -- napari bridge, ImageJ bridge, OME export, batch processing
- [Figure Composer](user_interface.md#figure-composer) -- Publication figure layout (covered in UI guide)
- [Microscopy Simulation](simulation.md) -- Synthetic data generation, PSF models, camera noise, benchmarking
- [Microscopy Analysis](analysis.md) -- FRAP, FRET, Calcium imaging, Spectral unmixing, Morphometry

### Extending Flika

- [Plugins](plugins.md) -- Plugin system, installing and writing plugins, info.xml format
- [API Reference](api_reference.md) -- Key classes, scripting console, macro recording

### Support

- [Troubleshooting](troubleshooting.md) -- Common issues, dependency checking, GPU status

---

## Quick Start

```python
# Launch flika
import flika
flika.start_flika()

# Open an image
from flika.process.file_ import open_file
win = open_file('/path/to/image.tif')

# Apply a Gaussian blur
from flika.process.filters import gaussian_blur
blurred = gaussian_blur(2.0)

# Access the current image array
import numpy as np
image_data = win.image  # numpy array (T, X, Y) or (X, Y)
```

## Architecture Overview

Flika is organized around a few core concepts:

- **Window** -- Each image or stack is displayed in a `Window` object containing a pyqtgraph `ImageView`.
- **Global Variables (`g`)** -- `g.win` is the current window, `g.windows` lists all open windows, `g.settings` holds persistent settings.
- **BaseProcess** -- All processing operations inherit from `BaseProcess`, which handles GUI dialogs, undo/redo, and macro recording.
- **ROIs** -- Regions of interest are drawn on windows and can generate time traces, measurements, and masks.
- **Metadata** -- Each window carries a `metadata` dict for file info, SPT data, provenance, and more.

## Recent Highlights

- **Microscopy Simulation** -- Full synthetic data generation pipeline with PSF models,
  camera noise simulation, 10 quick presets, and integrated benchmarking
- **Claude Live Session** -- Interactive multi-turn AI chat for image analysis guidance
  and code generation directly within the application
- **AI Safety Settings** -- Configurable safety controls for AI operations including
  confirmation prompts, resource limits, and audit logging
- **Structure Detection** -- Frangi vesselness, skeletonization, medial axis, Hough
  transforms, corner detection, LBP texture, and structure tensor analysis
- **Microscopy Analysis** -- Dedicated modules for FRAP, FRET, Calcium imaging,
  Spectral unmixing, and Morphometry with publication-ready output
- **Background Subtraction** -- manual ROI, auto-detected ROI, and statistical methods
  (mean/median/mode/percentile) with per-frame or whole-stack options
- **Enhanced Overlays** -- timestamp and scale bar auto-populate from pixel_size and
  frame_interval settings; full customization (colors, fonts, locations, formats)
- **Secure API Key Storage** -- Anthropic API key stored via system keyring, never in
  plaintext; "Delete API Key" button for secure removal
- **Built-in Documentation** -- Help > Documentation opens a searchable manual browser
- **All 26 Settings Verified** -- debug mode, GPU memory limit, and auto-export
  provenance are now fully operational

## User Data Directory

Flika stores user data in `~/.FLIKA/`:

| File/Folder | Purpose |
|---|---|
| `settings.json` | Persistent user settings |
| `plugins/` | Installed plugin packages |
| `models/` | AI model weights |
| `templates/` | Workflow templates |
| `logs/` | Application log files |
| `plugin_guidelines.md` | Guidelines and conventions for plugin development |
