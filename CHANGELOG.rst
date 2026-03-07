0.3.0
-----

Major update with new features and modernized platform support.

**Platform & Modernization:**

* Ported to Python 3.12 / PyQt6 (via qtpy; PyQt5 still supported)
* NumPy 2.0, SciPy 1.15+, pyqtgraph 0.14 compatibility
* Undo / redo system for all image-processing operations
* Structured macro recorder with provenance export
* Dependency checker for optional packages
* Built-in documentation browser (Help > Documentation) with 14 user manual
  pages, sidebar TOC, search, and navigation history
* Plugin info.xml dependency declarations for automated dependency checking

**4-D & Multi-Channel:**

* 4-D data support (T, Z/C, Y, X) with axis-order detection
* per_plane decorator for broadcasting 2-D filters across planes
* Channel Compositor with 9 scientific colormaps and additive blending

**I/O & Data Formats:**

* Format registry with pluggable handlers (TIFF, HDF5, NPY, Zarr, OME-Zarr,
  BioFormats, Imaris, BMP, ND2)
* Dask-backed lazy loading for out-of-core files
* Drag-and-drop file opening

**GPU & Acceleration:**

* Device abstraction (auto-detect CUDA / MPS / CPU)
* GPU memory limit setting (arrays exceeding limit stay on CPU)
* Accelerated filters via CuPy / Numba / Torch

**AI Ecosystem:**

* AI Pixel Classifier, AI Particle Localizer
* SAM (Segment Anything) dialog
* BioImage.IO Model Zoo browser
* Self-supervised denoiser (Noise2Void / CAREamics)
* Cellpose, StarDist, micro-SAM segmentation wrappers
* PSF Simulator
* AI Plugin Generator (Anthropic API)
* Secure API key storage via system keyring (macOS Keychain, Windows Credential
  Manager, Linux Secret Service) -- never stored in plaintext settings
* "Delete API Key" button for secure credential removal

**Single-Particle Tracking:**

* Full SPT pipeline: detection (U-Track, ThunderSTORM), linking (greedy,
  U-Track LAP, Trackpy), feature extraction, SVM classification
* ParticleData model (pandas DataFrame-backed)
* Results Table with sortable columns and CSV export
* 6-tab SPT Control Panel
* Track overlay, per-track detail window, flower plot, all-tracks heatmap,
  MSD/CDF diffusion analysis, chart dock
* Batch processing with 6 expert presets
* ThunderSTORM CSV, flika CSV, JSON I/O

**ROI & Measurement:**

* ROI Manager dockable panel
* Center-Surround ROI tool
* Colocalization analysis (Pearson, Manders, Costes, Li ICQ)
* Watershed segmentation
* ROI histogram and line-profile viewers
* Volume viewer for 3-D rendering

**Image Processing:**

* Background Subtraction (Process > Background Subtraction) with three methods:
  manual ROI, auto-detected ROI (dark-corner algorithm), and statistical
  (mean/median/mode/percentile); per-frame or whole-stack
* Enhanced Timestamp overlay: auto-populates from frame_interval setting;
  font size, 8 preset + custom colors, background colors, 4-corner placement,
  bold text, show frame number, custom format strings with placeholders
* Enhanced Scale Bar overlay: auto-populates from pixel_size setting; linked
  physical/pixel width, unit selection (um/nm/mm/px), bar thickness, separate
  bar and label colors, bold text, nice-number rounding, offset controls

**Publication & Reproducibility:**

* Figure Composer (grid layout, PNG/SVG/PDF export)
* Workflow templates
* Provenance export (JSON, OME companion, REMBI metadata)
* Auto-export provenance setting (saves sidecar JSON on each file save)
* REMBI metadata editor

**Settings:**

* 26 settings, all verified operational
* Pixel size and frame interval in Settings
* Default axis order for TIFF import
* Debug mode now toggles logger to DEBUG level
* GPU memory limit enforced in should_use_gpu()
* Auto-export provenance setting with UI checkbox
* Secure Anthropic API key field with keyring storage


0.2.17
------

* New documentation (roi, tracefig, app.utils)
* updated test_images

0.2.16
------

* fixed plugin update icon
* fixed check_updates and added update_flika
* added requests dependency for logging to REST API
* updated tifffile to 2017 version

0.2.1
-----

* added docs
