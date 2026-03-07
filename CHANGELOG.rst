0.3.0
-----

Major update with new features and modernized platform support.

**Platform & Modernization:**

* Ported to Python 3.12 / PyQt6 (via qtpy; PyQt5 still supported)
* NumPy 2.0, SciPy 1.15+, pyqtgraph 0.14 compatibility
* Undo / redo system for all image-processing operations
* Structured macro recorder with provenance export
* Dependency checker for optional packages

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
* GPU memory limit setting
* Accelerated filters via CuPy / Numba / Torch

**AI Ecosystem:**

* AI Pixel Classifier, AI Particle Localizer
* SAM (Segment Anything) dialog
* BioImage.IO Model Zoo browser
* Self-supervised denoiser (Noise2Void / CAREamics)
* Cellpose, StarDist, micro-SAM segmentation wrappers
* PSF Simulator
* AI Plugin Generator (Anthropic API)

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

**Publication & Reproducibility:**

* Figure Composer (grid layout, PNG/SVG/PDF export)
* Workflow templates
* Provenance export (JSON, OME companion, REMBI metadata)
* REMBI metadata editor

**Settings:**

* Pixel size and frame interval in Settings
* Default axis order for TIFF import
* Anthropic API key field for AI features
* Acceleration device and GPU memory limit settings


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
