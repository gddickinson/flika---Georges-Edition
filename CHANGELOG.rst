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
* Keyboard shortcuts: Ctrl+O/S/W/D/G/T, Left/Right/Home/End frame navigation

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
* Claude Live Session -- agentic mode with tool-use API (execute_code,
  get_window_info, list_windows, get_image_stats, get_action_log), action
  logging with export-as-script
* Claude Script Assistant -- multi-turn chat with context-aware code
  generation, Insert/Run/Copy buttons, collapsible instructions
* Claude Plugin Generator -- preview pane, structural validation (AST-based),
  auto-fix for missing module instance and self.start(), plugin guidelines file
  (~/.FLIKA/plugin_guidelines.md), auto-reload after save (no restart needed)
* AI Safety Settings (AI > Claude > Safety Settings) -- configurable policy
  restrictions for file read/write, network, subprocess, OS access; approval
  dialog; safety policy injected into all AI system prompts
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

**Microscopy Analysis:**

* FRAP analysis (normalization, single/double exponential fit, Soumpasis
  diffusion model)
* FRET analysis (apparent/corrected efficiency, stoichiometry, histogram)
* Calcium imaging (dF/F calculation, event detection, statistics, smoothing)
* Spectral unmixing (NNLS/least-squares, PCA endmember estimation)
* Morphometry (region properties, Haralick texture, Hu moments)

**Structure Detection:**

* Frangi vesselness filter
* Skeletonization and medial axis
* Skeleton graph analysis (branch points, endpoints, lengths)
* Hough line and circle detection
* Corner detection (Harris, Shi-Tomasi)
* LBP texture classification
* Structure tensor (coherency, orientation)

**Microscopy Simulation:**

* Full synthetic microscopy data generation package
* PSF models: Gaussian, Airy, Born-Wolf, Vectorial, Astigmatic
* Camera noise models: CCD, EMCCD, sCMOS with realistic photon counting
* Biological structures: beads, filaments, cells, organelles, cell fields
* Fluorophore photophysics: 10 presets, blinking/bleaching simulation
* Optical modalities: TIRF, confocal, light-sheet, widefield, SIM, STED
* Dynamics: Brownian/directed/confined/anomalous/switching diffusion, calcium
  transients/waves, DNA-PAINT binding, FLIM decay
* 10 named presets, 4-tab Simulation Builder dialog
* Ground truth storage in window.metadata['simulation']
* Evaluation metrics: detection (Hungarian matching), tracking (MOTA/MOTP),
  segmentation (Jaccard/Dice), calcium events
* 8 registered benchmarks with Benchmark Dialog

**ROI & Measurement:**

* ROI Manager dockable panel
* Center-Surround ROI tool
* Colocalization analysis (Pearson, Manders, Costes, Li ICQ)
* Watershed segmentation
* ROI histogram and line-profile viewers
* Volume viewer for 3-D rendering

**Image Processing:**

* Deconvolution: Richardson-Lucy iterative, Wiener frequency-domain,
  Gaussian/Airy PSF generation
* Image stitching: phase cross-correlation, linear blending, 2D/3D support
* Bleach correction: exponential fit, histogram matching, ratio-to-mean
* Color conversions: RGB to/from HSV/LAB/YCrCb, grayscale
  (luminance/average/lightness)
* Batch export: TIFF/PNG/NPY, image sequences
* Background Subtraction (Process > Background Subtraction) with three methods:
  manual ROI, auto-detected ROI (dark-corner algorithm), and statistical
  (mean/median/mode/percentile); per-frame or whole-stack
* Enhanced Timestamp overlay: auto-populates from frame_interval setting;
  font size, 8 preset + custom colors, background colors, 4-corner placement,
  bold text, show frame number, custom format strings with placeholders
* Enhanced Scale Bar overlay: auto-populates from pixel_size setting; linked
  physical/pixel width, unit selection (um/nm/mm/px), bar thickness, separate
  bar and label colors, bold text, nice-number rounding, offset controls

**Plugin Manager:**

* Enable/disable individual plugins
* Suppress startup messages
* GitHub repository browser for discovering plugins
* Thread-safe output filtering
* Reload Plugins button (re-scan without restart)

**Publication & Reproducibility:**

* Figure Composer (grid layout, PNG/SVG/PDF export)
* Workflow templates
* Provenance export (JSON, OME companion, REMBI metadata)
* Auto-export provenance setting (saves sidecar JSON on each file save)
* REMBI metadata editor

**Performance:**

* Removed dead code paths
* Cached scatter brushes, pre-allocated getScatterPts()
* RedrawPartialThread replaced 50ms busy-loop with signal-driven updates

**Settings:**

* 26 settings, all verified operational
* Pixel size and frame interval in Settings
* Default axis order for TIFF import
* Debug mode now toggles logger to DEBUG level
* GPU memory limit enforced in should_use_gpu()
* Auto-export provenance setting with UI checkbox
* Secure Anthropic API key field with keyring storage

**Test Suite:**

* 1296 tests passing (5 skipped)
* Correctness tests for all process functions
* Structure detection tests
* Benchmark evaluation tests


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
