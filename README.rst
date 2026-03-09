flika v0.3.0
============

.. image:: flika/docs/_static/img/flika_screenshot.png
   :alt: flika screenshot

**flika** is an interactive image processing program for biologists written in
Python.  This fork extends the `original flika <https://github.com/flika-org/flika>`_
with modern Python/Qt support, 4-D data handling, GPU acceleration, single-particle
tracking, AI-assisted analysis, microscopy simulation, and many other features while
retaining full backward compatibility with existing flika plugins.

| **Source:** `github.com/gddickinson/flika---Georges-Edition <https://github.com/gddickinson/flika---Georges-Edition>`_
| **Plugins:** `github.com/gddickinson/flika_plugins <https://github.com/gddickinson/flika_plugins>`_


Feature Highlights
------------------

Modernization & Platform
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Python 3.12 / PyQt6** -- ported from Python 3.11 / PyQt5 via ``qtpy``
  (PyQt5 still works as a fallback).
- **NumPy 2.0, SciPy 1.15+, pyqtgraph 0.14** compatibility.
- **Undo / redo** system for all image-processing operations.
- **Structured macro recorder** with provenance export
  (JSON, OME companion, REMBI metadata).
- **Dependency checker** -- warns at startup about missing optional packages
  instead of crashing.
- **Built-in documentation browser** (``Help > Documentation``) with sidebar
  TOC, search, and navigation history.

4-D & Multi-Channel Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **4-D support (T, Z/C, Y, X)** with automatic axis-order detection.
- **per_plane decorator** -- wraps any 2-D filter to operate on every Z/C
  plane, with optional parallelism.
- **Channel Compositor** -- multi-channel overlay with 9 scientific colormaps,
  per-channel LUT, opacity, and additive blending.

I/O & Data Formats
~~~~~~~~~~~~~~~~~~~

- **Format registry** with pluggable handlers: TIFF / OME-TIFF, HDF5, NPY,
  Zarr, OME-Zarr, BioFormats, Imaris (``.ims``), BMP, ND2.
- **Lazy loading** via dask-backed ``LazyArray`` for out-of-core files.
- **Drag-and-drop** file opening.

GPU & Acceleration
~~~~~~~~~~~~~~~~~~~

- Auto-detects **CUDA**, **MPS** (Apple Silicon), or CPU.
- Configurable GPU memory limit -- arrays exceeding the limit stay on CPU.
- Accelerated filters via CuPy, Numba, and PyTorch when available.

AI Ecosystem
~~~~~~~~~~~~~

- **Claude Live Session** -- agentic tool-use for interactive analysis.
- **Script Assistant** -- multi-turn AI chat for scripting help.
- **Plugin Generator** -- generate flika plugins from natural-language
  descriptions with auto-validation, auto-fix, and guideline enforcement.
- **AI Denoiser** -- self-supervised denoising (Noise2Void / CAREamics).
- **Pixel Classifier** -- train and apply pixel-level classifiers.
- **Particle Localizer** -- deep-learning-based particle detection.
- **Segmentation** -- Cellpose, StarDist, micro-SAM wrappers.
- **BioImage.IO Model Zoo** -- browse and load community models.
- **PSF Simulator** -- generate synthetic PSFs for calibration.
- **Safety Settings** -- configurable policy restrictions for AI features.
- **Secure API key storage** via system keyring (never in plaintext).

Single-Particle Tracking (SPT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full SPT pipeline with detection, linking, analysis, classification, and
visualization:

- **Detection** -- U-Track (wavelet + local-max) and ThunderSTORM
  (wavelet/Gaussian/DoG, Gaussian LSQ/MLE/phasor fitters).
- **Linking** -- greedy nearest-neighbour, U-Track LAP (Kalman-filtered
  gap-closing, merge/split), Trackpy adapter.
- **Feature extraction** -- geometric (Rg, asymmetry, fractal dimension),
  kinematic (MSD, velocity, direction), spatial (nearest-neighbour distances),
  autocorrelation.
- **Classification** -- SVM pipeline (Box-Cox + PCA + RBF kernel).
- **Batch processing** -- multi-file pipeline with 6 expert presets.
- **Visualization** -- track overlay, per-track detail window, flower plot,
  all-tracks intensity heatmap, MSD + CDF diffusion analysis,
  scatter/histogram chart dock.
- **I/O** -- ThunderSTORM CSV, flika CSV, JSON.

ROI & Measurement Tools
~~~~~~~~~~~~~~~~~~~~~~~~~

- **ROI Manager** -- dockable panel for organizing, grouping, and
  batch-exporting ROIs.
- **Center-Surround ROI** tool (circle / ellipse / square, configurable
  inner ratio).
- **Colocalization** -- Pearson, Manders, Costes auto-threshold, Li ICQ
  with scatter-plot widget.
- **Watershed segmentation** -- distance-transform + marker-controlled
  watershed.
- ROI histogram, line profile, and volume viewer.

Image Processing
~~~~~~~~~~~~~~~~~

- **Background Subtraction** -- manual ROI, auto-detected ROI, and
  statistical methods (mean/median/mode/percentile).
- **Deconvolution** -- Richardson-Lucy iterative and Wiener
  (frequency-domain) with built-in PSF generation.
- **Image Stitching** -- phase cross-correlation registration with linear
  blending.
- **Bleach Correction** -- exponential fit, histogram matching, ratio-to-mean.
- **Color Conversions** -- RGB to/from HSV, LAB, YCrCb; grayscale
  (luminance, average, lightness).
- **Batch Export** -- TIFF, PNG, or NumPy arrays; stacks as image sequences.
- **Structure Detection** -- Frangi vesselness, skeletonization, Hough
  lines/circles, corner detection (Harris/Shi-Tomasi), LBP texture,
  structure tensor (coherency/orientation).
- **Enhanced Overlays** -- timestamp and scale bar with full customization.

Microscopy Analysis
~~~~~~~~~~~~~~~~~~~~

- **FRAP** -- normalization, single/double exponential fit, Soumpasis
  diffusion model.
- **FRET** -- apparent/corrected efficiency, stoichiometry, histogram.
- **Calcium Imaging** -- dF/F computation, event detection, statistics,
  smoothing.
- **Spectral Unmixing** -- NNLS/least-squares, PCA endmember estimation.
- **Morphometry** -- region properties, Haralick texture features, Hu moments.

Microscopy Simulation
~~~~~~~~~~~~~~~~~~~~~~

Full synthetic data generation for benchmarking and training:

- **PSF models** -- Gaussian, Airy, Born-Wolf, Vectorial, Astigmatic.
- **Camera noise** -- shot noise, read noise; CCD, EMCCD, sCMOS models.
- **Biological structures** -- beads, filaments, cells, organelles, cell
  fields.
- **Fluorophore photophysics** -- 10 presets, blinking and bleaching
  simulation.
- **Optics** -- TIRF, confocal, light-sheet, widefield, SIM, STED.
- **Dynamics** -- Brownian/directed/confined/anomalous diffusion, calcium
  transients/waves, DNA-PAINT binding, FLIM decay.
- **10 named presets** -- beads, TIRF, confocal, PALM/STORM, DNA-PAINT,
  light-sheet, FLIM, calcium, SIM, tracking.
- **Ground truth and benchmarking** -- detection, tracking, segmentation,
  and calcium event metrics.

Publication & Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Figure Composer** -- grid-layout figure builder with PNG / SVG / PDF
  export.
- **Workflow templates** -- save and replay processing pipelines.
- **Provenance export** -- full processing history as JSON with OME-companion
  and REMBI metadata.
- **REMBI metadata editor** for standardized microscopy metadata.

Settings & Usability
~~~~~~~~~~~~~~~~~~~~~

- **26+ settings** accessible via ``File > Settings``.
- **Keyboard shortcuts** -- ``Ctrl+O`` open, ``Ctrl+S`` save, ``Ctrl+W``
  close, ``Ctrl+D`` duplicate, ``Left/Right`` frame navigation,
  ``Home/End`` first/last frame.
- **Plugin Manager** -- enable/disable, download from GitHub, remove
  installed plugins.
- **Debug mode** for detailed diagnostics.


Installation
------------

1. Create a conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    conda create -n flika python=3.12
    conda activate flika

2. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/gddickinson/flika---Georges-Edition.git
    cd flika---Georges-Edition

3. Install in development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    pip install -e .

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~~

Install extras for specific features:

.. code:: bash

    pip install -e ".[ai]"              # AI ecosystem (requires Anthropic API key)
    pip install -e ".[gpu]"             # CuPy GPU acceleration
    pip install -e ".[accel]"           # Numba + PyTorch acceleration
    pip install -e ".[all-formats]"     # h5py, zarr, ome-zarr, aicsimageio
    pip install -e ".[lazy]"            # dask for lazy loading
    pip install -e ".[segmentation]"    # Cellpose, StarDist, micro-SAM
    pip install -e ".[model-zoo]"       # BioImage.IO model zoo
    pip install -e ".[denoising]"       # CAREamics / Noise2Void
    pip install -e ".[spt]"            # Trackpy for particle tracking
    pip install -e ".[dev]"            # pytest, flake8, mypy


Quick Start
-----------

From the command line:

.. code:: bash

    flika

From Python:

.. code:: python

    from flika import *
    start_flika()

Load and process an image:

.. code:: python

    from flika import *
    start_flika()

    # Open a file
    w = open_file('path/to/image.tif')

    # Apply a Gaussian blur
    gaussian_blur(2.0)

    # Threshold
    threshold(50)


Requirements
------------

=========== =============== ==========================
Package     Minimum Version Notes
=========== =============== ==========================
Python      3.10            3.12 recommended
PyQt6       --              or PyQt5 via qtpy
qtpy        2.3             Qt abstraction layer
NumPy       1.24            2.0 compatible
SciPy       1.10            1.15+ recommended
pyqtgraph   0.13            0.14 recommended
pandas      1.5
matplotlib  3.6
scikit-image 0.20
scikit-learn 1.2
tifffile    2022.5.4
=========== =============== ==========================

See ``pyproject.toml`` for the full dependency list and optional extras.


Documentation
-------------

Built-in: **Help > Documentation** in the application.

Original flika docs: `flika-org.github.io <http://flika-org.github.io/>`_


Credits
-------

Original FLIKA
~~~~~~~~~~~~~~

flika was created by **Kyle Ellefsen**, **Brett Settle**, and **Kevin Tarhan**
at the Parker Lab, University of California, Irvine.

If you use flika in your research, please cite:

    Ellefsen, K.L., Bhatt, D., Bhatt, K.A., and Parker, I. (2019).
    "**flika -- a Python-based image-processing and analysis platform for
    fluorescence microscopy.**"
    DOI: `10.1101/2019.12.15.876425 <https://doi.org/10.1101/2019.12.15.876425>`_

This Fork
~~~~~~~~~~

Developed by **George Dickinson** with contributions from
**Claude (Anthropic)** for AI-assisted code generation.


References
----------

**Single-particle tracking:**

- Jaqaman, K., *et al.* (2008).
  "Robust single-particle tracking in live-cell time-lapse sequences."
  *Nature Methods*, 5(8), 695--702.
  DOI: `10.1038/nmeth.1227 <https://doi.org/10.1038/nmeth.1227>`_

- Ovesny, M., *et al.* (2014).
  "ThunderSTORM: a comprehensive ImageJ plug-in for PALM and STORM data
  analysis and super-resolution imaging."
  *Bioinformatics*, 30(16), 2389--2390.
  DOI: `10.1093/bioinformatics/btu202 <https://doi.org/10.1093/bioinformatics/btu202>`_

- Allan, D.B., Caswell, T., Keim, N.C., van der Wel, C.M., and Verweij, R.W.
  (2023). "soft-matter/trackpy." Zenodo.
  DOI: `10.5281/zenodo.7699596 <https://doi.org/10.5281/zenodo.7699596>`_

**Segmentation:**

- Stringer, C., Wang, T., Michaelos, M., and Pachitariu, M. (2021).
  "Cellpose: a generalist algorithm for cellular segmentation."
  *Nature Methods*, 18(1), 100--106.
  DOI: `10.1038/s41592-020-01018-x <https://doi.org/10.1038/s41592-020-01018-x>`_

- Schmidt, U., Weigert, M., Broaddus, C., and Myers, G. (2018).
  "Cell Detection with Star-Convex Polygons."
  *MICCAI 2018*.
  DOI: `10.1007/978-3-030-00934-2_30 <https://doi.org/10.1007/978-3-030-00934-2_30>`_

- Kirillov, A., *et al.* (2023).
  "Segment Anything."
  *ICCV 2023*.
  DOI: `10.1109/ICCV51070.2023.00371 <https://doi.org/10.1109/ICCV51070.2023.00371>`_

**Denoising:**

- Krull, A., Buchholz, T.-O., and Jost, F. (2019).
  "Noise2Void -- Learning Denoising From Single Noisy Images."
  *CVPR 2019*.
  DOI: `10.1109/CVPR.2019.00223 <https://doi.org/10.1109/CVPR.2019.00223>`_

**Model interoperability:**

- Ouyang, W., *et al.* (2022).
  "BioImage Model Zoo: A Community-Driven Resource for Accessible Deep
  Learning in BioImage Analysis."
  *bioRxiv*.
  DOI: `10.1101/2022.06.07.495102 <https://doi.org/10.1101/2022.06.07.495102>`_


License
-------

MIT License. See `LICENSE <LICENSE>`_ for details.
