# API Reference

This document covers the key classes and APIs for scripting flika from the embedded
console or external scripts.

## Global Variables (`g`)

The `flika.global_vars` module (imported as `g`) provides access to application state:

```python
from flika import global_vars as g
```

| Variable | Type | Description |
|---|---|---|
| `g.m` | `FlikaApplication` | The main application window |
| `g.win` | `Window` or `None` | The currently selected window |
| `g.windows` | `list[Window]` | All open windows |
| `g.currentWindow` | `Window` or `None` | Alias for `g.win` |
| `g.traceWindows` | `list[TraceFig]` | All open trace figure windows |
| `g.currentTrace` | `TraceFig` or `None` | The currently selected trace |
| `g.settings` | `Settings` | Persistent settings dictionary |
| `g.clipboard` | varies | Internal clipboard |
| `g.menus` | `list[QMenu]` | Image and Process menus |
| `g.dialogs` | `list` | Open dialog references |
| `g.headless` | `bool` | True when running without GUI |

### Settings

`g.settings` behaves like a dictionary. Changes are automatically saved to
`~/.FLIKA/settings.json`.

```python
# Read a setting
dtype = g.settings['internal_data_type']  # 'float64'

# Change a setting
g.settings['roi_color'] = '#00ff00'

# Batch changes (single disk write)
with g.settings.batch():
    g.settings['point_size'] = 8
    g.settings['point_color'] = '#ff0000'
```

Key settings:

| Key | Default | Description |
|---|---|---|
| `internal_data_type` | `'float64'` | Default dtype for images |
| `multiprocessing` | `True` | Enable parallel processing |
| `nCores` | CPU count | Number of worker cores |
| `mousemode` | `'rectangle'` | Current ROI drawing mode |
| `roi_color` | `'#ffff00'` | Default ROI color |
| `point_color` | `'#ff0000'` | Default point ROI color |
| `point_size` | `5` | Point marker size |
| `pixel_size` | `108.0` | Pixel size in nm |
| `frame_interval` | `0.05` | Frame interval in seconds |
| `acceleration_device` | `'Auto'` | GPU/CPU selection |
| `debug_mode` | `False` | Enable verbose logging |

### Alerts and Messages

```python
g.alert("Something happened!")  # Shows a popup dialog
g.status_msg("Processing...")   # Shows in the status bar
```

## Window Class

`flika.window.Window` is the central class. Each image is displayed in a Window.

### Creating Windows

```python
from flika.window import Window
import numpy as np

# From a numpy array
data = np.random.randn(100, 256, 256).astype('float32')
win = Window(data, name='My Image')

# From a file
from flika.process.file_ import open_file
win = open_file('/path/to/image.tif')
```

### Constructor

```python
Window(tif, name='flika', filename='', commands=None, metadata=None)
```

| Parameter | Type | Description |
|---|---|---|
| `tif` | `numpy.ndarray` | Image array: (X, Y), (T, X, Y), or (T, X, Y, Z) |
| `name` | `str` | Display name |
| `filename` | `str` | Source file path |
| `commands` | `list[str]` | Command history |
| `metadata` | `dict` | Metadata dictionary |

### Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `image` | `ndarray` | The image array (T, X, Y) or (X, Y) |
| `volume` | `ndarray` or `None` | 4D array (T, X, Y, Z) if present |
| `name` | `str` | Window display name |
| `filename` | `str` | Source file path |
| `metadata` | `dict` | Metadata dictionary |
| `rois` | `list[ROI_Base]` | ROIs drawn on this window |
| `currentIndex` | `int` | Current frame index |
| `mx`, `my` | `int` | Image dimensions (width, height) |
| `mt` | `int` | Number of frames |
| `imageview` | `ImageView` | The pyqtgraph ImageView widget |
| `commands` | `list[str]` | Operation history |

### Key Methods

```python
# Close the window
win.close()

# Navigate frames
win.setIndex(frame_number)

# Toggle views
win.toggleOrthogonalViews()
win.toggleVolumeViewer()
```

### Signals

| Signal | Emitted When |
|---|---|
| `closeSignal` | Window is closed |
| `sigTimeChanged(int)` | Frame index changes |
| `sigSliceChanged` | Z/C/D4 slider changes |
| `sigROICreated(object)` | An ROI is added |
| `sigROIRemoved(object)` | An ROI is removed |
| `gainedFocusSignal` | Window gains focus |
| `lostFocusSignal` | Window loses focus |
| `keyPressSignal(QEvent)` | Key pressed in window |

## BaseProcess

All processing operations inherit from `BaseProcess`. It handles GUI creation, undo/redo,
and macro recording.

### Writing a Custom Operation

```python
from flika.utils.BaseProcess import BaseProcess, SliderLabel
from flika import global_vars as g
import numpy as np

class InvertImage(BaseProcess):
    def __init__(self):
        super().__init__()

    def __call__(self, keepSourceWindow=False):
        self.start(keepSourceWindow)
        self.newtif = g.win.image.max() - g.win.image
        self.newname = f'{g.win.name} - Inverted'
        return self.end()

    def gui(self):
        self.gui_reset()
        super().gui()

invert = InvertImage()

# Use it
result_window = invert()
```

### BaseProcess Flow

1. `gui()` -- builds and shows the dialog
2. `__call__(**params)` -- executes the operation
3. `start(keepSourceWindow)` -- snapshots state for undo
4. Set `self.newtif` (result array) and `self.newname` (result name)
5. `end()` -- creates the result window, records macro, pushes undo

## Process Module Functions

All process functions are callable instances of `BaseProcess` subclasses. They can be
called from the console, scripts, or macros. Every function accepts a `keepSourceWindow`
keyword argument (default `False`); when `False`, the source window is replaced.

### Filters (`flika.process.filters`)

```python
from flika.process.filters import (
    gaussian_blur, median_filter, bilateral_filter, tv_denoise,
    bleach_correction, mean_filter, variance_filter,
    difference_of_gaussians, butterworth_filter, fourier_filter,
    wavelet_filter, sobel_filter, laplacian_filter,
    gaussian_laplace_filter, gaussian_gradient_magnitude_filter,
    sato_tubeness, meijering_neuriteness, hessian_filter,
    gabor_filter, maximum_filter, minimum_filter, percentile_filter,
    flash_remover, difference_filter, boxcar_differential_filter,
)
```

Key signatures:

```python
gaussian_blur(sigma, norm_edges=False, keepSourceWindow=False)
median_filter(nFrames, keepSourceWindow=False)
bilateral_filter(soft, beta, width, stoptol, maxiter, keepSourceWindow=False)
tv_denoise(weight, keepSourceWindow=False)
bleach_correction(method='Exponential Fit', keepSourceWindow=False)
mean_filter(nFrames, keepSourceWindow=False)
variance_filter(nFrames, keepSourceWindow=False)
difference_of_gaussians(sigma1, sigma2, keepSourceWindow=False)
butterworth_filter(filter_order, low, high, framerate=0, keepSourceWindow=False)
fourier_filter(frame_rate, low, high, loglogPreview, keepSourceWindow=False)
wavelet_filter(low, high, keepSourceWindow=False)
sobel_filter(keepSourceWindow=False)
laplacian_filter(ksize=3, keepSourceWindow=False)
gaussian_laplace_filter(sigma, keepSourceWindow=False)
gaussian_gradient_magnitude_filter(sigma, keepSourceWindow=False)
sato_tubeness(sigma_min, sigma_max, black_ridges=False, keepSourceWindow=False)
meijering_neuriteness(sigma_min, sigma_max, black_ridges=False, keepSourceWindow=False)
hessian_filter(sigma_min, sigma_max, black_ridges=False, keepSourceWindow=False)
gabor_filter(frequency, theta, keepSourceWindow=False)
maximum_filter(size, keepSourceWindow=False)
minimum_filter(size, keepSourceWindow=False)
percentile_filter(percentile, size, keepSourceWindow=False)
```

### Binary / Morphology (`flika.process.binary`)

```python
from flika.process.binary import (
    threshold, adaptive_threshold, canny_edge_detector,
    logically_combine, remove_small_blobs, binary_dilation,
    binary_erosion, generate_rois, analyze_particles,
    grayscale_opening, grayscale_closing, morphological_gradient,
    h_maxima, h_minima, area_opening, area_closing,
    remove_small_holes, flood_fill_process,
    hysteresis_threshold, multi_otsu_threshold,
)
```

Key signatures:

```python
threshold(value, darkBackground=False, keepSourceWindow=False)
adaptive_threshold(value, block_size, darkBackground=False, keepSourceWindow=False)
canny_edge_detector(sigma, keepSourceWindow=False)
logically_combine(window1, window2, operator, keepSourceWindow=False)
remove_small_blobs(rank, value, keepSourceWindow=False)
binary_dilation(rank, connectivity, iterations, keepSourceWindow=False)
binary_erosion(rank, connectivity, iterations, keepSourceWindow=False)
analyze_particles(min_area=1, max_area=0, keepSourceWindow=False)
hysteresis_threshold(low, high, keepSourceWindow=False)
multi_otsu_threshold(classes, keepSourceWindow=False)
```

### Math (`flika.process.math_`)

```python
from flika.process.math_ import (
    subtract, multiply, divide, ratio, power, sqrt,
    absolute_value, normalize, histogram_equalize,
    subtract_trace, divide_trace,
)
```

Key signatures:

```python
subtract(value, keepSourceWindow=False)
multiply(value, keepSourceWindow=False)
divide(value, keepSourceWindow=False)
ratio(first_frame, nFrames, ratio_type, black_level=0, keepSourceWindow=False)
power(value, keepSourceWindow=False)
sqrt(keepSourceWindow=False)
absolute_value(keepSourceWindow=False)
normalize(method='Min-Max (0-1)', keepSourceWindow=False)
histogram_equalize(nbins=256, keepSourceWindow=False)
```

### Segmentation (`flika.process.segmentation`)

```python
from flika.process.segmentation import connected_components
```

### Deconvolution (`flika.process.deconvolution`)

```python
from flika.process.deconvolution import (
    richardson_lucy, wiener_deconvolution, generate_psf,
)

richardson_lucy(psf_sigma, psf_size=11, iterations=20, keepSourceWindow=False)
wiener_deconvolution(psf_sigma, psf_size=11, noise_variance=0.01, keepSourceWindow=False)
generate_psf(psf_type='Gaussian', size=21, sigma_or_radius=3.0, keepSourceWindow=False)
```

### Stitching (`flika.process.stitching`)

```python
from flika.process.stitching import stitch_images

stitch_images(window1, window2, direction='Horizontal', overlap=0.1, ...)
```

Stitches two windows together using phase cross-correlation alignment with
configurable blend width.

## Simulation API

The simulation package generates synthetic microscopy data with known ground truth,
useful for benchmarking detection, tracking, and segmentation algorithms.

### Quick Start with Presets

```python
from flika.process.simulation import simulate

# Run a named preset (opens a Window with the result)
w = simulate.run(preset='Beads - PSF Calibration')

# Access ground truth stored in window metadata
gt = w.metadata['simulation']
positions = gt['positions']        # (N, 3) true emitter positions
```

### Available Presets

```python
from flika.simulation.presets import PRESETS

# PRESETS is a dict mapping name -> SimulationConfig
print(list(PRESETS.keys()))
```

| Preset Name | Modality | Description |
|---|---|---|
| `'Beads - PSF Calibration'` | widefield | 256x256, 100 beads, sCMOS |
| `'TIRF - Single Molecules'` | tirf | 512x512, 100 frames, EMCCD |
| `'Confocal - Cell with Organelles'` | confocal | 512x512, 30 z-slices |
| `'PALM/STORM - Dense Filaments'` | smlm | 256x256, 5000 frames |
| `'DNA-PAINT - Grid Pattern'` | dnapaint | 256x256, 2000 frames |
| `'Light-Sheet - Cell Spheroid'` | lightsheet | Large volume |
| `'FLIM - Two Lifetimes'` | widefield | Fluorescence lifetime |
| `'Calcium - Neuron Spikes'` | widefield | Calcium transients |
| `'SIM - Resolution Target'` | sim | Structured illumination |
| `'Tracking - Brownian Particles'` | widefield | Brownian motion tracks |

### Custom Simulation

```python
from flika.simulation.engine import SimulationEngine, SimulationConfig
from flika.simulation.noise import CameraConfig
from flika.simulation.fluorophores import FLUOROPHORE_PRESETS

# Build a custom configuration
config = SimulationConfig(
    nx=256, ny=256, nt=100,
    modality='tirf',
    structure_type='beads',
    structure_params={'n_beads': 200, 'bead_radius': 0.1},
    fluorophore='Alexa647',
    camera=CameraConfig(type='EMCCD', em_gain=300),
)

engine = SimulationEngine(config)
stack, metadata = engine.run()
```

**SimulationConfig fields:**

| Field | Default | Description |
|---|---|---|
| `nx`, `ny` | 256 | Image width/height in pixels |
| `nz` | 1 | Number of z-slices (1 for 2D) |
| `nt` | 1 | Number of time frames |
| `pixel_size` | 0.1 | Pixel size in um |
| `z_step` | 0.3 | Z step in um |
| `dt` | frame interval | Frame interval in seconds |
| `modality` | `'widefield'` | Microscope modality |
| `wavelength` | 680.0 | Emission wavelength in nm |
| `NA` | 1.4 | Numerical aperture |
| `psf_model` | `'gaussian'` | PSF model name |
| `structure_type` | `'beads'` | Sample structure type |
| `structure_params` | `{}` | Structure-specific parameters |
| `fluorophore` | `'Alexa647'` | Fluorophore preset name or `'custom'` |
| `camera` | `CameraConfig()` | Camera noise model |
| `motion_type` | `'static'` | Particle dynamics type |
| `motion_params` | `{}` | Motion-specific parameters |
| `modality_params` | `{}` | Modality-specific parameters |

**CameraConfig fields:** `type` (`'sCMOS'`, `'EMCCD'`, `'CCD'`), `read_noise`,
`quantum_efficiency`, `em_gain`, `dark_current`, `baseline`.

**FLUOROPHORE_PRESETS:** `'Alexa488'`, `'Alexa647'`, `'EGFP'`, `'mCherry'`,
`'Atto655'`, `'Cy3'`, `'Cy5'`, `'TMR'`, `'DAPI'`, `'Hoechst'`.

### Ground Truth

After simulation, ground truth data is stored in `window.metadata['simulation']`:

| Key | Type | Description |
|---|---|---|
| `positions` | `(N, 3)` array | True emitter positions [x, y, z] |
| `positions_per_frame` | dict | Frame-indexed position arrays |
| `track_dict` | dict | `{track_id: (N, 3) array [frame, x, y]}` |
| `binary_mask` | ndarray | Ground truth binary segmentation |
| `segmentation_mask` | ndarray | Labeled segmentation mask |
| `diffusion_coefficients` | array | Per-particle D values |
| `calcium_trace` | array | True calcium signal |
| `calcium_spike_frames` | array | True spike frame indices |
| `emitter_states` | array | Per-frame fluorophore on/off states |

### Benchmarks

```python
from flika.simulation.benchmarks import run_benchmark, run_all_benchmarks
from flika.simulation.evaluation import BenchmarkResult

result = run_benchmark('detection_snr_sweep')
print(result.metrics)

# Run all 8 registered benchmarks
results = run_all_benchmarks()
```

## Single Particle Tracking (SPT) API

The SPT package provides detection, linking, feature computation, classification,
and batch processing for single-molecule and particle tracking experiments.

### Detection

```python
from flika.spt.detection.utrack_detector import UTrackDetector
from flika.spt.detection.thunderstorm import ThunderSTORMDetector

# U-Track statistical detection
detector = UTrackDetector(psf_sigma=1.5, alpha=0.05, min_intensity=0.0)
locs = detector.detect_stack(image_stack)   # (M, 4) array [frame, x, y, intensity]

# Single-frame detection
frame_locs = detector.detect_frame(image_2d)  # (N, 3) array [x, y, intensity]

# ThunderSTORM-style detection
ts_detector = ThunderSTORMDetector(sigma=1.5, threshold=3.0)
```

### Linking

```python
from flika.spt.linking.utrack_linker import UTrackLinker
from flika.spt.linking.greedy_linker import greedy_linker
from flika.spt.linking.trackpy_linker import trackpy_linker

# LAP-based linker with Kalman prediction
linker = UTrackLinker(max_distance=10.0, max_gap=5, min_track_length=3,
                       motion_model='mixed')
tracks = linker.link(locs)  # dict: {track_id: (N, 3) array [frame, x, y]}

# Simple greedy linker
tracks = greedy_linker(locs, max_distance=5.0)
```

### Feature Computation

```python
from flika.spt.features.feature_calculator import FeatureCalculator

calc = FeatureCalculator(pixel_size=108.0, frame_interval=1.0,
                         enable_geometric=True, enable_kinematic=True,
                         enable_spatial=True)

# From a DataFrame with columns: track_number, frame, x, y, intensity
features_df = calc.compute_all_from_df(tracks_df)

# From raw arrays
features_df = calc.compute_all(locs_array, tracks)
```

**Feature categories:**

| Module | Features |
|---|---|
| `geometric` | Radius of gyration, asymmetry, fractal dimension, bounding ellipse |
| `kinematic` | MSD, velocity autocorrelation, direction changes, instantaneous speed |
| `spatial` | Nearest-neighbor distances, spatial density |
| `autocorrelation` | Intensity and displacement autocorrelation |

### Classification

```python
from flika.spt.classification.svm_classifier import SVMClassifier

classifier = SVMClassifier()  # Box-Cox + PCA + RBF SVM pipeline
classifier.fit(features_df, labels)
predictions = classifier.predict(new_features_df)
```

### Batch Processing

```python
from flika.spt.batch.batch_pipeline import batch_pipeline
from flika.spt.batch.expert_configs import EXPERT_CONFIGS

# 6 expert presets for common experiments
print(list(EXPERT_CONFIGS.keys()))

# Run batch pipeline on multiple files
results = batch_pipeline(file_list, config=EXPERT_CONFIGS['single_molecule'])
```

### SPT I/O

```python
from flika.spt.io.spt_formats import (
    load_thunderstorm_csv, save_thunderstorm_csv,
    load_flika_csv, save_flika_csv,
    load_tracks_json, save_tracks_json,
)
```

### Process Entry Points

```python
from flika.process.spt import spt_analysis, detect_particles, link_particles_process

# Full pipeline via GUI or scripting
spt_analysis()

# Individual steps
detect_particles()
link_particles_process()
```

Track data is stored in `window.metadata['spt']` as a dict:
`{track_id: (N, 3) array [frame, x, y]}`.

## Microscopy Analysis APIs

Specialized analysis modules for common microscopy experiments. All are accessible
via the Process > Dynamics menu.

### Calcium Imaging (`flika.process.calcium`)

```python
from flika.process.calcium import (
    compute_dff, compute_dff_image,
    detect_calcium_events, compute_calcium_stats,
    smooth_trace,
)

# Compute dF/F from a 1D trace
dff = compute_dff(trace, baseline_frames=range(0, 50), baseline_method='mean')

# Compute dF/F for entire image stack
dff_stack = compute_dff_image(stack, baseline_frames=range(0, 50))

# Detect events in a dF/F trace
events = detect_calcium_events(dff_trace, threshold=2.0, min_duration=3)

# Compute statistics for detected events
stats = compute_calcium_stats(events)

# Smooth a trace
smoothed = smooth_trace(trace, method='savgol', window_length=11, polyorder=3)
```

The `CalciumAnalysis` BaseProcess class provides the GUI interface.

### FRAP (`flika.process.frap`)

```python
from flika.process.frap import frap_analysis

# Runs FRAP normalization and curve fitting
# Supports single/double exponential and Soumpasis diffusion models
frap_analysis()
```

### FRET (`flika.process.fret`)

```python
from flika.process.fret import fret_efficiency

# Computes apparent/corrected FRET efficiency and stoichiometry
fret_efficiency()
```

### Spectral Unmixing (`flika.process.spectral`)

```python
from flika.process.spectral import spectral_unmixing

# NNLS or least-squares unmixing with optional PCA endmember estimation
spectral_unmixing()
```

### Morphometry (`flika.process.morphometry`)

```python
from flika.process.morphometry import region_properties, haralick_texture

# Compute region properties (area, perimeter, eccentricity, etc.)
region_properties()

# Compute Haralick texture features and Hu moments
haralick_texture()
```

### Structure Detection (`flika.process.structures`)

```python
from flika.process.structures import (
    frangi_vesselness, skeletonize, hough_lines, hough_circles,
    corner_detection, local_binary_pattern, structure_tensor_analysis,
    medial_axis, skeleton_analysis,
)
```

Available via Process > Structures menu:

| Function | Description |
|---|---|
| `frangi_vesselness` | Frangi vesselness filter for tubular structures |
| `skeletonize` | Morphological skeletonization |
| `medial_axis` | Medial axis transform |
| `skeleton_analysis` | Graph extraction from skeleton (branch points, endpoints, segments) |
| `hough_lines` | Hough line detection |
| `hough_circles` | Hough circle detection |
| `corner_detection` | Harris or Shi-Tomasi corner detection |
| `local_binary_pattern` | LBP texture descriptor |
| `structure_tensor_analysis` | Coherency and orientation from structure tensor |

## I/O Registry

The format registry provides a pluggable system for reading and writing image files.

```python
from flika.io.registry import FormatRegistry, FormatHandler

registry = FormatRegistry()
```

**Supported formats:** TIFF (`.tif`, `.tiff`), HDF5 (`.h5`, `.hdf5`), NumPy (`.npy`),
Zarr (`.zarr`), OME-Zarr (`.ome.zarr`), BioFormats, Imaris (`.ims`), BMP (`.bmp`).

### Reading Files

```python
array, metadata = registry.read('/path/to/image.tif')
```

### Writing Files

```python
registry.write('/path/to/output.tif', data, metadata=metadata)
```

### Custom Format Handler

```python
from flika.io.registry import FormatHandler

@registry.register('.xyz')
class XyzHandler(FormatHandler):
    extensions = ['.xyz']

    def read(self, path):
        # Return (ndarray, dict)
        ...

    def write(self, path, data, metadata=None):
        ...
```

## Provenance

Flika records processing provenance for reproducibility. Each window's command history,
input file hash, and software versions are captured.

```python
from flika.utils.provenance import export_provenance, build_provenance

# Export provenance as a JSON sidecar file
path = export_provenance(window, '/path/to/provenance.json')

# Build a ProvenanceRecord without writing to disk
record = build_provenance(window)
print(record.commands)       # List of CommandRecord objects
print(record.input_file)     # Original file path
print(record.input_file_hash)  # SHA256 of first 1MB
```

**ProvenanceRecord fields:** `software`, `software_version`, `python_version`,
`platform`, `numpy_version`, `input_file`, `input_file_hash`, `commands`,
`output_shape`, `output_dtype`, `created`, `exported`.

**CommandRecord fields:** `command`, `timestamp`, `duration_seconds`,
`input_window`, `input_shape`, `output_window`, `output_shape`.

## AI Safety API

The AI safety module provides code validation for AI-generated code (used by the
Script Assistant, Live Session, and Plugin Generator).

```python
from flika.ai.safety import check_code_safety, get_policy_summary

# Validate AI-generated code against current safety settings
is_safe, warnings = check_code_safety(code_string)
# is_safe: bool -- False if code contains blocked patterns
# warnings: list[str] -- human-readable descriptions of violations

# Get a human-readable summary of current AI permissions
policy = get_policy_summary()
```

### Safety Settings

Controlled via `g.settings` (and the Claude > Safety Settings dialog):

| Setting Key | Default | Description |
|---|---|---|
| `ai_allow_file_write` | `False` | Allow AI code to write files |
| `ai_allow_file_read` | `True` | Allow AI code to read files |
| `ai_allow_network` | `False` | Allow network access |
| `ai_allow_subprocess` | `False` | Allow subprocess/os.system calls |
| `ai_allow_os_access` | `False` | Allow direct OS access |
| `ai_allowed_directories` | `''` | Comma-separated allowed paths |
| `ai_require_approval` | `True` | Require user approval before execution |

## ROI Types

All ROI types inherit from `ROI_Base` in `flika.roi`:

| Class | Kind String | Description |
|---|---|---|
| `ROI_rectangle` | `'rectangle'` | Rectangular region |
| `ROI_ellipse` | `'ellipse'` | Elliptical region |
| `ROI_line` | `'line'` | Line segment |
| `ROI_freehand` | `'freehand'` | Freeform polygon |
| `ROI_point` | `'point_roi'` | Single-pixel crosshair |
| `ROI_rect_line` | `'rect_line'` | Multi-segment line with width |
| `ROI_center_surround` | `'center_surround'` | Concentric inner/outer regions |

### Common ROI Methods

```python
roi = g.win.rois[0]

# Get the time trace
trace = roi.getTrace()       # 1D array, length = number of frames

# Get the binary mask
mask = roi.getMask()          # 2D boolean array

# Get position and bounds
pos = roi.pos()
# ROI kind
print(roi.kind)              # 'rectangle', 'ellipse', etc.
```

## Macro Recording

Flika records all operations as macro commands for reproducibility.

```python
from flika.app.macro_recorder import MacroRecorder

recorder = MacroRecorder.instance()

# Get the recorded commands
commands = recorder.commands

# Save macro to file
recorder.save('/path/to/macro.py')

# Replay a macro
exec(open('/path/to/macro.py').read())
```

## Undo/Redo

The undo system uses a stack of `ProcessCommand` objects.

```python
from flika.core.undo import undo_stack

undo_stack.undo()   # Undo last operation
undo_stack.redo()   # Redo last undone operation
undo_stack.clear()  # Clear undo history (frees memory)
```

## Console Scripting Examples

### Batch Apply a Filter

```python
from flika.process.filters import gaussian_blur
for win in g.windows:
    g.win = win
    gaussian_blur(sigma=1.5)
```

### Extract ROI Data

```python
import numpy as np
traces = np.array([roi.getTrace() for roi in g.win.rois])
np.savetxt('traces.csv', traces.T, delimiter=',')
```

### Create a Montage

```python
import numpy as np
from flika.window import Window

images = [w.image[0] for w in g.windows[:4]]  # First frame of 4 windows
row1 = np.hstack(images[:2])
row2 = np.hstack(images[2:])
montage = np.vstack([row1, row2])
Window(montage, name='Montage')
```

### Measure Across Windows

```python
for w in g.windows:
    img = w.image
    print(f"{w.name}: mean={img.mean():.2f}, std={img.std():.2f}, "
          f"shape={img.shape}")
```

### Run a Simulation and Benchmark Detection

```python
from flika.process.simulation import simulate
from flika.spt.detection.utrack_detector import UTrackDetector

# Generate synthetic data with known ground truth
w = simulate.run(preset='TIRF - Single Molecules')

# Detect particles
detector = UTrackDetector(psf_sigma=1.5, alpha=0.05)
locs = detector.detect_stack(w.image)

# Compare against ground truth
gt_positions = w.metadata['simulation']['positions']
print(f"Detected {len(locs)} localizations, {len(gt_positions)} true emitters")
```

### Calcium Imaging Workflow

```python
from flika.process.calcium import compute_dff_image, detect_calcium_events

# Compute dF/F
dff = compute_dff_image(g.win.image, baseline_frames=range(0, 50))

# Draw an ROI, then extract and analyze its trace
trace = g.win.rois[0].getTrace()
events = detect_calcium_events(trace, threshold=2.5, min_duration=3)
print(f"Found {len(events)} calcium events")
```

## See Also

- [Plugins](plugins.md) -- Writing plugins with BaseProcess
- [ROI Guide](roi_guide.md) -- Interactive ROI usage
- [Keyboard Shortcuts](keyboard_shortcuts.md) -- Keyboard reference
- [Process Menu](process_menu.md) -- All built-in operations
