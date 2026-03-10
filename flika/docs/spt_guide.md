# Single Particle Tracking (SPT) Guide

Flika includes a full single particle tracking pipeline for detecting, linking, analyzing,
and classifying particle trajectories in time-lapse microscopy data.

## Overview

The SPT workflow consists of four stages:

1. **Detection** -- Find particles in each frame
2. **Linking** -- Connect detections across frames into tracks
3. **Analysis** -- Compute features (MSD, diffusion, geometry)
4. **Classification** -- Categorize tracks by motion type

Access the SPT tools via **Process > SPT Analysis** or the SPT Control Panel.

## SPT Control Panel

**Process > SPT Analysis > SPT Control Panel** opens a 6-tab dockable panel:

| Tab | Purpose |
|---|---|
| **Detection** | Configure and run particle detection |
| **Linking** | Configure and run particle linking |
| **Analysis** | Compute track features and statistics |
| **Classification** | Classify tracks by motion type |
| **Batch** | Run SPT pipeline on multiple files |
| **Visualization** | Track display and plotting options |

## Detection Methods

### U-Track Detector

A robust multi-scale Gaussian fitting detector. Parameters:

| Parameter | Description |
|---|---|
| Sigma | Expected particle sigma (in pixels) |
| Alpha | Statistical significance threshold |
| Max iterations | Maximum fitting iterations |

```python
from flika.process.spt import detect_particles
detect_particles()  # Opens the detection dialog
```

### ThunderSTORM Detector

Compatible with ThunderSTORM-style detection for super-resolution data. Uses wavelet
filtering followed by local maximum detection and sub-pixel fitting.

## Linking Methods

### Greedy Linker

Simple nearest-neighbor linking. Fast but may produce errors in dense samples.

| Parameter | Description |
|---|---|
| Max distance | Maximum linking distance between frames (pixels) |
| Max gap | Maximum number of frames a track can skip |

### U-Track Linker (LAP)

Linear Assignment Problem (LAP) based linking with Kalman filtering. The recommended
method for most applications.

| Parameter | Description |
|---|---|
| Max distance | Maximum linking distance |
| Max gap | Maximum gap closing distance (frames) |
| Kalman filter | Enable motion prediction for better linking |

```python
from flika.process.spt import link_particles_process
link_particles_process()  # Opens the linking dialog
```

### Trackpy Linker

Adapter for the trackpy library's linking algorithm. Useful for compatibility with
existing trackpy workflows.

## Track Data Format

Track data is stored in `window.metadata['spt']` as a dictionary:

```python
spt_data = g.win.metadata['spt']

# Particle data: DataFrame with columns [frame, x, y, intensity, ...]
particles = spt_data['particle_data']

# Track dict: {track_id: (N, 3) array of [frame, x, y]}
tracks = spt_data.get('tracks', {})
```

## Feature Analysis

The Analysis tab computes quantitative features for each track.

### Geometric Features

| Feature | Description |
|---|---|
| Radius of gyration (Rg) | Spatial spread of the track |
| Asymmetry | Deviation from circular symmetry |
| Fractal dimension | Complexity measure of the track path |

### Kinematic Features

| Feature | Description |
|---|---|
| MSD curve | Mean squared displacement vs. lag time |
| Diffusion coefficient | From MSD fitting (D = MSD / 4dt) |
| Velocity | Instantaneous and average velocity |
| Direction | Angular direction of motion |
| Confinement ratio | End-to-end distance / total path length |

### Spatial Features

| Feature | Description |
|---|---|
| Nearest-neighbor distance | Distance to nearest concurrent track |

### Autocorrelation

Velocity autocorrelation analysis for detecting directed vs. diffusive motion.

```python
# Access computed features
from flika.spt.features.feature_calculator import calculate_features
features = calculate_features(tracks, pixel_size=0.108, frame_interval=0.05)
```

## Classification

The Classification tab uses an SVM pipeline to categorize tracks by motion type:

1. Features are normalized with Box-Cox transformation
2. PCA reduces dimensionality
3. RBF-kernel SVM classifies tracks

Common motion categories:
- **Brownian** -- Free diffusion
- **Confined** -- Restricted to a small area
- **Directed** -- Active transport with persistent direction
- **Anomalous** -- Sub- or super-diffusive behavior

## Batch Processing

The Batch tab processes multiple files through the full SPT pipeline.

### Expert Configs

Six preset configurations for common experimental scenarios:

| Preset | Use Case |
|---|---|
| Default | General-purpose settings |
| Fast diffusion | Rapidly moving particles |
| Slow diffusion | Nearly stationary particles |
| Dense | High particle density |
| Super-resolution | SMLM-style data |
| Custom | User-defined parameters |

### Running Batch

1. Select input folder with image files
2. Choose an expert config or customize parameters
3. Set output directory
4. Click Run -- processing happens in a background QThread

```python
from flika.spt.batch.batch_pipeline import run_batch
run_batch(input_dir='/path/to/data/', config='default', output_dir='/path/to/output/')
```

## Results Table

**Process > SPT Analysis > Results Table** opens a dockable table showing particle
detection results. Columns include frame, x, y, intensity, sigma, and track ID
(after linking).

## Visualization

### Track Overlay

**Image > Overlay > Track Overlay** draws colored particle tracks directly on the image.
Each track is drawn as a colored path following the particle across frames.

### Track Window

Click on a track in the results table or overlay to open a per-track analysis window
with 6 panels:

| Panel | Content |
|---|---|
| Track path | XY trajectory |
| Intensity | Intensity vs. time |
| MSD | Mean squared displacement plot |
| Velocity | Speed vs. time |
| Direction | Angular direction vs. time |
| Displacement | Step size distribution |

### Diffusion Plot

Dedicated MSD analysis and fitting window:

- MSD vs. lag time with fitting
- 1, 2, or 3-component CDF fitting for heterogeneous populations
- Diffusion coefficient extraction

### Flower Plot

Origin-centered display of multiple tracks overlaid. All tracks are translated so they
start at (0, 0), revealing the distribution of track shapes and extents.

### All Tracks Plot

Multi-track intensity extraction and heatmap view. Shows intensity time series for all
tracks as a heatmap, sorted by track duration or other properties.

### Chart Dock

General-purpose scatter plot and histogram dock for visualizing feature distributions
across all tracks.

## SPT I/O

### Import Formats

| Format | Source |
|---|---|
| ThunderSTORM CSV | ThunderSTORM ImageJ plugin output |
| Flika CSV | Flika's native SPT format |
| JSON | Structured JSON with metadata |

### Export

Results can be exported from the Results Table or programmatically:

```python
from flika.spt.io.spt_formats import save_thunderstorm_csv
save_thunderstorm_csv(particle_data, '/path/to/output.csv')
```

## See Also

- [Process Menu](process_menu.md#spt-analysis) -- Menu entries for SPT
- [Image Menu](image_menu.md#track-overlay) -- Track overlay
- [AI Tools](ai_tools.md#particle-localizer) -- AI-assisted detection
