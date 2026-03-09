# Microscopy Simulation

Flika includes a comprehensive microscopy simulation system for generating synthetic image
data with realistic noise, optics, and fluorophore behavior. Simulated data is useful for
algorithm benchmarking, teaching, and validating analysis pipelines against known ground truth.

---

## Simulation Pipeline

The simulation follows a staged pipeline:

1. **Sample** -- Define biological structures (beads, filaments, cells, organelles) and
   place fluorophore labels at specified densities.
2. **Fluorophore** -- Assign photophysical properties (brightness, blinking rates,
   bleaching lifetime) from built-in presets or custom parameters.
3. **Optics / PSF** -- Apply a point spread function model and an optical modality
   (widefield, TIRF, confocal, etc.) to produce the raw image.
4. **Camera / Noise** -- Simulate shot noise (Poisson), read noise (Gaussian), dark
   current, and detector-specific characteristics (CCD, EMCCD, sCMOS).
5. **Output** -- Deliver the resulting image stack as a new Flika window with ground
   truth stored in metadata.

---

## Accessing the Simulation System

### Simulation Builder Dialog

**Simulation > Simulation Builder...**

The Simulation Builder is a four-tab dialog for configuring every aspect of the simulation:

| Tab | Contents |
|---|---|
| **Quick Start** | Select a named preset, preview a description, and generate data with one click. |
| **Sample** | Choose structure type, density, labeling parameters, and fluorophore preset. |
| **Microscope** | Select optical modality, PSF model, numerical aperture, wavelength, and camera model. |
| **Output** | Set image dimensions (X, Y, T), pixel size, frame interval, bit depth, and output format. |

### Quick Presets Submenu

**Simulation > Quick Presets**

Ten ready-made configurations for common imaging scenarios:

| Preset | Description |
|---|---|
| **Beads** | Sparse fluorescent beads for PSF calibration and resolution testing |
| **TIRF** | Surface-proximal molecules under total internal reflection illumination |
| **Confocal** | Optical-sectioned volume with pinhole-filtered out-of-focus light |
| **PALM/STORM** | Single-molecule localization with stochastic blinking (Alexa647) |
| **DNA-PAINT** | Transient binding-based super-resolution with predictable kinetics |
| **Light-sheet** | Selective plane illumination of a thick sample volume |
| **FLIM** | Fluorescence lifetime imaging with exponential decay curves |
| **Calcium** | GCaMP6f calcium transients and propagating waves in cell fields |
| **SIM** | Structured illumination microscopy with patterned excitation |
| **Tracking** | Multiple diffusing particles for single particle tracking benchmarks |

---

## Scripting Interface

Run simulations from the scripting console or macro scripts:

```python
from flika.process.simulation import simulate

# Use a named preset
simulate.run(preset='beads')

# Or pass custom parameters
simulate.run(
    preset='tracking',
    nx=256, ny=256, nt=100,
    pixel_size=0.1, frame_interval=0.05,
)
```

The `simulate` object is a `BaseProcess_noPriorWindow` subclass, so it integrates with
undo/redo and macro recording like any other Flika process.

---

## PSF Models

The simulation supports five point spread function models, selected per-simulation in the
Microscope tab:

| Model | Description |
|---|---|
| **Gaussian** | Simple 2D/3D Gaussian approximation; fast, suitable for most benchmarks. |
| **Airy** | Airy disk pattern from scalar diffraction; accurate for in-focus widefield. |
| **Born-Wolf** | Full scalar diffraction integral including defocus; good for 3D stacks. |
| **Vectorial** | High-NA vectorial diffraction model; accounts for polarization effects. |
| **Astigmatic** | Elliptical PSF with z-dependent asymmetry for astigmatism-based 3D localization. |

All PSF functions live in `flika/simulation/psf.py` and accept numerical aperture,
emission wavelength, refractive index, and pixel size as parameters.

---

## Camera Models

Three detector types are available in `flika/simulation/noise.py`, configured through the
`CameraConfig` dataclass:

| Parameter | CCD | EMCCD | sCMOS |
|---|---|---|---|
| **Quantum Efficiency** | 0.5--0.7 | 0.9--0.95 | 0.7--0.8 |
| **Read Noise (e-)** | 5--10 | ~0 (with EM gain) | 1--2 |
| **Dark Current (e-/s)** | 0.01 | 0.01 | 0.01 |
| **EM Gain** | N/A | 100--300 | N/A |

The camera model applies shot noise (Poisson statistics on photon counts), adds dark
current, applies gain, and adds read noise to produce realistic detector output.

---

## Biological Structures

Structure generators in `flika/simulation/structures.py`:

- **Beads** -- Random or grid-placed point emitters at configurable density and brightness.
- **Filaments** -- Curved or branching linear structures mimicking cytoskeletal networks.
- **Cells** -- Elliptical cell bodies with optional nucleus and membrane labeling.
- **Organelles** -- Small punctate or tubular structures distributed within cell boundaries.
- **Cell Fields** -- Multiple cells tiled across the field of view with random variation.

---

## Fluorophore Presets

Ten built-in fluorophore presets in `flika/simulation/fluorophores.py`, each defining
brightness, on/off rates, and bleaching lifetime:

| Preset | Typical Use |
|---|---|
| **EGFP** | General live-cell imaging |
| **mCherry** | Red channel live-cell imaging |
| **Alexa488** | Immunofluorescence, green channel |
| **Alexa647** | STORM / dSTORM super-resolution |
| **Atto655** | STED, far-red channel |
| **GCaMP6f** | Fast calcium indicator |
| **GCaMP6s** | Slow calcium indicator (higher sensitivity) |
| **DAPI** | Nuclear stain, blue channel |
| **Hoechst** | Live-cell nuclear stain |
| **MitoTracker** | Mitochondria labeling |

The blinking and bleaching simulation stochastically transitions each emitter between
fluorescent, dark, and bleached states at each time step.

---

## Optical Modalities

Modality models in `flika/simulation/optics.py` modify how fluorescence from the sample
reaches the detector:

| Modality | Key Behavior |
|---|---|
| **Widefield** | Uniform illumination; all planes contribute to image. |
| **TIRF** | Evanescent wave excitation; only ~100 nm above the coverslip is illuminated. |
| **Confocal** | Pinhole rejects out-of-focus light; optical sectioning. |
| **Light-sheet** | Selective plane illumination; low phototoxicity for thick samples. |
| **SIM** | Patterned excitation with frequency-space reconstruction for 2x resolution. |
| **STED** | Depletion beam shrinks the effective PSF below the diffraction limit. |
| **SMLM** | Single-molecule localization via stochastic activation (PALM/STORM). |
| **DNA-PAINT** | Transient DNA hybridization produces blinking without photoactivation. |
| **FLIM** | Time-resolved detection; generates fluorescence lifetime decay data. |

---

## Dynamics

Dynamic processes in `flika/simulation/dynamics.py` animate the sample over time:

### Diffusion Models

- **Brownian** -- Free isotropic diffusion with specified diffusion coefficient D.
- **Directed** -- Diffusion with a constant drift velocity.
- **Confined** -- Diffusion within a circular or rectangular confinement region.
- **Anomalous** -- Sub- or super-diffusive motion parameterized by anomalous exponent alpha.
- **Switching** -- Particles switch stochastically between two diffusion states.

### Biological Dynamics

- **Calcium Transients** -- Rapid rise and exponential decay mimicking single-cell calcium events.
- **Calcium Waves** -- Propagating calcium signals across a cell field with configurable speed.
- **DNA-PAINT Binding** -- Stochastic association/dissociation kinetics at binding sites.
- **FLIM Decay** -- Mono- or bi-exponential fluorescence lifetime decay per pixel.

---

## Ground Truth

Every simulation stores complete ground truth in `window.metadata['simulation']`:

| Key | Contents |
|---|---|
| `positions_per_frame` | List of (x, y) emitter positions for each frame |
| `track_dict` | Dictionary mapping track ID to (N, 3) arrays of [frame, x, y] |
| `binary_mask` | Boolean mask of labeled regions |
| `segmentation_mask` | Integer-labeled segmentation mask |
| `diffusion_coefficients` | Per-particle diffusion coefficient values |
| `calcium_trace` | Ground truth dF/F time course |
| `calcium_spike_frames` | Frame indices of true calcium events |
| `emitter_states` | Per-emitter fluorescent/dark/bleached state at each frame |

Ground truth conversion utilities are in `flika/simulation/ground_truth.py`:

- `positions_to_detection_table()` -- Convert positions to a detection DataFrame.
- `trajectories_to_track_dict()` -- Convert trajectory arrays to Flika SPT track format.

---

## Benchmarking

The benchmarking system in `flika/simulation/benchmarks.py` and
`flika/simulation/evaluation.py` provides quantitative evaluation of analysis algorithms
against simulation ground truth.

### Available Benchmarks

Eight registered benchmarks covering four analysis domains:

| Domain | Benchmarks | Metrics |
|---|---|---|
| **Detection** | 3 benchmarks (sparse, dense, low SNR) | Precision, Recall, F1, RMSE of localization |
| **Tracking** | 2 benchmarks (simple, complex) | MOTA, MOTP, track fragmentation |
| **Segmentation** | 2 benchmarks (binary, multi-label) | Jaccard index, Dice coefficient |
| **Calcium** | 1 benchmark | Event detection F1, timing error |

### Benchmark Dialog

**Simulation > Benchmarks...**

The `BenchmarkDialog` provides a tree-based benchmark selector, a results table, and
export options (plain text and JSON). Run individual benchmarks or the full suite with
`run_benchmark()` and `run_all_benchmarks()`.

### Evaluation Metrics

Evaluation functions in `flika/simulation/evaluation.py`:

- **Detection** -- Hungarian matching between predicted and ground truth positions;
  reports precision, recall, F1, and localization RMSE.
- **Tracking** -- MOTA (Multiple Object Tracking Accuracy) and MOTP (Precision) following
  the MOT challenge conventions.
- **Segmentation** -- Jaccard (IoU) and Dice coefficients computed per-label and averaged.
- **Calcium** -- Event-level F1 score with configurable timing tolerance window.

Results are returned as `BenchmarkResult` dataclass instances for programmatic access.
